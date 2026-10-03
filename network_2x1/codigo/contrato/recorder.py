import os, time, math

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
import numpy as np
import h5py
import jax, jax.numpy as jnp
from jax import vmap
import solver_jee as SJ
import contract_io as CIO

_SOLVER = os.environ.get("GRAVADOR_SOLVER", "thomas").lower()
if _SOLVER == "pcr":
    import solver_jax_pcr as SX
else:
    import solver_jax as SX

SEED0 = int(os.environ.get("GRAVADOR_SEED0", 0))
N_SEEDS = int(os.environ.get("GRAVADOR_NSEEDS", 2000))
B_BLOCO = int(os.environ.get("GRAVADOR_BBLOCO", 8192))
OUT_DIR = os.environ.get("GRAVADOR_OUTDIR", "dados_brutos")
Y0_LO = float(os.environ.get("GRAVADOR_Y0LO", 0.50))
Y0_HI = float(os.environ.get("GRAVADOR_Y0HI", 0.75))
COURANT = 0.4
NT_CAP = 12000
TF_CAP = 3600.0
NX = 71
NMAX = CIO.NMAX
TREF = SJ.TREF
R = SJ.R

LOG_PARAMS = {"B_ref", "kL", "dp", "Dm"}
os.makedirs(OUT_DIR, exist_ok=True)
ARQUIVO = "%s/dataset_2comp_1leito_s%d-%d.h5" % (OUT_DIR, SEED0, SEED0 + N_SEEDS - 1)


def _fmt(seg):
    """seconds -> 'Xm Ys' or 'Xh Ym' for ETA prints."""
    seg = max(0.0, float(seg))
    if seg < 90:
        return "%.0fs" % seg
    if seg < 5400:
        return "%dm %02ds" % (int(seg // 60), int(seg % 60))
    return "%dh %02dm" % (int(seg // 3600), int((seg % 3600) // 60))


def sample_cfg_22(seed):
    rng = np.random.default_rng(seed)
    XR = CIO.X_RANGES

    def draw(name, n=1):
        lo, hi = XR[name]
        if name in LOG_PARAMS:
            v = 10 ** rng.uniform(np.log10(lo), np.log10(hi), n)
        else:
            v = rng.uniform(lo, hi, n)
        return v

    qm_ref = draw("qm_ref", NMAX)
    k2 = draw("k2", NMAX)
    B_ref = draw("B_ref", NMAX)
    k4 = draw("k4", NMAX)
    n_ref = draw("n_ref", NMAX)
    k6 = draw("k6", NMAX)
    kL = draw("kL", NMAX)
    dH = draw("dH", NMAX)
    Cpg = draw("Cpg", NMAX)
    k1 = qm_ref - k2 * TREF
    k3 = B_ref * np.exp(-k4 / TREF)
    k5 = n_ref - k6 / TREF
    order = np.argsort(B_ref)

    def ro(a):
        return a[order]

    qm_ref, k2, B_ref, k4, n_ref, k6, kL, dH, k1, k3, k5, Cpg = (
        ro(qm_ref),
        ro(k2),
        ro(B_ref),
        ro(k4),
        ro(n_ref),
        ro(k6),
        ro(kL),
        ro(dH),
        ro(k1),
        ro(k3),
        ro(k5),
        ro(Cpg),
    )
    layer = dict(
        k1=k1,
        k2=k2,
        k3=k3,
        k4=k4,
        k5=k5,
        k6=k6,
        Bref=B_ref,
        om=kL,
        dH=dH,
        Cpg=Cpg,  # v22
        eb=float(draw("eb")[0]),
        rho_b=float(draw("rho_b")[0]),
        Cps=float(draw("Cps")[0]),
    )
    yc = float(rng.uniform(Y0_LO, Y0_HI))
    y = np.array([yc, 1.0 - yc])
    return dict(
        layers=[layer],
        y=y,
        carrier=0,
        vs=float(draw("vs")[0]),
        Tin=float(draw("Tin")[0]),
        P=float(draw("P")[0]),
        L=float(draw("L")[0]),
        hw=float(draw("hw")[0]),
        lam=float(draw("lam")[0]),
        dp=float(draw("dp")[0]),
        Dm=float(draw("Dm")[0]),
        Dt=float(draw("Dt")[0]),
    )  # v22


def nt_da_seed(seed):
    S = SJ.setup(sample_cfg_22(seed), NX, tf_cap=TF_CAP)
    nt = S["TF"] * S["u_int"] / (COURANT * S["h"])
    return int(min(NT_CAP, math.ceil(nt)))


def gera_bloco(seeds, NTblk):
    B = len(seeds)
    cfgs = [sample_cfg_22(s) for s in seeds]
    Ss = [SJ.setup(cfg, NX, tf_cap=TF_CAP) for cfg in cfgs]
    pk = ["K1", "K2", "K3", "K4", "K5", "K6", "dH", "om"]
    K = [jnp.asarray(np.stack([S["par"][k] for S in Ss])) for k in pk]

    def stk(key):
        return jnp.asarray(np.stack([np.asarray(S[key]) for S in Ss]))

    def scl(key):
        return jnp.asarray(np.array([float(S[key]) for S in Ss]))

    eb = stk("eb")
    rho_b = stk("rho_b")
    Cps = stk("Cps")
    Wsink = stk("Wsink")
    Ve = stk("Ve")
    De = jnp.asarray(np.array([float(S["De"][0]) for S in Ss]))
    cfeed = stk("cfeed")
    Cpmol = stk("Cpmol")
    Vz = scl("Vz")
    kap = scl("kap")
    lam = scl("lam")
    Tin = scl("Tin")
    hh = scl("h")
    TFb = np.array([S["TF"] for S in Ss])
    dt = jnp.asarray(TFb / NTblk)
    c0b = jnp.asarray(np.stack([S["c0"] for S in Ss]))
    q0b = jnp.asarray(np.stack([S["q0"] for S in Ss]))
    T0b = jnp.asarray(np.stack([S["T0"] for S in Ss]))
    Pb = (
        K[0],
        K[1],
        K[2],
        K[3],
        K[4],
        K[5],
        K[6],
        K[7],
        eb,
        rho_b,
        Cps,
        Wsink,
        De,
        Ve,
        dt,
        cfeed,
        Vz,
        kap,
        lam,
        Tin,
        Tin,
        hh,
        Cpmol,
    )
    solve = jax.jit(
        vmap(SX.make_solver(NMAX, NX, NTblk), in_axes=(0, 0, 0, tuple(0 for _ in range(23))))
    )
    yt, Tt, qz, Tz, cz = solve(c0b, q0b, T0b, Pb)
    yt.block_until_ready()
    yt = np.asarray(yt)
    Tt = np.asarray(Tt)
    qz = np.asarray(qz)
    Tz = np.asarray(Tz)

    c_out = yt
    tot = np.maximum(c_out.sum(axis=2, keepdims=True), 1e-30)
    y_out = c_out / tot

    N_ads = qz.mean(axis=2).astype(np.float32)

    X = np.stack([CIO.vetor_X_v3(c) for c in cfgs]).astype(np.float32)
    valid = np.zeros(B, np.int8)
    dTmax = np.zeros(B, np.float32)
    cauda_inc = np.array([1 if Ss[b]["cauda_inc"] else 0 for b in range(B)], np.int8)
    tbreak = np.full((B, NMAX), np.nan, np.float32)
    tsat = np.full((B, NMAX), np.nan, np.float32)
    for b in range(B):
        fin = np.all(np.isfinite(c_out[b])) and np.all(np.isfinite(Tz[b]))
        dTmax[b] = (max(Tz[b].max(), Tt[b].max()) - cfgs[b]["Tin"]) if fin else np.nan
        cf = cfgs[b]["y"]
        ctot = Ss[b]["c_tot"]
        carrier = cfgs[b]["carrier"]
        tgrid = np.linspace(0, Ss[b]["TF"], NTblk)
        broke = False
        for i in range(NMAX):
            if i == carrier:
                continue
            yi = c_out[b, :, i] / (cf[i] * ctot + 1e-30)  # c_i/c_i_feed
            idx = np.where(yi >= 0.05)[0]
            if len(idx):
                tbreak[b, i] = tgrid[idx[0]]
                broke = True
            ids = np.where(yi >= 0.95)[0]
            if len(ids):
                tsat[b, i] = tgrid[ids[0]]
        valid[b] = 1 if (fin and broke) else 0
    return dict(
        X=X,
        c_out=c_out.astype(np.float32),
        y_out=y_out.astype(np.float32),
        T_out=Tt.astype(np.float32),
        Tz=Tz.astype(np.float32),
        qz=qz.astype(np.float32),
        N_ads=N_ads,
        tbreak=tbreak,
        tsat=tsat,
        valid=valid,
        dTmax=dTmax.astype(np.float32),
        cauda_inc=cauda_inc,
        TF=TFb.astype(np.float32),
        NT=NTblk,
        seeds=np.array(seeds, np.int64),
    )


def main():
    print(
        "device:",
        jax.devices(),
        "| solver:",
        _SOLVER,
        "| target:",
        N_SEEDS,
        "seeds 2c/1bed ->",
        ARQUIVO,
        flush=True,
    )
    t0 = time.time()


    resume_from = 0
    modo = "w"
    if os.path.exists(ARQUIVO):
        try:
            with h5py.File(ARQUIVO, "r") as fchk:
                if "feito" in fchk.attrs and int(fchk.attrs.get("n_seeds", -1)) == N_SEEDS:
                    resume_from = int(fchk.attrs["feito"])
                    modo = "a"
                    print(
                        "[RESUME] existing file, %d seeds already recorded. Continuing."
                        % resume_from,
                        flush=True,
                    )
        except Exception as e:
            print("[RESUME] existing file unreadable (%s); restarting from zero." % e, flush=True)
            modo = "w"

    with h5py.File(ARQUIVO, modo) as f:
        f.attrs.update(
            dict(
                config="2comp_1leito",
                NX=NX,
                courant=COURANT,
                seed0=SEED0,
                n_seeds=N_SEEDS,
                nmax=NMAX,
                tf_cap=TF_CAP,
            )
        )
        vlf = h5py.special_dtype(vlen=np.float32)
        if modo == "w":
            dsc = f.create_dataset("c_out", (N_SEEDS, NMAX), dtype=vlf)
            dsy = f.create_dataset("y_out", (N_SEEDS, NMAX), dtype=vlf)
            dst = f.create_dataset("T_out", (N_SEEDS,), dtype=vlf)
            f.create_dataset("X_v2", (N_SEEDS, CIO.X_DIM), dtype=np.float32)
            f.create_dataset("Tz", (N_SEEDS, NX), dtype=np.float32)
            f.create_dataset("qz", (N_SEEDS, NMAX, NX), dtype=np.float32)
            f.create_dataset("N_ads", (N_SEEDS, NMAX), dtype=np.float32)
            f.create_dataset("tbreak", (N_SEEDS, NMAX), dtype=np.float32)
            f.create_dataset("tsat", (N_SEEDS, NMAX), dtype=np.float32)
            f.create_dataset("valid", (N_SEEDS,), dtype=np.int8)
            f.create_dataset("cauda_inc", (N_SEEDS,), dtype=np.int8)
            f.create_dataset("dTmax", (N_SEEDS,), dtype=np.float32)
            f.create_dataset("TF", (N_SEEDS,), dtype=np.float32)
            f.create_dataset("NT", (N_SEEDS,), dtype=np.int32)
            f.create_dataset("seeds", (N_SEEDS,), dtype=np.int64)
            f.attrs["feito"] = 0
        else:
            dsc = f["c_out"]
            dsy = f["y_out"]
            dst = f["T_out"]


        print("[1/2] pre-computing NT per seed (Courant=%.2f)..." % COURANT, flush=True)
        all_seeds = list(range(SEED0, SEED0 + N_SEEDS))
        nts = np.empty(N_SEEDS, np.int32)
        tpc = time.time()
        passo = max(1, N_SEEDS // 20)
        for k, sd in enumerate(all_seeds):
            nts[k] = nt_da_seed(sd)
            if (k + 1) % passo == 0 or (k + 1) == N_SEEDS:
                el = time.time() - tpc
                taxa = (k + 1) / max(el, 1e-9)
                eta = (N_SEEDS - (k + 1)) / max(taxa, 1e-9)
                print(
                    "   NT %d/%d  (%.0f seeds/s, elapsed %s, ETA %s)"
                    % (k + 1, N_SEEDS, taxa, _fmt(el), _fmt(eta)),
                    flush=True,
                )
        order = np.argsort(nts)
        sorted_seeds = [all_seeds[i] for i in order]
        nt_map = {all_seeds[i]: int(nts[i]) for i in range(N_SEEDS)}
        print(
            "   NT: min=%d  median=%d  max=%d  (cap=%d)"
            % (nts.min(), int(np.median(nts)), nts.max(), NT_CAP),
            flush=True,
        )


        nb = math.ceil(N_SEEDS / B_BLOCO)
        done = 0
        print(
            "[2/2] generating %d blocks of up to %d seeds (1st block of each NT compiles)..."
            % (nb, B_BLOCO),
            flush=True,
        )
        tg0 = time.time()
        for blk in range(nb):
            seeds = sorted_seeds[blk * B_BLOCO : min((blk + 1) * B_BLOCO, N_SEEDS)]
            B = len(seeds)
            if done + B <= resume_from:
                done += B
                continue
            NTblk = int(min(NT_CAP, max(nt_map[s] for s in seeds)))
            tb0 = time.time()
            out = gera_bloco(seeds, NTblk)
            sl = slice(done, done + B)
            for b in range(B):
                dsc[done + b] = [out["c_out"][b, :, i] for i in range(NMAX)]
                dsy[done + b] = [out["y_out"][b, :, i] for i in range(NMAX)]
                dst[done + b] = out["T_out"][b]
            f["X_v2"][sl] = out["X"]
            f["Tz"][sl] = out["Tz"]
            f["qz"][sl] = out["qz"]
            f["N_ads"][sl] = out["N_ads"]
            f["tbreak"][sl] = out["tbreak"]
            f["tsat"][sl] = out["tsat"]
            f["valid"][sl] = out["valid"]
            f["dTmax"][sl] = out["dTmax"]
            f["cauda_inc"][sl] = out["cauda_inc"]
            f["TF"][sl] = out["TF"]
            f["NT"][sl] = out["NT"]
            f["seeds"][sl] = out["seeds"]
            done += B
            f.attrs["feito"] = done
            f.flush()
            dtb = time.time() - tb0
            vr = out["valid"].mean() * 100
            eta = (N_SEEDS - done) * (dtb / B)
            print(
                "   block %d/%d  seeds %d..%d  NT=%d  valid=%.1f%%  %.1fs (%.2f ms/seed)"
                "  total %d/%d  ETA ~%s"
                % (
                    blk + 1,
                    nb,
                    seeds[0],
                    seeds[-1],
                    out["NT"],
                    vr,
                    dtb,
                    1000 * dtb / B,
                    done,
                    N_SEEDS,
                    _fmt(eta),
                ),
                flush=True,
            )
    print("DONE in %s  ->  %s" % (_fmt(time.time() - t0), ARQUIVO), flush=True)


if __name__ == "__main__":
    main()