import os, sys, glob, time
import numpy as np
import h5py

sys.path.insert(0, os.getcwd())
import src.contrato.contract_io as CIO
import src.contrato.features as FEAT

USAR_FEATURES = os.environ.get("USAR_FEATURES", "1") == "1"

M = int(os.environ.get("M", 100))
FRAC_DENSA = float(os.environ.get("FRAC_DENSA", 0.7))
MARGEM = float(os.environ.get("MARGEM", 0.05))
SAIDA = os.environ.get("SAIDA", "pre.npz")
CHUNK = int(os.environ.get("CHUNK", 4000))
COMP = 1  # strong component; the carrier (comp0) does roll-up and does not break through


def grid_tau(tbreak_f, tsat_f):
    t0 = max(0.0, tbreak_f - MARGEM)
    t1 = tsat_f if np.isfinite(tsat_f) else 1.0
    t1 = min(max(t1, t0 + 1e-3), 1.0)
    n_densa = int(round(FRAC_DENSA * M))
    n_pre = max(2, int(0.10 * M))
    n_pos = M - n_densa - n_pre
    if n_pos < 0:
        n_densa += n_pos
        n_pos = 0
    pre = np.linspace(0.0, t0, n_pre, endpoint=False)
    densa = np.linspace(t0, t1, n_densa, endpoint=(n_pos == 0))
    pos = np.linspace(t1, 1.0, n_pos + 1)[1:] if n_pos > 0 else np.array([])
    tau = np.concatenate([pre, densa, pos])
    tau = np.clip(np.sort(tau), 0.0, 1.0)
    if len(tau) != M:
        tau = np.linspace(0, 1, M)
    return tau


def reamostra(y, tau_grid):
    nt = len(y)
    t_orig = np.linspace(0.0, 1.0, nt)
    return np.interp(tau_grid, t_orig, y)


def acha_arquivo():
    p = [a for a in sys.argv[1:] if not a.startswith("-")]
    if p:
        return p[0]
    c = sorted(glob.glob("dados/dataset_2comp_1leito_LIMPO.h5"))
    if not c:
        c = sorted(glob.glob("dados/dataset_2comp_1leito_*.h5"))
    if not c:
        sys.exit("Provide the file (preferably the CLEAN one).")
    return c[-1]


def main():
    arq = acha_arquivo()
    print("File:", arq, "| M =", M, "| dense frac =", FRAC_DENSA, flush=True)
    t0 = time.time()
    with h5py.File(arq, "r") as f:
        N = f["X_v2"].shape[0]
        NX = f["Tz"].shape[1]
        X = f["X_v2"][:].astype(np.float32)
        if USAR_FEATURES:
            X = FEAT.X_enriquecido(X, logar=True)
            print(
                "X enriched: 31 -> %d columns (+%d features)" % (X.shape[1], FEAT.N_FEAT),
                flush=True,
            )
        qz = f["qz"][:].astype(np.float32)
        Tz = f["Tz"][:].astype(np.float32)
        N_ads = f["N_ads"][:].astype(np.float32)
        TF = f["TF"][:].astype(np.float64)
        tbreak = f["tbreak"][:, COMP].astype(np.float64)
        tsat = f["tsat"][:, COMP].astype(np.float64)

        forma_y = np.zeros((N, M), np.float32)  # comp1 (strong)
        forma_y0 = np.zeros((N, M), np.float32)  # comp0 (carrier, roll-up)
        forma_T = np.zeros((N, M), np.float32)
        tau_all = np.zeros((N, M), np.float32)
        yfeed = np.zeros(N, np.float32)
        yfeed0 = np.zeros(N, np.float32)
        Tin_all = np.zeros(N, np.float32)

        pc = len(CIO.PARAM_COMP_NAMES)
        i_Tin = pc * CIO.NMAX + len(CIO.PROP_LAYER_NAMES) + CIO.GLOB_NAMES.index("Tin")

        print("Resampling %d seeds (shape yi comp1 + comp0 + T_out)..." % N, flush=True)
        passo = max(1, N // 20)
        for i in range(N):
            tb_f = tbreak[i] / TF[i] if TF[i] > 0 else 0.0
            ts_f = tsat[i] / TF[i] if (TF[i] > 0 and np.isfinite(tsat[i])) else np.nan
            tau = grid_tau(tb_f, ts_f)
            tau_all[i] = tau
            y = np.asarray(f["y_out"][i, COMP], np.float64)  # comp1
            y0 = np.asarray(f["y_out"][i, 0], np.float64)  # comp0 (carrier)
            T = np.asarray(f["T_out"][i], np.float64)
            yf = max(y.max(), 1e-6)
            yfeed[i] = yf
            yf0 = max(y0.max(), 1e-6)
            yfeed0[i] = yf0
            Tin_all[i] = X[i, i_Tin]
            forma_y[i] = reamostra(y / yf, tau)  # comp1 normalized
            forma_y0[i] = reamostra(y0 / yf0, tau)  # comp0 normalized
            forma_T[i] = reamostra(T, tau)
            if (i + 1) % passo == 0 or (i + 1) == N:
                el = time.time() - t0
                print("   %d/%d  (%.0f seeds/s)" % (i + 1, N, (i + 1) / max(el, 1e-9)), flush=True)

    os.makedirs(os.path.dirname(SAIDA) or ".", exist_ok=True)

    K_POD = int(os.environ.get("K_POD", 8))

    def base_pod(F):
        Fm = F.mean(0).astype(np.float32)
        Fc = F - Fm
        U, S, Vt = np.linalg.svd(Fc, full_matrices=False)
        modos = Vt[:K_POD].astype(np.float32)
        ev = S**2
        ev = ev / ev.sum()
        var_k = float(np.cumsum(ev)[K_POD - 1])
        coef = Fc @ modos.T
        return Fm, modos, coef.astype(np.float32), var_k

    ymean, ymodos, ycoef, yvar = base_pod(forma_y.astype(np.float64))
    y0mean, y0modos, y0coef, y0var = base_pod(forma_y0.astype(np.float64))
    Tmean, Tmodos, Tcoef, Tvar = base_pod(forma_T.astype(np.float64))
    print(
        "POD: %d modes -> yi(c1) %.2f%% | yi(c0 roll-up) %.2f%% | T_out %.2f%%"
        % (K_POD, 100 * yvar, 100 * y0var, 100 * Tvar),
        flush=True,
    )

    np.savez_compressed(
        SAIDA,
        X=X,
        forma_y=forma_y,
        forma_y0=forma_y0,
        forma_T=forma_T,
        qz=qz,
        Tz=Tz,
        N_ads=N_ads,
        TF=TF.astype(np.float32),
        tbreak=tbreak.astype(np.float32),
        tsat=tsat.astype(np.float32),
        tau_grid=tau_all,
        yfeed=yfeed,
        yfeed0=yfeed0,
        Tin=Tin_all,
        M=np.int32(M),
        NX=np.int32(NX),
        K_POD=np.int32(K_POD),
        y_pod_mean=ymean,
        y_pod_modos=ymodos,
        y_pod_coef=ycoef,
        y0_pod_mean=y0mean,
        y0_pod_modos=y0modos,
        y0_pod_coef=y0coef,
        T_pod_mean=Tmean,
        T_pod_modos=Tmodos,
        T_pod_coef=Tcoef,
    )
    print("\nCache saved: %s  (%d seeds, M=%d)" % (SAIDA, N, M))
    print(
        "  forma_y(c1):", forma_y.shape, "forma_y0(c0):", forma_y0.shape, "forma_T:", forma_T.shape
    )
    com_tsat = np.isfinite(tsat).mean() * 100
    print("  seeds with tsat: %.1f%%  (rest: dense grid until end)" % com_tsat)


if __name__ == "__main__":
    main()