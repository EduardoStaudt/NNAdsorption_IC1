import os, sys, time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import h5py

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "treino"))
import importlib.util


def _load(modpath, modname):
    spec = importlib.util.spec_from_file_location(modname, modpath)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

P = _load(os.environ.get("BASE_PREPROC", "preprocess.py"), "preprocess")
import contract_io as CIO
import features as FEAT

USAR_FEATURES = os.environ.get("USAR_FEATURES", "1") == "1"
M = int(os.environ.get("M", 100))
FRAC_DENSA = float(os.environ.get("FRAC_DENSA", 0.7))
MARGEM = float(os.environ.get("MARGEM", 0.05))
SAIDA = os.environ.get("SAIDA", "derivados/pre.npz")
CHUNK = int(os.environ.get("CHUNK", 4000))
K_POD = int(os.environ.get("K_POD", 8))
MASK = os.environ.get("MASK", "")
COMP = 1


def acha_arquivo():
    p = [a for a in sys.argv[1:] if not a.startswith("-")]
    if p:
        return p[0]
    sys.exit("Provide the file (COMPLETE).")


def base_pod(F, K):
    Fm = F.mean(0).astype(np.float32)
    Fc = F - Fm
    U, S, Vt = np.linalg.svd(Fc, full_matrices=False)
    modos = Vt[:K].astype(np.float32)
    ev = S**2
    ev = ev / ev.sum()
    var_k = float(np.cumsum(ev)[K - 1])
    coef = Fc @ modos.T
    return Fm, modos, coef.astype(np.float32), var_k


def main():
    arq = acha_arquivo()
    t0 = time.time()

    with h5py.File(arq, "r") as f:
        N = f["X_v2"].shape[0]
        NX = f["Tz"].shape[1]

        if MASK and os.path.exists(MASK):
            d = np.load(MASK)
            idx = np.asarray(d["idx_bons"], np.int64)
            idx.sort()
            print(
                "MASK:",
                MASK,
                "| approved seeds =",
                len(idx),
                "of",
                N,
                "(%.1f%%)" % (100 * len(idx) / N),
                flush=True,
            )
        else:
            idx = np.arange(N, dtype=np.int64)
            print("NO MASK -> processes all", N, "seeds", flush=True)

        Nf = len(idx)
        print("File:", arq, "| M =", M, "| dense frac =", FRAC_DENSA, "| Nf =", Nf, flush=True)

        X = f["X_v2"][:][idx].astype(np.float32)
        if USAR_FEATURES:
            X = FEAT.X_enriquecido(X, logar=True)
            print(
                "X enriched: %d -> %d columns (+%d features)"
                % (CIO.X_DIM, X.shape[1], FEAT.N_FEAT),
                flush=True,
            )
        qz = f["qz"][:][idx].astype(np.float32)
        Tz = f["Tz"][:][idx].astype(np.float32)
        N_ads = f["N_ads"][:][idx].astype(np.float32)
        TF = f["TF"][:][idx].astype(np.float64)
        tbreak = f["tbreak"][:, COMP][idx].astype(np.float64)
        tsat = f["tsat"][:, COMP][idx].astype(np.float64)

        forma_y = np.zeros((Nf, M), np.float32)
        forma_y0 = np.zeros((Nf, M), np.float32)
        forma_T = np.zeros((Nf, M), np.float32)
        tau_all = np.zeros((Nf, M), np.float32)
        yfeed = np.zeros(Nf, np.float32)
        yfeed0 = np.zeros(Nf, np.float32)
        Tin_all = np.zeros(Nf, np.float32)

        pc = len(CIO.PARAM_COMP_NAMES)
        i_Tin = pc * CIO.NMAX + len(CIO.PROP_LAYER_NAMES) + CIO.GLOB_NAMES.index("Tin")

        print("Resampling %d seeds (shape yi comp1 + comp0 + T_out)..." % Nf, flush=True)
        passo = max(1, Nf // 20)
        # j iterates 0..Nf-1 (position in output array); i = idx[j] (real row in h5)
        for j in range(Nf):
            i = int(idx[j])
            tb_f = tbreak[j] / TF[j] if TF[j] > 0 else 0.0
            ts_f = tsat[j] / TF[j] if (TF[j] > 0 and np.isfinite(tsat[j])) else np.nan
            tau = P.grid_tau(tb_f, ts_f)
            tau_all[j] = tau
            y = np.asarray(f["y_out"][i, COMP], np.float64)
            y0 = np.asarray(f["y_out"][i, 0], np.float64)
            T = np.asarray(f["T_out"][i], np.float64)
            yf = max(y.max(), 1e-6)
            yfeed[j] = yf
            yf0 = max(y0.max(), 1e-6)
            yfeed0[j] = yf0
            Tin_all[j] = X[j, i_Tin]
            forma_y[j] = P.reamostra(y / yf, tau)
            forma_y0[j] = P.reamostra(y0 / yf0, tau)
            forma_T[j] = P.reamostra(T, tau)
            if (j + 1) % passo == 0 or (j + 1) == Nf:
                el = time.time() - t0
                eta = (Nf - (j + 1)) / max((j + 1) / max(el, 1e-9), 1e-9)
                print(
                    "   %d/%d  (%.0f seeds/s, ETA %dm %02ds)"
                    % (j + 1, Nf, (j + 1) / max(el, 1e-9), int(eta // 60), int(eta % 60)),
                    flush=True,
                )

    os.makedirs(os.path.dirname(SAIDA) or ".", exist_ok=True)

    ymean, ymodos, ycoef, yvar = base_pod(forma_y.astype(np.float64), K_POD)
    y0mean, y0modos, y0coef, y0var = base_pod(forma_y0.astype(np.float64), K_POD)
    Tmean, Tmodos, Tcoef, Tvar = base_pod(forma_T.astype(np.float64), K_POD)
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
    print("\nCache saved: %s  (%d seeds, M=%d)" % (SAIDA, forma_y.shape[0], M))
    com_tsat = np.isfinite(tsat).mean() * 100
    print("  seeds with tsat: %.1f%%  (rest: dense grid until end, tsat=TF in training)" % com_tsat)


if __name__ == "__main__":
    main()