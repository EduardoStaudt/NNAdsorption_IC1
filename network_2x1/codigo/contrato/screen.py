import os, glob, numpy as np, h5py
import contract_io as CIO

PASTA_H5 = os.environ.get("PASTA_H5", "dados_brutos")
PASTA_DERIV = os.environ.get("PASTA_DERIV", "derivados")
REFAZER = os.environ.get("REFAZER", "0") == "1"
os.makedirs(PASTA_DERIV, exist_ok=True)
TOL = 1.05
TREF = 298.0
DT_BAIXO = 15.0
DT_CIMA = 100.0
AMP_LIM = 0.30
DT_TOUT_BAIXO = 30.0
DT_TOUT_CIMA = 150.0

npar = len(CIO.PARAM_COMP_NAMES)
idx_qm = [npar * c + 0 for c in range(CIO.NMAX)]
idx_k2 = [npar * c + 1 for c in range(CIO.NMAX)]
i_Tin = npar * CIO.NMAX + len(CIO.PROP_LAYER_NAMES) + CIO.GLOB_NAMES.index("Tin")


def amp(y):
    y = np.asarray(y, np.float64)
    rng = max(y.max() - y.min(), 1e-9)
    k = max(5, len(y) // 8)
    t = np.convolve(y, np.ones(k) / k, mode="same")
    return np.percentile(np.abs(y - t), 95) / rng


h5s = sorted(glob.glob(os.path.join(PASTA_H5, "dataset_2comp_1leito_s6*.h5")))
tot_in = tot_out = 0
for h5 in h5s:
    base = os.path.basename(h5).replace(".h5", "")
    pre_niv = os.path.join(PASTA_DERIV, "pre_nivel_%s.npz" % base)
    mask_v = os.path.join(PASTA_DERIV, "mascara_valid_%s.npz" % base)
    saida = os.path.join(PASTA_DERIV, "filtrado_%s.npz" % base)

    if os.path.exists(saida) and not REFAZER:
        try:
            n_ok = len(np.load(saida)["X"])
            print("%s | already exists (%d seeds) -> SKIPPING" % (base, n_ok), flush=True)
            tot_out += n_ok
            continue
        except Exception as e:
            print(
                "%s | existing filtered unreadable (%s) -> redoing" % (base, type(e).__name__),
                flush=True,
            )

    d = np.load(pre_niv)
    data = {k: d[k] for k in d.files}
    N = len(data["X"])
    X = data["X"].astype(np.float64)
    qz = data["qz"].astype(np.float64)
    Tz = data["Tz"].astype(np.float64)
    N_ads = data["N_ads"].astype(np.float64)
    forma_y = data["forma_y"].astype(np.float64)
    forma_T = data["forma_T"].astype(np.float64)
    Tin = X[:, i_Tin]

    idxv = np.load(mask_v)["idx_bons"]
    with h5py.File(h5, "r") as f:
        cauda_h5 = f["cauda_inc"][:] if "cauda_inc" in f else np.zeros(f["y_out"].shape[0], bool)
    cauda = cauda_h5[idxv].astype(bool)

    qmax_c = np.empty_like(qz)
    for c in range(CIO.NMAX):
        qm_ref = X[:, idx_qm[c]][:, None]
        k2 = X[:, idx_k2[c]][:, None]
        piso = np.maximum(0.1 * qm_ref, 1e-3)
        qmax_c[:, c, :] = np.maximum(qm_ref + k2 * (Tz - TREF), piso)

    c_cauda = ~cauda
    c_overq = ~((qz > TOL * qmax_c).any(axis=(1, 2)))
    c_overqr = ~((qz > TOL * X[:, idx_qm][:, :, None]).any(axis=(1, 2)))
    c_qneg = ~((qz < -1e-6).any(axis=(1, 2)))
    c_frio = ~(((Tin[:, None] - Tz).max(axis=1)) > DT_BAIXO)
    c_quente = ~((Tz.max(axis=1) - Tin) > DT_CIMA)
    c_degen = N_ads[:, 1] > 1e-6
    amp_y = np.array([amp(forma_y[i]) for i in range(N)])
    c_amp = amp_y <= AMP_LIM
    c_tout = ~(
        ((Tin[:, None] - forma_T).max(axis=1) > DT_TOUT_BAIXO)
        | ((forma_T.max(axis=1) - Tin) > DT_TOUT_CIMA)
    )

    mask = c_cauda & c_overq & c_overqr & c_qneg & c_frio & c_quente & c_degen & c_amp & c_tout
    out = {}
    for k, v in data.items():
        v = np.asarray(v)
        out[k] = v[mask] if (v.ndim >= 1 and v.shape[0] == N) else v
    np.savez(saida, **out)
    tot_in += N
    tot_out += mask.sum()
    print("%s | %d -> %d (%.1f%%)" % (base, N, mask.sum(), 100 * mask.mean()), flush=True)

print(
    "\nTOTAL new: %d -> %d kept (%.1f%%)" % (tot_in, tot_out, 100 * tot_out / max(tot_in, 1))
)