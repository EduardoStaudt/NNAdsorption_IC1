import numpy as np, h5py, glob, os

PASTA_H5 = os.environ.get("PASTA_H5", "dados_brutos")
PASTA_DERIV = os.environ.get("PASTA_DERIV", "derivados")
REFAZER = (
    os.environ.get("REFAZER", "0") == "1"
)
os.makedirs(PASTA_DERIV, exist_ok=True)
arquivos = sorted(glob.glob(os.path.join(PASTA_H5, "dataset_2comp_1leito_s6*.h5")))
print("files found:", len(arquivos))

for h5 in arquivos:
    base = os.path.basename(h5).replace(".h5", "")
    saida = os.path.join(PASTA_DERIV, "mascara_valid_%s.npz" % base)

    if os.path.exists(saida) and not REFAZER:
        try:
            n_h5 = h5py.File(h5, "r")["y_out"].shape[0]
            idx_ok = np.load(saida)["idx_bons"]
            if len(idx_ok) <= n_h5 and (len(idx_ok) == 0 or idx_ok.max() < n_h5):
                print(
                    "%s | already exists (%d approved of %d) -> SKIPPING" % (base, len(idx_ok), n_h5),
                    flush=True,
                )
                continue
            print("%s | existing mask INCONSISTENT with h5 -> redoing" % base, flush=True)
        except Exception as e:
            print(
                "%s | existing mask unreadable (%s) -> redoing" % (base, type(e).__name__),
                flush=True,
            )

    with h5py.File(h5, "r") as f:
        N = f["y_out"].shape[0]
        valid = f["valid"][:].astype(bool) if "valid" in f else np.ones(N, bool)
        tem_nan = np.zeros(N, bool)
        for i in range(N):
            y = f["y_out"][i, 1]
            y0 = f["y_out"][i, 0]
            T = f["T_out"][i]
            if np.isnan(y).any() or np.isnan(y0).any() or np.isnan(T).any() or np.isinf(y).any():
                tem_nan[i] = True
    bons = valid & (~tem_nan)
    idx_bons = np.where(bons)[0].astype(np.int64)
    np.savez(saida, idx_bons=idx_bons)
    print(
        "%s | N=%d valid=0:%d NaN:%d -> approved %d"
        % (base, N, (~valid).sum(), tem_nan.sum(), len(idx_bons)),
        flush=True,
    )
print("ok")