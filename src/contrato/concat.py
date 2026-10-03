import os, glob
import numpy as np
import os as _os, sys as _sys

_RAIZ = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
for _d in ("contrato", "figuras", "avaliacao"):
    _sys.path.insert(0, _os.path.join(_RAIZ, "codigo", _d))
_os.chdir(_RAIZ)
import src.contrato.contract_io as CIO

PASTA_DERIV = os.environ.get("PASTA_DERIV", "derivados")
SAIDA = os.environ.get("SAIDA", "dados/cache.npz")
K_POD = int(os.environ.get("K_POD", 8))

arqs = sorted(glob.glob(os.path.join(PASTA_DERIV, "filtrado_dataset_2comp_1leito_s6*.npz")))
print("filtered files found: %d" % len(arqs), flush=True)
for a in arqs:
    print("  " + os.path.basename(a))
if not arqs:
    raise SystemExit("no filtrado_*.npz in %s" % PASTA_DERIV)

partes = [np.load(a) for a in arqs]
d0 = partes[0]
N0 = len(d0["X"])
por_seed = [k for k in d0.files if np.asarray(d0[k]).ndim >= 1 and np.asarray(d0[k]).shape[0] == N0]
escalares = [k for k in d0.files if k not in por_seed]

out = {}
for k in por_seed:
    arrs = [np.asarray(p[k]) for p in partes if k in p.files]
    if len(arrs) != len(partes):
        print("  field missing in some file, skipping:", k, flush=True)
        continue
    out[k] = np.concatenate(arrs, axis=0)
    print("  %-14s %s" % (k, out[k].shape), flush=True)
for k in escalares:
    out[k] = d0[k]

N = len(out["X"])
y0 = out["X"][:, CIO.I_Y0]
print("\ntotal seeds: %d" % N, flush=True)
print(
    "X: %d columns (expected %d = %d contract + 16 features)"
    % (out["X"].shape[1], CIO.X_DIM + 16, CIO.X_DIM),
    flush=True,
)
print(
    "y0 (carrier in feed): %.3f to %.3f  -- continuous range, no cutoff at 0.50"
    % (y0.min(), y0.max()),
    flush=True,
)
print(
    "  (reference: y0<0.50 would be %d seeds; y0>=0.50, %d -- no longer separated)"
    % ((y0 < 0.50).sum(), (y0 >= 0.50).sum()),
    flush=True,
)
print(
    "Dt : %.4f to %.4f m   | Cpg c0: %.1f to %.1f | Cpg c1: %.1f to %.1f J/mol/K"
    % (
        out["X"][:, CIO.I_DT].min(),
        out["X"][:, CIO.I_DT].max(),
        out["X"][:, 8].min(),
        out["X"][:, 8].max(),
        out["X"][:, 17].min(),
        out["X"][:, 17].max(),
    ),
    flush=True,
)


def base_pod(F, K):
    mean = F.mean(0)
    Fc = F - mean
    U, S, Vt = np.linalg.svd(Fc, full_matrices=False)
    modos = Vt[:K].astype(np.float32)
    coef = (Fc @ modos.T).astype(np.float32)
    ev = (S**2) / (S**2).sum()
    return mean.astype(np.float32), modos, coef, float(np.cumsum(ev)[K - 1])


print("\nrecomputing POD over the UNIFIED set (may take a few minutes)...", flush=True)
for nome, campo in [("y", "forma_y"), ("y0", "forma_y0"), ("T", "forma_T")]:
    mean, modos, coef, var = base_pod(out[campo].astype(np.float64), K_POD)
    out["%s_pod_mean" % nome] = mean
    out["%s_pod_modos" % nome] = modos
    out["%s_pod_coef" % nome] = coef
    print("  POD %-2s: %.2f%% var" % (nome, 100 * var), flush=True)

os.makedirs(os.path.dirname(SAIDA) or ".", exist_ok=True)
np.savez_compressed(SAIDA, **out)
print("\nsaved: %s (%d seeds)" % (SAIDA, N), flush=True)