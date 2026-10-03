set -u
cd "$(dirname "$0")/../.."
source ~/jaxenv/bin/activate

PASTA="holdout_10k"
SEED0=69000000
NSEEDS=12000
BASE="dataset_2comp_1leito_s${SEED0}-$((SEED0+NSEEDS-1))"
H5="$PASTA/${BASE}.h5"

mkdir -p "$PASTA"

marca() { echo ""; echo "[$(date)] $1"; }

marca "1/5  Generation (${NSEEDS} seeds, y0 0.20-0.75 continuous)"
if [ -f "$H5" ] && [ -s "$H5" ]; then
    echo "  h5 already exists, skipping generation"
else
    GRAVADOR_SOLVER=pcr GRAVADOR_OUTDIR="$PASTA" \
    GRAVADOR_Y0LO=0.20 GRAVADOR_Y0HI=0.75 \
    GRAVADOR_SEED0=$SEED0 GRAVADOR_NSEEDS=$NSEEDS \
    python3 -u codigo/contrato/recorder.py || { echo "GENERATION FAILED"; exit 1; }
fi
ls -la "$H5"

marca "2/5  Mask"
PASTA_H5="$PASTA" PASTA_DERIV="$PASTA" python3 -u codigo/contrato/valid_mask.py || { echo "MASK FAILED"; exit 1; }

marca "3/5  Preprocess"
if [ -f "$PASTA/pre_${BASE}.npz" ]; then
    echo "  pre already exists, skipping"
else
    MASK="$PASTA/mascara_valid_${BASE}.npz" SAIDA="$PASTA/pre_${BASE}.npz" \
    python3 -u codigo/contrato/preprocess_filtered.py "$H5" || { echo "PREPROCESS FAILED"; exit 1; }
fi

marca "4/5  Level"
PASTA_H5="$PASTA" PASTA_DERIV="$PASTA" python3 -u codigo/contrato/level.py || { echo "LEVEL FAILED"; exit 1; }

marca "5/5  Screen"
PASTA_H5="$PASTA" PASTA_DERIV="$PASTA" python3 -u codigo/contrato/screen.py || { echo "SCREEN FAILED"; exit 1; }

marca "6/6  Subsample to exactly 10,000 seeds"
python3 - << 'PYEOF'
import numpy as np, glob, os
cand = sorted(glob.glob("holdout_10k/filtrado_*.npz"))
if not cand:
    raise SystemExit("holdout filtered file not found")
d = {k: np.load(cand[0])[k] for k in np.load(cand[0]).files}
N = len(d["X"])
ALVO = 10000
print("filtered: %d seeds" % N)
if N < ALVO:
    raise SystemExit("only %d valid seeds (<%d). Increase NSEEDS and run again." % (N, ALVO))
rng = np.random.default_rng(0)  # fixed seed: reproducible subsample
idx = np.sort(rng.choice(N, ALVO, replace=False))
out = {k: (np.asarray(v)[idx] if (np.asarray(v).ndim >= 1 and np.asarray(v).shape[0] == N) else v)
       for k, v in d.items()}
saida = "saidas/holdout_10k.npz"
os.makedirs("saidas", exist_ok=True)
np.savez_compressed(saida, **out)
print("saved: %s (%d seeds)" % (saida, ALVO))

import sys; sys.path.insert(0, "codigo/contrato")
import contract_io as CIO
import features as F22
sev = out["X"][:, CIO.X_DIM + F22.NMS.index("severidade")]
y0 = out["X"][:, CIO.I_Y0]
print("\nfinal holdout check:")
print("  X: %d columns" % out["X"].shape[1])
print("  y0: %.3f to %.3f" % (y0.min(), y0.max()))
print("  Dt: %.4f to %.4f m" % (out["X"][:, CIO.I_DT].min(), out["X"][:, CIO.I_DT].max()))
print("  in domain (sev<0.70): %d of %d (%.1f%%)" % ((sev < 0.70).sum(), ALVO, 100*(sev < 0.70).mean()))
PYEOF

marca "Holdout 10k complete -> saidas/holdout_10k.npz"