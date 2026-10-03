set -u
cd "$(dirname "$0")/../.."
source ~/jaxenv/bin/activate

OUTDIR="dados_brutos"
NSEEDS=200000
ESPACO_MIN_GB=25
SEEDS=(63000000 64000000 65000000 66000000 67000000 68000000)

mkdir -p "$OUTDIR"

verifica_h5() {
    python3 - "$1" << 'PYEOF'
import sys, os
try:
    import h5py, numpy as np
except ImportError:
    sys.exit(2)
p = sys.argv[1]
if not os.path.exists(p) or os.path.getsize(p) == 0:
    print("    -> file missing or 0 bytes"); sys.exit(1)
try:
    with h5py.File(p, "r") as f:
        n = f["X_v2"].shape[0]; cols = f["X_v2"].shape[1]
        v = int(f["valid"][:].sum())
        if cols != 31:
            print("    -> X_v2 with %d columns (expected 31)" % cols); sys.exit(1)
        print("    -> OK: %d seeds, %d columns, %d valid (%.1f%%), %.1f GB"
              % (n, cols, v, 100*v/n, os.path.getsize(p)/1e9))
except Exception as e:
    print("    -> ERROR opening: %s: %s" % (type(e).__name__, e)); sys.exit(1)
PYEOF
}

echo "[$(date)] Generating 1.2M seeds in ${#SEEDS[@]} blocks of $NSEEDS"

for s0 in "${SEEDS[@]}"; do
    s1=$((s0 + NSEEDS - 1))
    arq="$OUTDIR/dataset_2comp_1leito_s${s0}-${s1}.h5"
    echo ""
    echo "[$(date)] Block seeds ${s0}..${s1}"

    if [ -f "$arq" ]; then
        echo "  file already exists, checking..."
        if verifica_h5 "$arq"; then
            echo "  -> already complete and intact. Skipping."
            continue
        else
            echo "  -> corrupted/incomplete. Deleting and redoing."
            rm -f "$arq"
        fi
    fi

    livre=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
    echo "  free space on C: ${livre} GB"
    if [ "$livre" -lt "$ESPACO_MIN_GB" ]; then
        echo "  !! aborting: less than ${ESPACO_MIN_GB} GB free."
        exit 1
    fi

    GRAVADOR_SOLVER=pcr GRAVADOR_OUTDIR="$OUTDIR" \
    GRAVADOR_Y0LO=0.20 GRAVADOR_Y0HI=0.75 \
    GRAVADOR_SEED0="$s0" GRAVADOR_NSEEDS="$NSEEDS" \
    python3 -u codigo/contrato/recorder.py
    codigo=$?

    echo "  recorder finished with code $codigo. Checking the file..."
    if verifica_h5 "$arq"; then
        echo "  [$(date)] block ${s0} complete and verified."
    else
        echo "  !! block ${s0} failed the check. Stopping here to avoid"
        echo "     wasting hours on the following blocks. Run the script again"
        echo "     (already-good blocks will be skipped) after investigating."
        exit 1
    fi
done

echo ""
echo "[$(date)] All blocks complete"
ls -la "$OUTDIR"/*.h5
echo ""
df -h . | tail -1
echo ""
echo "Next step: the funnel (mask -> preprocess -> level -> screen)."