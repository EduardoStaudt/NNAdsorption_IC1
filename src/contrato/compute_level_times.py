import os, sys, numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "treino"))
H5 = os.environ.get("H5", "")
SAIDA = os.environ.get("SAIDA", "nivel.npz")
NIVEIS = [float(x) for x in os.environ.get("NIVEIS", "0.25,0.50,0.75").split(",")]
COMP = int(os.environ.get("COMP", 1))
CHUNK = int(os.environ.get("CHUNK", 4000))
if not H5 or not os.path.exists(H5):
    sys.exit("Define H5=<path to dataset_2comp_1leito_COMPLETO.h5>")


def tempos_nivel_curva(y, y1_feed, TF, niveis):
    """Returns (tempos[len(niveis)], flags[len(niveis)]) for one curve.
    tempo = instant when prog>=nivel (nan if not reached); flag = reached?"""
    L = len(y)
    if L < 2 or y1_feed <= 1e-9:
        return np.full(len(niveis), np.nan), np.zeros(len(niveis), np.int8)
    prog = y / y1_feed
    t_axis = np.linspace(0.0, TF, L)
    tempos = np.full(len(niveis), np.nan)
    flags = np.zeros(len(niveis), np.int8)
    for j, niv in enumerate(niveis):
        idx = np.where(prog >= niv)[0]
        if len(idx) > 0:
            k = idx[0]
            if k > 0:
                p0, p1 = prog[k - 1], prog[k]
                t0, t1 = t_axis[k - 1], t_axis[k]
                frac = (niv - p0) / max(p1 - p0, 1e-12)
                tempos[j] = t0 + frac * (t1 - t0)
            else:
                tempos[j] = t_axis[0]
            flags[j] = 1
    return tempos, flags


print("Opening:", H5, flush=True)
with h5py.File(H5, "r") as f:
    N = f["y_out"].shape[0]
    TF = f["TF"][:].astype(np.float64) if "TF" in f else None
    import src.contrato.contract_io as CIO22
    # y1_feed = 1 - y0. The y0 index comes from the contract (col 30); column 27 is dp,
    # using it would give y1_feed = 1 - dp, silently wrong.
    X = f["X_v2"][:, :].astype(np.float64) if "X_v2" in f else None
    if X is not None and X.shape[1] != CIO22.X_DIM:
        sys.exit(
            "X_v2 has %d columns, contract v22 expects %d. Is it a v21 file?"
            % (X.shape[1], CIO22.X_DIM)
        )
    y0_feed = X[:, CIO22.I_Y0] if X is not None else None
    if y0_feed is None:
        sys.exit("X_v2 not found for y1_feed")
    y1_feed = 1.0 - y0_feed
    if TF is None:
        sys.exit("TF not found in h5")
    print("N=%d seeds | niveis=%s" % (N, NIVEIS), flush=True)

    tempos = np.full((N, len(NIVEIS)), np.nan, np.float32)
    flags = np.zeros((N, len(NIVEIS)), np.int8)
    passo = max(1, N // 40)
    for i in range(N):
        y = np.asarray(f["y_out"][i, COMP], np.float64)
        tempos[i], flags[i] = tempos_nivel_curva(y, y1_feed[i], TF[i], NIVEIS)
        if (i + 1) % passo == 0 or (i + 1) == N:
            print("   %d/%d" % (i + 1, N), flush=True)

os.makedirs(os.path.dirname(SAIDA) or ".", exist_ok=True)
np.savez(SAIDA, tempos_nivel=tempos, flags_nivel=flags, niveis=np.array(NIVEIS))
print("\nSaved:", SAIDA, flush=True)
print("Coverage per level (fraction reached):")
for j, niv in enumerate(NIVEIS):
    print("  t%.0f: %.1f%% reached (flag=1)" % (niv * 100, 100 * flags[:, j].mean()))