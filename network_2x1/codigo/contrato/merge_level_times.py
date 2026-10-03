import os, sys, numpy as np

NIVEL = os.environ.get("NIVEL", "")
MASK = os.environ.get("MASK", "")
CACHE_IN = os.environ.get("CACHE_IN", "")
CACHE_OUT = os.environ.get("CACHE_OUT", "")
if not (NIVEL and MASK and CACHE_IN and CACHE_OUT):
    raise SystemExit("define NIVEL, MASK, CACHE_IN and CACHE_OUT in the environment")

dn = np.load(NIVEL)
tempos = dn["tempos_nivel"]
flags = dn["flags_nivel"]
niveis = dn["niveis"]
print("tempos_nivel: %d seeds (all from COMPLETO) | niveis=%s" % (len(tempos), niveis))

dm = np.load(MASK)
print("mask: fields =", list(dm.files))
idx = None
for k in ["idx_bons", "idx", "indices", "seeds"]:
    if k in dm.files:
        idx = np.asarray(dm[k], np.int64)
        print("  using field '%s': %d indices" % (k, len(idx)))
        break
if idx is None:
    sys.exit("idx_bons not found in mask")

dc = np.load(CACHE_IN)
Ncache = len(dc["X"])
print("training cache: %d seeds" % Ncache)

tempos_f = tempos[idx]
flags_f = flags[idx]
print("after mask: tempos_nivel = %d seeds" % len(tempos_f))
if len(tempos_f) != Ncache:
    print(
        "!!! SIZE MISMATCH (%d vs %d). Mask may not match the cache."
        % (len(tempos_f), Ncache)
    )
    print("    verify that idx_bons corresponds exactly to cache_pre_completo_limpo.")
    sys.exit(1)
print("OK: sizes match (%d = %d)" % (len(tempos_f), Ncache))

out = {k: dc[k] for k in dc.files}
out["tempos_nivel"] = tempos_f.astype(np.float32)
out["flags_nivel"] = flags_f.astype(np.int8)
out["niveis"] = niveis.astype(np.float32)
np.savez(CACHE_OUT, **out)
print("\nSaved:", CACHE_OUT)
print("flag coverage in cache:")
for j, niv in enumerate(niveis):
    print("  t%.0f: %.1f%% reached" % (niv * 100, 100 * flags_f[:, j].mean()))