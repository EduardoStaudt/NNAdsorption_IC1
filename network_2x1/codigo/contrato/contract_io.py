import numpy as np

NMAX = 2
NLAY = 1
TREF = 298.0

PARAM_COMP_NAMES = ["qm_ref", "k2", "B_ref", "k4", "n_ref", "k6", "kL", "dH", "Cpg"]
N_PARAM_COMP = len(PARAM_COMP_NAMES)
PROP_LAYER_NAMES = ["eb", "rho_b", "Cps"]
GLOB_NAMES = ["vs", "Tin", "P", "L", "hw", "lam", "dp", "Dm", "Dt"]
FEED_NAMES = ["y0"]

N_ISO = NLAY * NMAX * N_PARAM_COMP
N_PROP = NLAY * len(PROP_LAYER_NAMES)
N_GLOB = len(GLOB_NAMES)
N_FEED = len(FEED_NAMES)
X_DIM = N_ISO + N_PROP + N_GLOB + N_FEED


def vetor_X_v3(cfg):
    out = []
    for L in cfg["layers"]:
        for i in range(NMAX):
            out += [
                L["k1"][i] + L["k2"][i] * TREF,  # qm_ref
                L["k2"][i],
                L["Bref"][i],
                L["k4"][i],
                L["k5"][i] + L["k6"][i] / TREF,  # n_ref
                L["k6"][i],
                L["om"][i],
                L["dH"][i],
                L["Cpg"][i],
            ]
    for L in cfg["layers"]:
        out += [L["eb"], L["rho_b"], L["Cps"]]
    out += [
        cfg["vs"],
        cfg["Tin"],
        cfg["P"],
        cfg["L"],
        cfg["hw"],
        cfg["lam"],
        cfg["dp"],
        cfg["Dm"],
        cfg["Dt"],
    ]
    out += list(cfg["y"][: NMAX - 1])
    return np.array(out, dtype=np.float32)


def log_x_idx():
    bref, kl = [], []
    for c in range(NLAY):
        for i in range(NMAX):
            base = (c * NMAX + i) * N_PARAM_COMP
            bref.append(base + PARAM_COMP_NAMES.index("B_ref"))
            kl.append(base + PARAM_COMP_NAMES.index("kL"))
    return bref + kl


LOG_X_IDX = log_x_idx()


def x_column_names():
    names = []
    for c in range(NLAY):
        for i in range(NMAX):
            for p in PARAM_COMP_NAMES:
                names.append("L%d_c%d_%s" % (c, i, p))
    for c in range(NLAY):
        for p in PROP_LAYER_NAMES:
            names.append("L%d_%s" % (c, p))
    names += GLOB_NAMES
    names += FEED_NAMES
    return names


I_DP = N_ISO + N_PROP + GLOB_NAMES.index("dp")
I_DM = N_ISO + N_PROP + GLOB_NAMES.index("Dm")
I_DT = N_ISO + N_PROP + GLOB_NAMES.index("Dt")
I_Y0 = N_ISO + N_PROP + N_GLOB


X_RANGES = {
    "qm_ref": (1.0, 15.0),
    "k2": (-0.1, 0.0),
    "B_ref": (1e-4, 1.0),
    "k4": (0.0, 3200.0),
    "n_ref": (0.5, 1.5),
    "k6": (-2200.0, 2200.0),
    "kL": (0.01, 1.0),
    "dH": (5e3, 50e3),
    "Cpg": (25.0, 45.0),
    "eb": (0.30, 0.50),
    "rho_b": (400.0, 900.0),
    "Cps": (800.0, 1200.0),
    "vs": (0.001, 0.05),
    "Tin": (288.0, 323.0),
    "P": (1e5, 3e6),
    "L": (0.3, 1.5),
    "hw": (20.0, 100.0),
    "lam": (0.1, 0.8),
    "dp": (0.5e-3, 5e-3),
    "Dm": (5e-6, 3e-5),
    "Dt": (0.01, 0.06),
}


if __name__ == "__main__":
    n = x_column_names()
    assert len(n) == X_DIM, (len(n), X_DIM)
    print("X_DIM =", X_DIM, "(v21 had 28)")
    print("LOG_X_IDX =", LOG_X_IDX)
    print("I_DP =", I_DP, " I_DM =", I_DM, " I_DT =", I_DT, " I_Y0 =", I_Y0)
    print("columns:")
    for k, nm in enumerate(n):
        marca = "  <<< NEW" if ("Cpg" in nm or nm == "Dt") else ""
        print("  %2d  %s%s" % (k, nm, marca))