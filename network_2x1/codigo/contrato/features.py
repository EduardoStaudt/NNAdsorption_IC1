import numpy as np

R = 8.314
NMS = [
    "tau_resid",
    "NTU_0",
    "NTU_1",
    "Pe",
    "cf_0",
    "cf_1",
    "K_cap_0",
    "K_cap_1",
    "selet_B",
    "selet_q",
    "adT_0",
    "adT_1",
    "ratio_kL",
    "Bi_dp",
    "severidade",
    "NTU_wall",
]
N_FEAT = len(NMS)

I_EB, I_RHOB, I_CPS = 18, 19, 20
I_VS, I_TIN, I_P, I_L, I_HW, I_LAM, I_DP, I_DM, I_DT = 21, 22, 23, 24, 25, 26, 27, 28, 29
I_Y0 = 30


def _col(X, c, p):
    """Column of parameter p from component c."""
    base = c * 9
    idx = {
        "qm_ref": 0,
        "k2": 1,
        "B_ref": 2,
        "k4": 3,
        "n_ref": 4,
        "k6": 5,
        "kL": 6,
        "dH": 7,
        "Cpg": 8,
    }
    return X[:, base + idx[p]]


def _norm01(v, lo, hi, log=False):
    v = np.asarray(v, np.float64)
    if log:
        v = np.log(np.maximum(v, 1e-30))
        lo = np.log(max(lo, 1e-30))
        hi = np.log(max(hi, 1e-30))
    return np.clip((v - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def severidade_regime(X):
    """Severity index that delimits the declared application domain (s < 0.9 in the
    final set). Formula and weights are fixed: the index is part of the model's input
    contract."""
    Bc1 = _col(X, 1, "B_ref")
    kLc1 = _col(X, 1, "kL")
    qmc0 = _col(X, 0, "qm_ref")
    k6c0 = _col(X, 0, "k6")
    P = X[:, I_P]
    vs = X[:, I_VS]
    L = X[:, I_L]
    NTU_c1 = kLc1 * L / np.maximum(vs, 1e-12)

    s = (
        0.30 * _norm01(Bc1, 1e-4, 1.0, log=True)
        + 0.15 * _norm01(kLc1, 0.01, 1.0, log=True)
        + 0.15 * _norm01(P, 1e5, 3e6)
        + 0.15 * _norm01(qmc0, 1.0, 15.0)
        + 0.15 * _norm01(-k6c0, -2200.0, 2200.0)
        + 0.10 * _norm01(NTU_c1, 0.0, 5.0)
    )
    return s.astype(np.float64)


def features_derivadas(X, logar=True):
    """X [N,31] -> F [N,N_FEAT]."""
    eb = X[:, I_EB]
    rho_b = X[:, I_RHOB]
    Cps = X[:, I_CPS]
    vs = X[:, I_VS]
    Tin = X[:, I_TIN]
    P = X[:, I_P]
    L = X[:, I_L]
    hw = X[:, I_HW]
    dp = X[:, I_DP]
    Dm = X[:, I_DM]
    Dt = X[:, I_DT]
    y0 = X[:, I_Y0]
    y = [y0, 1.0 - y0]

    qm = [_col(X, c, "qm_ref") for c in range(2)]
    B = [_col(X, c, "B_ref") for c in range(2)]
    kL = [_col(X, c, "kL") for c in range(2)]
    dH = [_col(X, c, "dH") for c in range(2)]
    Cpg = [_col(X, c, "Cpg") for c in range(2)]

    cf = [np.maximum(y[c] * P / (R * Tin), 1e-9) for c in range(2)]
    denom = 1.0 + B[0] * cf[0] + B[1] * cf[1]
    qstar = [qm[c] * B[c] * cf[c] / denom for c in range(2)]
    K_cap = [rho_b * qstar[c] / (eb * cf[c]) for c in range(2)]

    eps = 1e-12
    ct = np.maximum(P / (R * Tin), eps)  # mol/m3, ideal gas
    Cpg_mist = y[0] * Cpg[0] + y[1] * Cpg[1]  # J/(mol.K)
    NTU_wall = (4.0 * hw * L / np.maximum(Dt, eps)) / np.maximum(vs * ct * Cpg_mist, eps)

    F = np.stack(
        [
            L * eb / np.maximum(vs, eps),  # tau_resid
            kL[0] * L / np.maximum(vs, eps),  # NTU_0
            kL[1] * L / np.maximum(vs, eps),  # NTU_1
            vs * L / np.maximum(Dm, eps),  # Pe
            cf[0],
            cf[1],  # cf_0, cf_1
            np.maximum(K_cap[0], eps),
            np.maximum(K_cap[1], eps),
            np.maximum(B[1] / np.maximum(B[0], eps), eps),  # selet_B
            np.maximum((qm[1] * B[1]) / np.maximum(qm[0] * B[0], eps), eps),  # selet_q
            dH[0] / np.maximum(rho_b * Cps, eps),  # adT_0
            dH[1] / np.maximum(rho_b * Cps, eps),  # adT_1
            np.maximum(kL[1] / np.maximum(kL[0], eps), eps),  # ratio_kL
            vs * dp / np.maximum(Dm, eps),  # Bi_dp
            severidade_regime(X),  # severity [0,1]
            np.maximum(NTU_wall, eps),  # NTU_wall
        ],
        axis=1,
    ).astype(np.float64)

    if logar:
        # log on the multiplicative ones; adT (linear) and severidade ([0,1]) stay out
        log_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1], bool)
        F[:, log_mask] = np.log(np.maximum(F[:, log_mask], 1e-30))
    return F.astype(np.float32)


def X_enriquecido(X, logar=True):
    """Concatenates X + features -> [N, 47]."""
    return np.concatenate([X, features_derivadas(X, logar=logar)], axis=1)


if __name__ == "__main__":
    import contract_io as CIO22

    rng = np.random.default_rng(0)
    N = 5
    X = np.zeros((N, CIO22.X_DIM), np.float32)
    for j, nome in enumerate(CIO22.x_column_names()):
        chave = nome.split("_", 2)[-1] if nome.startswith("L0_c") else nome.replace("L0_", "")
        if chave in CIO22.X_RANGES:
            lo, hi = CIO22.X_RANGES[chave]
            X[:, j] = rng.uniform(lo, hi, N)
        elif chave == "y0":
            X[:, j] = rng.uniform(0.20, 0.75, N)
    F = features_derivadas(X)
    Xe = X_enriquecido(X)
    print("contract X_DIM =", CIO22.X_DIM)
    print("features (%d):" % N_FEAT, NMS)
    print(
        "F shape:",
        F.shape,
        "| X_enriquecido:",
        Xe.shape,
        "(expected (%d, %d))" % (N, CIO22.X_DIM + N_FEAT),
    )
    print("no nan:", not np.isnan(F).any(), "| no inf:", not np.isinf(F).any())
    print("\nNTU_wall (log) in 5 samples:", F[:, NMS.index("NTU_wall")])