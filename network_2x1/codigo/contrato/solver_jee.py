import numpy as np

R = 8.314
ATM = 101325.0
TREF = 298.0
NMAX = 2

RIN = 0.01855  # m, bed radius (legacy)
DT_BED = 2.0 * RIN  # m, used only if cfg does not carry "Dt"
CPMOL = 30.0  # J/mol/K, used only if the layer does not carry "Cpg"
TF_FACTOR = 2.0


def thomas(a, b, c, d):
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def qstar_slope(c, T, K1, K2, K3, K4, K5, K6):
    # LRC: qm=K1+K2*T ; B=K3*exp(K4/T) [1/atm] ; n=K5+K6/T ; p[atm]=c*R*T/ATM
    qm = np.maximum(K1 + K2 * T[None, :], 1e-6)
    B = K3 * np.exp(K4 / T[None, :])
    nn = np.clip(K5 + K6 / T[None, :], 0.3, 3.0)
    p = np.maximum(c * R * T[None, :] / ATM, 0.0)
    pf = np.maximum(p, 1e-6)
    t = B * p**nn
    S = t.sum(axis=0)
    den = 1.0 + S
    qs = qm * t / den
    dtdp = nn * B * pf ** (nn - 1.0)
    s = qm * (den - t) / den**2 * dtdp * (R * T[None, :] / ATM)
    return qs, np.clip(s, 0.0, 1e6)


def build_layer_params_21(NX, layer):
    N = NMAX

    def pn(a):
        arr = np.zeros((N, NX))
        arr[:] = np.asarray(a, float)[:, None]
        return arr

    return dict(
        K1=pn(layer["k1"]),
        K2=pn(layer["k2"]),
        K3=pn(layer["k3"]),
        K4=pn(layer["k4"]),
        K5=pn(layer["k5"]),
        K6=pn(layer["k6"]),
        dH=pn(layer["dH"]),
        om=pn(layer["om"]),
        eb=np.full(NX, layer["eb"]),
        rho_b=np.full(NX, layer["rho_b"]),
        Cps=np.full(NX, layer["Cps"]),
    )


def setup(cfg, NX, TF=None, tf_cap=None):
    N = NMAX
    layer = cfg["layers"][0]
    y = np.asarray(cfg["y"], float)
    Vz = cfg["vs"]
    L = cfg["L"]
    Tin = cfg["Tin"]
    P_pa = cfg["P"]
    hw = cfg["hw"]
    lam = cfg["lam"]
    dp = cfg["dp"]
    Dm = cfg["Dm"]
    h = L / (NX - 1)
    par = build_layer_params_21(NX, layer)
    eb = par["eb"]
    rho_b = par["rho_b"]
    Cps = par["Cps"]
    Uz = Vz / eb
    Wsink = rho_b / eb
    c_tot = P_pa / (R * Tin)
    cfeed = y * c_tot
    Cpmol = np.asarray(layer["Cpg"], float) if "Cpg" in layer else np.full(N, CPMOL)
    assert Cpmol.shape == (N,), "Cpg must have %d values (one per component), got %s" % (
        N,
        Cpmol.shape,
    )
    
    DL = 0.7 * Dm + 0.5 * Uz * dp
    De = DL / h**2
    Ve = Uz / h
    Dt_bed = float(cfg["Dt"]) if "Dt" in cfg else DT_BED
    kap = 4.0 * hw / Dt_bed
    u_int = float(Uz[0])

    qf, _ = qstar_slope(
        cfeed[:, None],
        np.array([Tin]),
        par["K1"][:, :1],
        par["K2"][:, :1],
        par["K3"][:, :1],
        par["K4"][:, :1],
        par["K5"][:, :1],
        par["K6"][:, :1],
    )
    cauda_inc = False
    if TF is None:
        tst_i = (L / Vz) * (eb[0] + Wsink[0] * eb[0] * qf[:, 0] / np.maximum(cfeed, 1e-9))
        TF_fisico = TF_FACTOR * float(tst_i[cfeed > 0].max())
        TF = TF_fisico
        if tf_cap is not None and TF_fisico > tf_cap:
            TF = float(tf_cap)
            cauda_inc = True

    c0 = np.zeros((N, NX))
    c0[0, :] = c_tot
    T0 = np.full(NX, Tin)
    q0, _ = qstar_slope(c0, T0, par["K1"], par["K2"], par["K3"], par["K4"], par["K5"], par["K6"])
    q0 = q0.copy()
    return dict(
        par=par,
        eb=eb,
        rho_b=rho_b,
        Cps=Cps,
        Wsink=Wsink,
        Cpmol=Cpmol,
        De=De,
        Ve=Ve,
        kap=kap,
        lam=lam,
        Vz=Vz,
        Tin=Tin,
        h=h,
        u_int=u_int,
        cfeed=cfeed,
        c_tot=c_tot,
        TF=TF,
        cauda_inc=cauda_inc,
        c0=c0,
        q0=q0,
        T0=T0,
    )


def solve(cfg, NX=141, NT=1600, TF=None, tf_cap=None):
    N = NMAX
    S = setup(cfg, NX, TF=TF, tf_cap=tf_cap)
    par = S["par"]
    eb = S["eb"]
    rho_b = S["rho_b"]
    Cps = S["Cps"]
    Wsink = S["Wsink"]
    Cpmol = S["Cpmol"]
    De = S["De"]
    Ve = S["Ve"]
    kap = S["kap"]
    lam = S["lam"]
    Vz = S["Vz"]
    Tin = S["Tin"]
    h = S["h"]
    cfeed = S["cfeed"]
    TF = S["TF"]
    dt = TF / NT
    c = S["c0"].copy()
    q = S["q0"].copy()
    T = S["T0"].copy()
    Tw = Tin

    yt = np.zeros((NT, N))
    Tt = np.zeros(NT)
    Tmax = Tin
    for step in range(NT):
        cold = c.copy()
        qold = q.copy()
        Told = T.copy()
        for _ in range(4):
            qs, s = qstar_slope(
                c, T, par["K1"], par["K2"], par["K3"], par["K4"], par["K5"], par["K6"]
            )
            G = Wsink[None, :] * par["om"] / (1.0 + par["om"] * dt)
            diag = (1 / dt + 2 * De + Ve) + G * s
            dd = cold / dt - G * (qs - qold) + G * s * c
            dd[:, 0] += (De[0] + Ve[0]) * cfeed
            diag[:, -1] = 1 / dt + De[-1] + Ve[-1] + G[:, -1] * s[:, -1]
            sub = np.tile(-(De + Ve), (N, 1))
            sup = np.full((N, NX), -De)
            for i in range(N):
                c[i] = thomas(sub[i], diag[i], sup[i], dd[i])
            c = np.clip(c, 0.0, None)
            qs2, _ = qstar_slope(
                c, T, par["K1"], par["K2"], par["K3"], par["K4"], par["K5"], par["K6"]
            )
            qnew = (qold + dt * par["om"] * qs2) / (1.0 + par["om"] * dt)
            dqdt = (qnew - qold) / dt
            ct = np.maximum(c.sum(0), 1e-9)
            rhoCp = eb * ct * (Cpmol @ (c / ct)) + rho_b * Cps
            Gc = Vz * ct * (Cpmol @ (c / ct))
            alE = lam / rhoCp / h**2
            VeE = Gc / rhoCp / h
            kapE = kap / rhoCp
            Qa = (rho_b * (par["dH"] * dqdt).sum(0)) / rhoCp
            diagT = 1 / dt + 2 * alE + VeE + kapE
            dTv = Told / dt + kapE * Tw + Qa
            dTv[0] += (alE[0] + VeE[0]) * Tin
            diagT[-1] = 1 / dt + alE[-1] + VeE[-1] + kapE[-1]
            T = thomas(-(alE + VeE), diagT, -alE, dTv)
        qs2, _ = qstar_slope(c, T, par["K1"], par["K2"], par["K3"], par["K4"], par["K5"], par["K6"])
        q = (qold + dt * par["om"] * qs2) / (1.0 + par["om"] * dt)
        Tmax = max(Tmax, T.max())
        yt[step] = c[:, -1]
        Tt[step] = T[-1]
    return yt, Tt, Tmax, TF, dt


if __name__ == "__main__":
    k3 = np.array([0.625e-4, 100.0e-4])
    k4 = np.array([1229.0, 1030.0])
    Bref = (k3 * np.exp(k4 / 298.0)).tolist()
    cfg = dict(
        layers=[
            dict(
                k1=[16.943, 28.797],
                k2=[-0.021, -0.070],
                k3=k3.tolist(),
                Bref=Bref,
                k4=k4.tolist(),
                k5=[0.980, 0.999],
                k6=[43.03, -37.04],
                om=[0.700, 0.036],
                dH=[2880 * 4.184, 5240 * 4.184],
                eb=0.433,
                rho_b=482.0,
                Cps=1046.0,
            )
        ],
        y=[0.70, 0.30],
        vs=0.02,
        Tin=299.0,
        P=5e5,
        L=1.0,
        hw=50.0,
        lam=0.6,
        dp=2e-3,
        Dm=1e-5,
    )

    yt, Tt, Tmax, TF, dt = solve(cfg, NX=141, NT=1600)
    NT = len(yt)
    tot = np.maximum(yt.sum(1, keepdims=True), 1e-12)
    yfrac = yt / tot
    print("=== binary H2/CO2 70:30, 5 atm, vs=2cm/s (activated carbon) ===")
    print("  TF=%.0fs  dt=%.3fs  dTmax=%.1fK (peak-Tin)" % (TF, dt, Tmax - 299.0))
    print("  t/TF    y_H2    y_CO2   T_out[K]")
    for k in range(0, NT, NT // 16):
        print("  %.3f   %.4f  %.4f   %.2f" % (k / NT, yfrac[k, 0], yfrac[k, 1], Tt[k]))
    yco2 = yfrac[:, 1]
    tgrid = np.arange(NT) / NT
    rng = yco2[-1] - yco2[0]
    if abs(rng) > 1e-3:
        prog = (yco2 - yco2[0]) / rng
        tb = tgrid[np.argmax(prog >= 0.05)]
        ts = tgrid[np.argmax(prog >= 0.95)]
        print("  frente CO2: tbreak(5%%)=%.3f  tsat(95%%)=%.3f  (t/TF)" % (tb, ts))