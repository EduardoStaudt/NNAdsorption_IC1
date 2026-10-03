import json
import math
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np

_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_RAIZ, "codigo", "contrato"))
os.chdir(_RAIZ)

import contract_io as CIO
import solver_jee as SJ
from features import NMS, X_enriquecido

TREF = 298.0
SEV_MAX = 0.90
M_TAU, FRAC_DENSA, MARGEM = 100, 0.70, 0.05


def cfg_from_json(e):
    v = lambda k: np.asarray(e[k], np.float64)
    layer = dict(
        k1=v("qm_ref") - v("k2") * TREF,
        k2=v("k2"),
        k3=v("B_ref") * np.exp(-v("k4") / TREF),
        k4=v("k4"),
        k5=v("n_ref") - v("k6") / TREF,
        k6=v("k6"),
        Bref=v("B_ref"),
        om=v("kL"),
        dH=v("dH"),
        Cpg=v("Cpg"),
        eb=float(e["eb"]),
        rho_b=float(e["rho_b"]),
        Cps=float(e["Cps"]),
    )
    return dict(
        layers=[layer],
        y=np.array([float(e["y0"]), 1.0 - float(e["y0"])]),
        carrier=0,
        vs=float(e["vs"]),
        Tin=float(e["Tin"]),
        P=float(e["P"]),
        L=float(e["L"]),
        hw=float(e["hw"]),
        lam=float(e["lam"]),
        dp=float(e["dp"]),
        Dm=float(e["Dm"]),
        Dt=float(e["Dt"]),
    )


def grid_tau(tb_f, ts_f):
    t0 = max(0.0, tb_f - MARGEM)
    t1 = min(max(ts_f if np.isfinite(ts_f) else 1.0, t0 + 1e-3), 1.0)
    n_densa = int(round(FRAC_DENSA * M_TAU))
    n_pre = max(2, int(0.10 * M_TAU))
    n_pos = M_TAU - n_densa - n_pre
    pre = np.linspace(0.0, t0, n_pre, endpoint=False)
    densa = np.linspace(t0, t1, n_densa, endpoint=(n_pos == 0))
    pos = np.linspace(t1, 1.0, n_pos + 1)[1:] if n_pos > 0 else np.array([])
    tau = np.clip(np.sort(np.concatenate([pre, densa, pos])), 0.0, 1.0)
    return tau if len(tau) == M_TAU else np.linspace(0, 1, M_TAU)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    with_solver = "--solver" in sys.argv
    arq = args[0] if args else os.path.join("codigo", "example_input.json")
    e = json.load(open(arq, encoding="utf-8"))
    cfg = cfg_from_json(e)

    X31 = np.asarray(CIO.vetor_X_v3(cfg), np.float64).reshape(1, -1)
    Xe = X_enriquecido(X31, logar=True).astype(np.float64)
    sev = float(Xe[0, CIO.X_DIM + NMS.index("severidade")])
    if sev >= SEV_MAX:
        raise SystemExit(
            "case OUTSIDE the applicability domain (severity %.3f >= %.2f)" % (sev, SEV_MAX)
        )
    Xl = Xe.copy()
    for j in list(CIO.LOG_X_IDX):
        Xl[:, j] = np.log(np.maximum(Xl[:, j], 1e-30))

    S = SJ.setup(cfg, NX=71, tf_cap=3600.0)
    tst = float(S["TF"]) / 2.0
    TF = 2.0 * tst

    from tensorflow import keras

    def load(b):
        m = keras.models.load_model(os.path.join("modelos", b + ".keras"), compile=False)
        return m, dict(np.load(os.path.join("modelos", b + "_meta.npz")))

    mt, met = load("modelo_tempos_1M")
    Yt = mt.predict((Xl - met["xmu"]) / met["xsd"], verbose=0) * met["ysd"] + met["ymu"]
    tb = float(np.exp(Yt[0, 0]) * tst)
    ts = float(np.exp(Yt[0, 1]) * tst)
    mp = float(Yt[0, 2])

    mf, mef = load("modelo_forma_sev090")
    X51 = np.concatenate([Xl, [[Yt[0, 0], Yt[0, 1], np.log(2.0), mp]]], axis=1)
    Yf = mf.predict((X51 - mef["xmu"]) / mef["xsd"], verbose=0) * mef["ysd"] + mef["ymu"]
    y0 = np.clip(Yf[0, :100], 0, 1)
    y1 = 1.0 - y0
    T = Yf[0, 100:200]
    t = grid_tau(tb / TF, ts / TF) * TF

    print("severity = %.3f (domain: < %.2f)" % (sev, SEV_MAX))
    print("tst = %.1f s | TF = %.1f s" % (tst, TF))
    print("t_break = %.1f s | t_sat = %.1f s | max progression = %.3f" % (tb, ts, mp))

    sol = None
    if with_solver:
        nt = S["TF"] * S["u_int"] / (0.4 * S["h"])
        NT = int(min(12000, math.ceil(nt)))
        print("running the reference solver (NX=71, NT=%d)..." % NT)
        yt_s, Tt_s, _, TF_s, dt_s = SJ.solve(cfg, NX=71, NT=NT, tf_cap=3600.0)
        tg = np.arange(len(Tt_s)) * dt_s
        yfrac = yt_s / np.maximum(yt_s.sum(1, keepdims=True), 1e-12)
        sol = dict(t=tg, y1=yfrac[:, 1], y0=yfrac[:, 0], T=Tt_s)
        y1_i = np.interp(t, tg, sol["y1"])
        T_i = np.interp(t, tg, sol["T"])
        print(
            "surrogate vs solver: RMSE y_strong = %.4f | RMSE T = %.3f K"
            % (
                float(np.sqrt(np.mean((y1 - y1_i) ** 2))),
                float(np.sqrt(np.mean((T - T_i) ** 2))),
            )
        )

    os.makedirs("saidas", exist_ok=True)
    with open(os.path.join("saidas", "predicted_curve.csv"), "w", encoding="utf-8") as f:
        f.write("t_s,y_strong,y_carrier,T_K\n")
        for i in range(M_TAU):
            f.write("%.4f,%.6f,%.6f,%.3f\n" % (t[i], y1[i], y0[i], T[i]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    if sol is not None:
        a1.plot(sol["t"], sol["y1"], "-", color="tab:blue", lw=1.6, label="solver (strong)")
        a1.plot(sol["t"], sol["y0"], "--", color="tab:blue", lw=1.2, label="solver (carrier)")
        a2.plot(sol["t"], sol["T"], "-", color="tab:blue", lw=1.6, label="solver")
    a1.plot(t, y1, "o", ms=3.5, color="tab:orange", label="surrogate (strong)")
    a1.plot(t, y0, "s", ms=3, color="tab:orange", mfc="none", label="surrogate (carrier)")
    a2.plot(t, T, "o", ms=3.5, color="tab:orange", label="surrogate")
    a1.set_xlabel("t (s)")
    a1.set_ylabel("outlet mole fraction")
    a2.set_xlabel("t (s)")
    a2.set_ylabel("outlet T (K)")
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join("saidas", "predicted_curve.png"), dpi=130)
    print("outputs: saidas/predicted_curve.csv and saidas/predicted_curve.png")


if __name__ == "__main__":
    main()
