import os, sys, time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
import numpy as np

import os as _os, sys as _sys

_RAIZ = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
for _d in ("contrato", "figuras", "avaliacao"):
    _sys.path.insert(0, _os.path.join(_RAIZ, "codigo", _d))
_os.chdir(_RAIZ)
import contract_io as CIO

CACHE = os.environ.get("CACHE", "dados/cache.npz")
PRED = os.environ.get("PRED", "saidas/tempos_preditos.npz")
EPOCHS = int(os.environ.get("EPOCHS", 500))
BATCH = int(os.environ.get("BATCH", 512))
VAL = float(os.environ.get("VAL", 0.1))
N_LAYERS = int(os.environ.get("N_LAYERS", 6))
N_UNITS = int(os.environ.get("N_UNITS", 384))
LR = float(os.environ.get("LR", 2e-3))
ALPHA = float(os.environ.get("ALPHA", 0.01))
EARLY = os.environ.get("EARLY", "1") == "1"
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", 5))
SEV_MIN = float(os.environ.get("SEV_MIN", 0.0))
SEV_MAX = float(os.environ.get("SEV_MAX", 0.70))

import features as _F22

_SEV_DEFAULT = CIO.X_DIM + _F22.NMS.index("severidade")
SEV_IDX = int(os.environ.get("SEV_IDX", _SEV_DEFAULT))
L2 = float(os.environ.get("L2", 1e-6))
MODELO_OUT = os.environ.get("MODELO_OUT", "saidas/modelo_forma.keras")
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", 25))
PATIENCE = int(os.environ.get("PATIENCE", 120))
RESUME_FROM = os.environ.get("RESUME_FROM", "")
RESUME_EPOCH = int(os.environ.get("RESUME_EPOCH", 0))
LOG_X_IDX = list(CIO.LOG_X_IDX)


def fit_scaler(A):
    mu = A.mean(0)
    sd = A.std(0)
    sd[sd < 1e-8] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def main():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import absl.logging

    absl.logging.set_verbosity(absl.logging.ERROR)
    tf.get_logger().setLevel("ERROR")

    d = {k: np.load(CACHE)[k] for k in np.load(CACHE).files}
    if not os.path.exists(PRED):
        sys.exit("Run the times network first (missing %s)." % PRED)
    pr = np.load(PRED)
    N = d["X"].shape[0]
    if len(pr["tbreak_pred"]) != N:
        sys.exit(
            "PRED (%d) does not match the cache (%d). Generate PRED from the SAME cache."
            % (len(pr["tbreak_pred"]), N)
        )

    sev = d["X"][:, SEV_IDX].astype(np.float64)
    forte = np.where((sev >= SEV_MIN) & (sev < SEV_MAX))[0]
    print(
        "Severity window [%.2f,%.2f): %d of %d seeds (%.1f%%)"
        % (SEV_MIN, SEV_MAX, len(forte), N, 100 * len(forte) / N),
        flush=True,
    )

    X = d["X"].astype(np.float32).copy()
    for j in LOG_X_IDX:
        X[:, j] = np.log(np.maximum(X[:, j], 1e-30))
    tst = pr["tst"].astype(np.float64)
    ft = np.stack(
        [
            np.log(np.maximum(pr["tbreak_pred"] / tst, 1e-6)),
            np.log(np.maximum(pr["tsat_pred"] / tst, 1e-6)),
            np.log(np.maximum(pr["TF_pred"] / tst, 1e-6)),
        ],
        axis=1,
    ).astype(np.float32)
    if "maxprog_pred" in pr.files:
        maxprog = pr["maxprog_pred"].astype(np.float32).reshape(-1, 1)
        print(
            "Using PREDICTED maxprog (autonomous pipeline):", maxprog.shape[1], "feature", flush=True
        )
        Xe = np.concatenate([X, ft, maxprog], axis=1)[forte]
    else:
        print("NO maxprog_pred in PRED -> training without it (baseline).", flush=True)
        Xe = np.concatenate([X, ft], axis=1)[forte]
    xmu, xsd = fit_scaler(Xe)
    Xn = (Xe - xmu) / xsd
    Yy = d["forma_y"][forte].astype(np.float32)
    Yy0 = d["forma_y0"][forte].astype(np.float32)
    Yt = d["forma_T"][forte].astype(np.float32)
    M = Yy0.shape[1]
    sl = {"y0": slice(0, M), "T": slice(M, 2 * M)}
    Y = np.concatenate([Yy0, Yt], axis=1)
    ymu, ysd = fit_scaler(Y)
    Yn = (Y - ymu) / ysd
    print(
        "CONSERVES: Input %d cols | Output %d (carrier+Tout; strong=1-carrier) | window"
        % (Xn.shape[1], Y.shape[1]),
        flush=True,
    )

    rng = np.random.default_rng(0)
    nf = len(forte)
    idx = rng.permutation(nf)
    nval = int(VAL * nf)
    vi, ti = idx[:nval], idx[nval:]
    yfeed_v = d["yfeed"][forte][vi].astype(np.float64)
    yb = {"y": Yy[vi], "y0": Yy0[vi], "T": Yt[vi], "yf": yfeed_v}

    def metr(Yp):
        Yf = Yp * ysd + ymu
        r = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
        y0_pred = Yf[:, sl["y0"]]
        ys_pred_real = 1.0 - np.clip(y0_pred, 0, 1)
        ys_real = yb["y"] * yb["yf"][:, None]
        r_forte = r(ys_pred_real, ys_real)
        return (r_forte, r(y0_pred, yb["y0"]), r(Yf[:, sl["T"]], yb["T"]))

    def piores(Yp):
        Yf = Yp * ysd + ymu
        y0_pred = Yf[:, sl["y0"]]
        ys_pred_real = 1.0 - np.clip(y0_pred, 0, 1)
        ys_real = yb["y"] * yb["yf"][:, None]
        e = np.sqrt(np.mean((ys_pred_real - ys_real) ** 2, axis=1))
        return np.percentile(e, 99), 100 * np.mean(e > 0.08)

    epocas_restantes = EPOCHS - RESUME_EPOCH
    steps_por_epoca = max(1, len(ti) // BATCH)
    if RESUME_FROM:
        print(
            "RESUMING from %s (epoch %d, %d remaining)" % (RESUME_FROM, RESUME_EPOCH, epocas_restantes),
            flush=True,
        )
        model = keras.models.load_model(RESUME_FROM, compile=False)
        sched = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=LR, decay_steps=epocas_restantes * steps_por_epoca, alpha=ALPHA
        )
        model.compile(optimizer=keras.optimizers.Adam(sched), loss="mse")
    else:
        inp = keras.Input(shape=(Xn.shape[1],))
        h = inp
        for _ in range(N_LAYERS):
            h = layers.Dense(
                N_UNITS, activation="gelu", kernel_regularizer=keras.regularizers.l2(L2)
            )(h)
        out = layers.Dense(Y.shape[1])(h)
        model = keras.Model(inp, out)
        sched = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=LR, decay_steps=EPOCHS * steps_por_epoca, alpha=ALPHA
        )
        model.compile(optimizer=keras.optimizers.Adam(sched), loss="mse")

    class Prog(keras.callbacks.Callback):
        def __init__(self):
            self.t0 = time.time()

        def on_epoch_end(self, ep, logs=None):
            ep_global = ep + 1 + RESUME_EPOCH
            if ep_global % PRINT_EVERY and ep_global != EPOCHS:
                return
            tr = float(logs["loss"]) ** 0.5
            va = float(logs["val_loss"]) ** 0.5
            fy, fy0, fT = metr(self.model.predict(Xn[vi], verbose=0))
            p99, ppi = piores(self.model.predict(Xn[vi], verbose=0))
            try:
                opt = self.model.optimizer
                step = opt.iterations
                lrv = opt.learning_rate
                lr = float(lrv(step).numpy()) if callable(lrv) else float(lrv.numpy())
            except Exception:
                lr = float("nan")
            print(
                "ep %3d/%d | lr=%.1e | forma=%.4f roll=%.4f Tout=%.3fK | p99=%.4f >0.08=%.2f%% | %.0fs"
                % (ep_global, EPOCHS, lr, fy, fy0, fT, p99, ppi, time.time() - self.t0),
                flush=True,
            )

    cbs = [Prog()]
    if EARLY:
        cbs.insert(
            0,
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=0
            ),
        )
    if CKPT_EVERY > 0:
        os.makedirs(os.path.dirname(MODELO_OUT) or ".", exist_ok=True)
        ckpt_path = MODELO_OUT.replace(".keras", "_ckpt.keras")
        epoch_txt = ckpt_path.replace(".keras", "_epoca.txt")

        class _Salva(keras.callbacks.Callback):
            def on_epoch_end(self, ep, logs=None):
                ep_g = ep + 1 + RESUME_EPOCH
                if ep_g % CKPT_EVERY == 0:
                    self.model.save(ckpt_path)
                    with open(epoch_txt, "w") as f:
                        f.write(str(ep_g))

        cbs.append(_Salva())
        print("Intermediate checkpoint every %d epochs: %s" % (CKPT_EVERY, ckpt_path), flush=True)
    print(
        "\nWINDOW NETWORK [%.2f,%.2f) COSINE: %dx%d, %d epochs (alpha=%.3f, early=%d)\n"
        % (SEV_MIN, SEV_MAX, N_LAYERS, N_UNITS, EPOCHS, ALPHA, EARLY),
        flush=True,
    )
    model.fit(
        Xn[ti],
        Yn[ti],
        validation_data=(Xn[vi], Yn[vi]),
        epochs=epocas_restantes,
        batch_size=BATCH,
        callbacks=cbs,
        verbose=0,
    )

    fy, fy0, fT = metr(model.predict(Xn[vi], verbose=0))
    p99, ppi = piores(model.predict(Xn[vi], verbose=0))
    print("\n" + "=" * 64)
    print("WINDOW NETWORK [%.2f,%.2f) COSINE final RMSE (val):" % (SEV_MIN, SEV_MAX))
    print("  shape yi = %.4f | roll = %.4f | T_out = %.3f K" % (fy, fy0, fT))
    print("  p99 = %.4f | >0.08 = %.2f%%" % (p99, ppi))
    print("=" * 64)
    os.makedirs(os.path.dirname(MODELO_OUT) or ".", exist_ok=True)
    model.save(MODELO_OUT)
    # meta with scalers and flags of the conservative format (output=[carrier,Tout], strong=1-carrier)
    meta_path = MODELO_OUT.replace(".keras", "_meta.npz")
    usa_maxprog = int("maxprog_pred" in pr.files)
    np.savez(
        meta_path,
        xmu=xmu,
        xsd=xsd,
        ymu=ymu,
        ysd=ysd,
        M=np.int32(M),
        conserva=np.int32(1),
        usa_maxprog=np.int32(usa_maxprog),
        ordem=np.array(["carrier", "Tout"]),
    )
    print("Model:", MODELO_OUT)
    print("Meta (CONSERVES%s):" % (" +MAXPROG" if usa_maxprog else ""), meta_path)
    print("  format: output=[carrier(M), Tout(M)]; strong = 1 - carrier (real fraction)")
    if usa_maxprog:
        print("  input includes 1 PREDICTED maxprog feature at the END of the feature vector")


if __name__ == "__main__":
    main()