import os, time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
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
EPOCHS = int(os.environ.get("EPOCHS", 300))
BATCH = int(os.environ.get("BATCH", 512))
VAL = float(os.environ.get("VAL", 0.1))
N_LAYERS = int(os.environ.get("N_LAYERS", 6))
N_UNITS = int(os.environ.get("N_UNITS", 256))
LR = float(os.environ.get("LR", 1.045e-3))
DROPOUT = float(os.environ.get("DROPOUT", 0.0))
L2 = float(os.environ.get("L2", 1.58e-7))
TF_FACTOR = float(os.environ.get("TF_FACTOR", 2.0))
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", 5))
PATIENCE = int(os.environ.get("PATIENCE", 120))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", 50))
RESUME_FROM = os.environ.get("RESUME_FROM", "")
RESUME_EPOCH = int(os.environ.get("RESUME_EPOCH", 0))
MODELO_OUT = os.environ.get("MODELO_OUT", "saidas/modelo_tempos.keras")
PRED_OUT = os.environ.get("PRED_OUT", "saidas/tempos_preditos.npz")
LOG_X_IDX = list(CIO.LOG_X_IDX)


def fit_scaler(A):
    mu = A.mean(0)
    sd = A.std(0)
    sd[sd < 1e-8] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def main():
    from tensorflow import keras
    from tensorflow.keras import layers

    d = {k: np.load(CACHE)[k] for k in np.load(CACHE).files}
    X = d["X"].astype(np.float32).copy()
    N = X.shape[0]
    for j in LOG_X_IDX:
        X[:, j] = np.log(np.maximum(X[:, j], 1e-30))
    xmu, xsd = fit_scaler(X)
    Xn = (X - xmu) / xsd

    TF = d["TF"].astype(np.float64)
    tst = TF / TF_FACTOR
    tbreak = d["tbreak"].astype(np.float64)
    tsat = d["tsat"].astype(np.float64)
    tsat = np.where(np.isfinite(tsat), tsat, TF)

    f_tb = np.log(np.maximum(tbreak / tst, 1e-6))
    f_ts = np.log(np.maximum(tsat / tst, 1e-6))
    yfeed = d["yfeed"].astype(np.float64)
    y1_feed = 1.0 - d["X"][:, CIO.I_Y0].astype(np.float64)
    maxprog = np.clip(yfeed / np.maximum(y1_feed, 1e-9), 0.0, 1.5).astype(np.float64)
    Y = np.stack([f_tb, f_ts, maxprog], axis=1).astype(np.float32)
    ymu, ysd = fit_scaler(Y)
    Yn = (Y - ymu) / ysd

    rng = np.random.default_rng(0)
    idx = rng.permutation(N)
    nval = int(VAL * N)
    vi, ti = idx[:nval], idx[nval:]

    def desnorm_tempos(Yz, tst_sub):
        Yf = Yz * ysd + ymu
        tb = np.exp(Yf[:, 0]) * tst_sub
        ts = np.exp(Yf[:, 1]) * tst_sub
        mp = Yf[:, 2]
        return tb, ts, mp

    def metricas(Yp, vsub):
        tb_p, ts_p, mp_p = desnorm_tempos(Yp, tst[vsub])

        def rel(p, real):
            return float(np.mean(np.abs(p - real) / np.maximum(real, 1e-6)) * 100)

        return (
            rel(tb_p, tbreak[vsub]),
            rel(ts_p, tsat[vsub]),
            float(np.mean(np.abs(mp_p - maxprog[vsub]))),
        )

    epocas_restantes = EPOCHS - RESUME_EPOCH
    steps_por_epoca = max(1, len(ti) // BATCH)
    if RESUME_FROM:
        print(
            "RESUMING from %s (epoch %d, %d remaining)" % (RESUME_FROM, RESUME_EPOCH, epocas_restantes),
            flush=True,
        )
        model = keras.models.load_model(RESUME_FROM, compile=False)

        sched = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=LR, decay_steps=epocas_restantes * steps_por_epoca, alpha=0.01
        )
        model.compile(optimizer=keras.optimizers.Adam(sched), loss="mse")
    else:
        inp = keras.Input(shape=(Xn.shape[1],))
        h = inp
        for _ in range(N_LAYERS):
            h = layers.Dense(
                N_UNITS, activation="gelu", kernel_regularizer=keras.regularizers.l2(L2)
            )(h)
            if DROPOUT > 0:
                h = layers.Dropout(DROPOUT)(h)
        out = layers.Dense(3)(h)
        model = keras.Model(inp, out)
        sched = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=LR, decay_steps=EPOCHS * steps_por_epoca, alpha=0.01
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
            gap = 100 * (va - tr) / max(tr, 1e-9)
            tbr, tsr, mpm = metricas(self.model.predict(Xn[vi], verbose=0), vi)
            try:
                lrv = self.model.optimizer.learning_rate
                lr = float(lrv.numpy()) if hasattr(lrv, "numpy") else float(lrv)
            except Exception:
                lr = float("nan")
            print(
                "ep %3d/%d | lr=%.1e | rmse_z tr=%.4f val=%.4f (gap %+.0f%%) | "
                "tbreak=%.2f%% tsat=%.2f%% maxprog_MAE=%.4f | %.0fs"
                % (ep_global, EPOCHS, lr, tr, va, gap, tbr, tsr, mpm, time.time() - self.t0),
                flush=True,
            )

    cbs = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=0
        ),
        Prog(),
    ]
    if CKPT_EVERY > 0:
        os.makedirs(os.path.dirname(MODELO_OUT) or ".", exist_ok=True)
        ckpt_path = MODELO_OUT.replace(".keras", "_ckpt.keras")
        epoch_txt = ckpt_path.replace(".keras", "_epoca.txt")

        class _Salva(keras.callbacks.Callback):
            def on_epoch_end(self, ep, logs=None):
                ep_global = ep + 1 + RESUME_EPOCH
                if ep_global % CKPT_EVERY == 0:
                    self.model.save(ckpt_path)
                    with open(epoch_txt, "w") as f:
                        f.write(str(ep_global))

        cbs.append(_Salva())
        print("Intermediate checkpoint every %d epochs: %s" % (CKPT_EVERY, ckpt_path), flush=True)
    print(
        "\nCASCADE network 1/2 (times) OPTUNA: %dx%d cosine dp=%.3f L2=%.0e, %d epochs\n"
        % (N_LAYERS, N_UNITS, DROPOUT, L2, EPOCHS),
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

    tbr, tsr, mpm = metricas(model.predict(Xn[vi], verbose=0), vi)
    print("\n" + "=" * 56)
    print("times network (tbreak,tsat,maxprog) final (val):")
    print("  tbreak = %.2f%%   tsat = %.2f%%   maxprog_MAE = %.4f" % (tbr, tsr, mpm))
    print("=" * 56)

    os.makedirs(os.path.dirname(MODELO_OUT) or ".", exist_ok=True)
    model.save(MODELO_OUT)
    Yp_all = model.predict(Xn, verbose=0)
    tb_all, ts_all, mp_all = desnorm_tempos(Yp_all, tst)
    np.savez(
        PRED_OUT,
        tbreak_pred=tb_all.astype(np.float32),
        tsat_pred=ts_all.astype(np.float32),
        TF_pred=(TF_FACTOR * tst).astype(np.float32),
        maxprog_pred=mp_all.astype(np.float32),
        tst=tst.astype(np.float32),
    )
    meta = MODELO_OUT.replace(".keras", "_meta.npz")
    np.savez(
        meta,
        xmu=xmu,
        xsd=xsd,
        ymu=ymu,
        ysd=ysd,
        log_x_idx=np.array(LOG_X_IDX),
        TF_FACTOR=np.float32(TF_FACTOR),
    )
    print("\nModel: %s\nPredicted times (all seeds): %s" % (MODELO_OUT, PRED_OUT))
    print("Next: python3 -u codigo/treino/train_forma.py")


if __name__ == "__main__":
    main()