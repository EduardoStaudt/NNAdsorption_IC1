import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
from functools import partial
from src.contrato.solver_jax import qstar_slope


def _shift_r(v, s, fill):
    n = v.shape[-1]
    pad = jnp.full(v.shape[:-1] + (s,), fill, v.dtype)
    return jnp.concatenate([pad, v[..., : n - s]], axis=-1)


def _shift_l(v, s, fill):
    n = v.shape[-1]
    pad = jnp.full(v.shape[:-1] + (s,), fill, v.dtype)
    return jnp.concatenate([v[..., s:], pad], axis=-1)


def pcr(a, b, c, d):
    n = b.shape[-1]
    a = a.at[..., 0].set(0.0)
    c = c.at[..., -1].set(0.0)
    s = 1
    while s < n:
        aL = _shift_r(a, s, 0.0)
        bL = _shift_r(b, s, 1.0)
        cL = _shift_r(c, s, 0.0)
        dL = _shift_r(d, s, 0.0)
        aR = _shift_l(a, s, 0.0)
        bR = _shift_l(b, s, 1.0)
        cR = _shift_l(c, s, 0.0)
        dR = _shift_l(d, s, 0.0)
        k1 = a / bL
        k2 = c / bR
        a_new = -k1 * aL
        b_new = b - k1 * cL - k2 * aR
        c_new = -k2 * cR
        d_new = d - k1 * dL - k2 * dR
        a, b, c, d = a_new, b_new, c_new, d_new
        s *= 2
    return d / b


def make_solver(N, NX, NT):
    def picard(c, T, cold, qold, Told, P):
        (
            K1,
            K2,
            K3,
            K4,
            K5,
            K6,
            dH,
            om,
            eb,
            rho_b,
            Cps,
            Wsink,
            De,
            Ve,
            dt,
            cfeed,
            Vz,
            kap,
            lam,
            Tw,
            Tin,
            h,
            Cpmol,
        ) = P
        qs, s = qstar_slope(c, T, K1, K2, K3, K4, K5, K6)
        G = Wsink[None, :] * om / (1.0 + om * dt)
        diag = (1.0 / dt + 2 * De + Ve[None, :]) + G * s
        dd = cold / dt - G * (qs - qold) + G * s * c
        dd = dd.at[:, 0].add((De + Ve[0]) * cfeed)
        diag = diag.at[:, -1].set(1.0 / dt + De + Ve[-1] + G[:, -1] * s[:, -1])
        sub = jnp.broadcast_to(-(De + Ve[None, :]), (N, NX))
        sup = jnp.full((N, NX), -De)
        c_new = jnp.clip(pcr(sub, diag, sup, dd), 0.0, None)  # PCR instead of Thomas
        qs2, _ = qstar_slope(c_new, T, K1, K2, K3, K4, K5, K6)
        qnew = (qold + dt * om * qs2) / (1.0 + om * dt)
        dqdt = (qnew - qold) / dt
        ct = jnp.maximum(c_new.sum(0), 1e-9)
        cp_mix = Cpmol @ (c_new / ct)
        rhoCp = eb * ct * cp_mix + rho_b * Cps
        Gc = Vz * ct * cp_mix
        alE = lam / rhoCp / h**2
        VeE = Gc / rhoCp / h
        kapE = kap / rhoCp
        Qa = (rho_b * (dH * dqdt).sum(0)) / rhoCp
        diagT = 1.0 / dt + 2 * alE + VeE + kapE
        dTv = Told / dt + kapE * Tw + Qa
        dTv = dTv.at[0].add((alE[0] + VeE[0]) * Tin)
        diagT = diagT.at[-1].set(1.0 / dt + alE[-1] + VeE[-1] + kapE[-1])
        T_new = pcr(-(alE + VeE), diagT, -alE, dTv)  # PCR instead of Thomas
        return c_new, T_new

    def step(carry, _, P):
        c, q, T = carry
        cold, qold, Told = c, q, T

        def body(i, ct):
            c_, T_ = ct
            return picard(c_, T_, cold, qold, Told, P)

        c, T = lax.fori_loop(0, 4, body, (c, T))
        om = P[7]
        dt = P[14]
        qs2, _ = qstar_slope(c, T, P[0], P[1], P[2], P[3], P[4], P[5])
        q = (qold + dt * om * qs2) / (1.0 + om * dt)
        return (c, q, T), (c[:, -1], T[-1])

    def solve(c0, q0, T0, P):
        (c, q, T), (yt, Tt) = lax.scan(partial(step, P=P), (c0, q0, T0), None, length=NT)
        return yt, Tt, q, T, c

    return jax.jit(solve)