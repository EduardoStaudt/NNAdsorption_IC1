import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax, vmap
from functools import partial

R = 8.314
ATM = 101325.0


def qstar_slope(c, T, K1, K2, K3, K4, K5, K6):
    # c:[N,NX] T:[NX] K*:[N,NX]  (LRC, p in atm)
    qm = jnp.maximum(K1 + K2 * T[None, :], 1e-6)
    B = K3 * jnp.exp(K4 / T[None, :])
    nn = jnp.clip(K5 + K6 / T[None, :], 0.3, 3.0)
    p = jnp.maximum(c * R * T[None, :] / ATM, 0.0)
    pf = jnp.maximum(p, 1e-6)
    t = B * p**nn
    den = 1.0 + t.sum(axis=0)
    qs = qm * t / den
    dtdp = nn * B * pf ** (nn - 1.0)
    s = qm * (den - t) / den**2 * dtdp * (R * T[None, :] / ATM)
    return qs, jnp.clip(s, 0.0, 1e6)


def thomas(a, b, c, d):
    # solves tridiagonal (a=sub, b=diag, c=sup, d=rhs); 1D of size NX
    def fwd(carry, inp):
        cp_p, dp_p = carry
        ai, bi, ci, di = inp
        m = bi - ai * cp_p
        cp_i = ci / m
        dp_i = (di - ai * dp_p) / m
        return (cp_i, dp_i), (cp_i, dp_i)

    _, (cp, dp) = lax.scan(fwd, (0.0, 0.0), (a, b, c, d))

    def bwd(x_next, inp):
        cp_i, dp_i = inp
        x_i = dp_i - cp_i * x_next
        return x_i, x_i

    _, x = lax.scan(bwd, 0.0, (cp, dp), reverse=True)
    return x


thomas_N = vmap(thomas, in_axes=(0, 0, 0, 0))  # over components


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
        c_new = jnp.clip(thomas_N(sub, diag, sup, dd), 0.0, None)
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
        T_new = thomas(-(alE + VeE), diagT, -alE, dTv)
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