import jax.numpy as jnp
from jax import random, jit, lax, vmap

from config import NeuronConfig, surrogate_sigma
from two_comp_neuron import TwoCompNeuron
from lif_neuron import LINeuron


# ══════════════════════════════════════════════════════════════════════
#  Core functions — each processes ONE sample.
#
#  These have no decorators so we can wrap them two ways:
#    jit(fn)            → fast single-sample execution
#    jit(vmap(fn, ...)) → fast batched execution (B samples in parallel)
# ══════════════════════════════════════════════════════════════════════

def _forward_and_accum(
    x_input, w_dend, w_soma, w_readout,
    alpha_s, alpha_d, alpha_m, T_p, config, alpha_w,
    h_carry_init, r_carry_init, A_d_init,
    rng_key, dropout_rate,
):
    """Forward pass + gradient accumulator bookkeeping for one sample.

    x_input:      (T, K)  input spike train
    h_carry_init: 8-tuple of hidden-neuron state zeros
    r_carry_init: 3-tuple of readout-neuron state zeros
    A_d_init:     (J, N, K) dendritic accumulator zeros
    rng_key:      PRNG key for dropout masks
    dropout_rate: fraction of hidden spikes to drop (0.0 = no dropout)

    Returns: readout_counts (J,), A_readout (J,N), A_soma (J,N,K), A_dend (J,N,K)
    """
    dend_inputs = x_input @ w_dend.T
    soma_inputs = x_input @ w_soma.T
    T = x_input.shape[0]
    n_hidden = w_dend.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)
    dropout_keys = random.split(rng_key, T)
    dropout_scale = 1.0 / (1.0 - dropout_rate)

    def step(carry, inputs):
        h_carry, r_carry, A_d = carry
        dend_in, soma_in, x_t, t, drop_key = inputs

        h_carry, h_o, h_v_pre, h_h, h_h_prev, h_mu_at_tp = TwoCompNeuron.forward_step(
            h_carry, dend_in, soma_in, t, alpha_s, alpha_d, T_p, config, alpha_w,
        )
        hidden_o_float = h_o.astype(jnp.float64)

        # Dropout: randomly zero out hidden spikes before the readout sees them.
        # Surviving spikes are scaled by 1/(1-p) so expected value is unchanged.
        # At dropout_rate=0.0 the mask is all-ones and scale is 1.0 (no-op).
        mask = random.bernoulli(drop_key, 1.0 - dropout_rate, (n_hidden,)).astype(jnp.float64)
        hidden_o_float = hidden_o_float * mask * dropout_scale

        r_carry, r_v, r_E = LINeuron.forward_step(
            r_carry, hidden_o_float, w_readout, alpha_m,
        )

        mu_c, v_c, h_c, tp_c, matp_c, E_soma_c, dmu_c, dmu_atp_c, w_c = h_carry
        E_soma_new = TwoCompNeuron.update_somatic_eligibility(
            E_soma_c, x_t.astype(jnp.float64), alpha_s,
        )
        dmu_new, dmu_atp_new = TwoCompNeuron.update_dendritic_eligibility(
            dmu_c, dmu_atp_c, x_t.astype(jnp.float64), h_h_prev, alpha_d,
        )
        h_carry = (mu_c, v_c, h_c, tp_c, matp_c, E_soma_new, dmu_new, dmu_atp_new, w_c)

        # LI readout is linear in its inputs: ∂v_j/∂input = 1, no surrogate needed.
        sp_readout = jnp.ones(w_readout.shape[0])
        sp_hidden = surrogate_sigma(
            h_v_pre + config.gamma * h_h - config.v_th, config.beta_s,
        )
        hp_hidden = surrogate_sigma(h_mu_at_tp - config.mu_th, config.beta_d)

        eta = sp_readout[:, None] * w_readout * sp_hidden[None, :]
        eta_d = eta * (hp_hidden * config.gamma)[None, :]
        A_d = A_d + jnp.einsum("ji,ik->jik", eta_d, dmu_atp_new)

        new_carry = (h_carry, r_carry, A_d)
        per_step = (sp_readout, sp_hidden, r_E, E_soma_new)
        return new_carry, per_step

    init_carry = (h_carry_init, r_carry_init, A_d_init)
    scan_inputs = (dend_inputs, soma_inputs, x_input, time_indices, dropout_keys)
    final_carry, per_step_all = lax.scan(step, init_carry, scan_inputs)

    sp_r, sp_h, E_r, E_s = per_step_all
    _, r_carry_f, A_d_f = final_carry
    mean_voltage = r_carry_f[1] / T  # sum_v / T

    A_readout = jnp.einsum("ti,tj->ij", sp_r, E_r)
    C_soma = jnp.einsum("tj,ti,tk->jik", sp_r, sp_h, E_s)
    A_soma = w_readout[:, :, None] * C_soma

    return mean_voltage, A_readout, A_soma, A_d_f


def _predict_only(
    x_input, w_dend, w_soma, w_readout,
    alpha_s, alpha_d, alpha_m, T_p, config, alpha_w,
):
    """Forward pass only — no gradient bookkeeping. Returns readout_counts (J,)."""
    dend_inputs = x_input @ w_dend.T
    soma_inputs = x_input @ w_soma.T
    T = x_input.shape[0]
    n_hidden = w_dend.shape[0]
    n_outputs = w_readout.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)

    def step(carry, inputs):
        mu, v, h, t_prime, mu_at_tp, w, r_v, r_counts = carry
        dend_in, soma_in, t = inputs

        t_prime_new = jnp.where(t == 0, 0, jnp.where(h == 1, t_prime, t))
        mu_new = jnp.where(t > 0, alpha_d * mu + (1 - h) * dend_in, dend_in)
        mu_at_tp_new = jnp.where(h == 0, mu_new, mu_at_tp)

        plat_dur = t - t_prime_new
        h_new = jnp.where(
            (mu_at_tp_new >= config.mu_th) & (plat_dur <= T_p) & (plat_dur >= 0),
            1, 0,
        ).astype(jnp.int32)

        v_pre = jnp.where(t > 0, alpha_s * v + soma_in - w, soma_in)
        o_h = jnp.where(v_pre >= config.v_th - config.gamma * h_new, 1, 0).astype(jnp.int32)
        v_new = v_pre * (1 - o_h)
        w_new = alpha_w * w + (1 - alpha_w) * config.a_adapt * v_pre + config.b_adapt * o_h

        r_in = o_h.astype(jnp.float64) @ w_readout.T
        r_v_new = alpha_m * r_v + r_in  # no threshold, no reset
        r_counts_new = r_counts + r_v_new  # accumulate voltage sum

        return (mu_new, v_new, h_new, t_prime_new, mu_at_tp_new, w_new, r_v_new, r_counts_new), None

    init = (
        jnp.zeros(n_hidden), jnp.zeros(n_hidden),
        jnp.zeros(n_hidden, dtype=jnp.int32), jnp.zeros(n_hidden, dtype=jnp.int32),
        jnp.zeros(n_hidden), jnp.zeros(n_hidden),
        jnp.zeros(n_outputs), jnp.zeros(n_outputs),
    )
    final, _ = lax.scan(step, init, (dend_inputs, soma_inputs, time_indices))
    return final[7] / T  # mean voltage


def _loss_and_grads(
    mean_voltage, A_readout, A_soma, A_dend,
    target_smoothed, T, loss_temperature, loss_count_bias,
):
    """Compute loss and weight gradients for one sample."""
    scaled_logits = mean_voltage / loss_temperature + loss_count_bias
    probs = jnp.exp(scaled_logits - jnp.max(scaled_logits))
    probs = probs / jnp.sum(probs)

    prediction = jnp.argmax(mean_voltage)
    loss = -jnp.sum(target_smoothed * jnp.log(probs + 1e-8))
    global_error = target_smoothed - probs

    grad_readout = (global_error[:, None] * A_readout) / T
    grad_soma = jnp.einsum("j,jik->ik", global_error, A_soma) / (T * 8.0)
    grad_dend = jnp.einsum("j,jik->ik", global_error, A_dend) / T

    return loss, prediction, grad_readout, grad_soma, grad_dend


def _apply_grads(
    w_dend, w_soma, w_readout, g_dend, g_soma, g_readout,
    lr, clip_value, weight_decay,
):
    """SGD with decoupled weight decay.

    Gradients are clipped first, then weights are updated as
        w ← w + lr·g − lr·λ·w,
    which for plain SGD is mathematically equivalent to adding
    (λ/2)·||w||² to the loss (the L2 penalty), but keeps the
    weight-decay term outside the clip so it can't be capped by
    the gradient clip.
    """
    g_readout = jnp.clip(g_readout, -clip_value, clip_value)
    g_soma = jnp.clip(g_soma, -clip_value, clip_value)
    g_dend = jnp.clip(g_dend, -clip_value, clip_value)
    return (
        w_dend + lr * g_dend - lr * weight_decay * w_dend,
        w_soma + lr * g_soma - lr * weight_decay * w_soma,
        w_readout + lr * g_readout - lr * weight_decay * w_readout,
    )


def _adam_apply(
    w_d, w_s, w_r,
    g_d, g_s, g_r,
    m_d, m_s, m_r,
    v_d, v_s, v_r,
    step, lr, beta1, beta2, eps, clip_value, weight_decay,
):
    """AdamW-style decoupled weight decay.

    The data gradient flows through the moment estimates and the
    adaptive 1/√v rescaling, but the λ·w term does not — it is
    subtracted directly from w after the Adam update. This matches
    Loshchilov & Hutter (2017) and avoids the parameter-dependent
    decay strength that arises if λ·w is added to the loss before
    Adam normalisation.
    """
    def update_one(w, g, m, v):
        g = jnp.clip(g, -clip_value, clip_value)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)
        w = w + lr * m_hat / (jnp.sqrt(v_hat) + eps) - lr * weight_decay * w
        return w, m, v

    w_d, m_d, v_d = update_one(w_d, g_d, m_d, v_d)
    w_s, m_s, v_s = update_one(w_s, g_s, m_s, v_s)
    w_r, m_r, v_r = update_one(w_r, g_r, m_r, v_r)
    return (w_d, w_s, w_r, m_d, m_s, m_r, v_d, v_s, v_r)


# ══════════════════════════════════════════════════════════════════════
#  TWO HIDDEN LAYERS — faithful cross-layer dendritic credit (Option B)
#
#  Architecture: x -> L1 (2-comp) -> L2 (2-comp) -> readout (LI).
#
#  Error reaches L1 through L2 via two pathways from each L1 spike o1_i:
#    - L2 soma weight w_soma2[m,i]  -> reads o1 at the CURRENT time t
#    - L2 dend weight w_dend2[m,i]  -> reads o1 at L2's FROZEN plateau time t'
#  The dendritic pathway is handled with a nested frozen eligibility trace
#  (P_*) that filters L1's own weight-sensitivities through L2's dendritic
#  decay and latches them at L2's plateau initiation.
#
#  Cost optimization: global_error and w_readout are time-independent, so the
#  heavy L1 accumulators are kept indexed by the L2 neuron m (shape N2,N1,K),
#  NOT by the readout j; G_m = sum_j gerr_j*w_readout[j,m] is applied at the end.
# ══════════════════════════════════════════════════════════════════════

def _forward_and_accum_2l(
    x_input, w_dend1, w_soma1, w_dend2, w_soma2, w_readout,
    alpha_s, alpha_d, alpha_m, T_p1, T_p2, config, alpha_w,
    h1_carry_init, h2_carry_init, r_carry_init, nested_init, A_init,
    rng_key, dropout_rate,
):
    """Forward + gradient bookkeeping for one sample, two hidden layers.

    Returns: mean_voltage (J,), A_r (N2,), C_soma2 (N2,N1), C_dend2 (N2,N1),
             C_S2_soma (N2,N1,K), C_S2_dend (N2,N1,K),
             C_D2_soma (N2,N1,K), C_D2_dend (N2,N1,K)
    """
    dend_in1 = x_input @ w_dend1.T
    soma_in1 = x_input @ w_soma1.T
    T = x_input.shape[0]
    n_hidden2 = w_dend2.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)
    dropout_keys = random.split(rng_key, T)
    dropout_scale = 1.0 / (1.0 - dropout_rate)

    def step(carry, inputs):
        h1_carry, h2_carry, r_carry, nested, A = carry
        dend_in1_t, soma_in1_t, x_t, t, drop_key = inputs
        P_soma, P_soma_atp, P_dend, P_dend_atp = nested
        (A_r, C_soma2, C_dend2,
         C_S2_soma, C_S2_dend, C_D2_soma, C_D2_dend) = A

        # ── L1 forward (presyn = input x) ──
        h1_carry, o1, v1_pre, h1_h, h1_h_prev, mu1_at_tp = TwoCompNeuron.forward_step(
            h1_carry, dend_in1_t, soma_in1_t, t, alpha_s, alpha_d, T_p1, config, alpha_w,
        )
        o1_f = o1.astype(jnp.float64)

        # ── L1 eligibility (presyn = input x) ──
        mu1, v1, h1, tp1, matp1, E_soma1_c, dmu1_c, dmu1_atp_c, w1c = h1_carry
        E_soma1 = TwoCompNeuron.update_somatic_eligibility(
            E_soma1_c, x_t.astype(jnp.float64), alpha_s,
        )
        dmu1, dmu1_atp = TwoCompNeuron.update_dendritic_eligibility(
            dmu1_c, dmu1_atp_c, x_t.astype(jnp.float64), h1_h_prev, alpha_d,
        )
        h1_carry = (mu1, v1, h1, tp1, matp1, E_soma1, dmu1, dmu1_atp, w1c)

        # ── L1 weight-sensitivities ∂o1_i/∂w1[i,k] (surrogate) ──
        sp_soma1 = surrogate_sigma(v1_pre + config.gamma * h1_h - config.v_th, config.beta_s)
        hp_dend1 = surrogate_sigma(mu1_at_tp - config.mu_th, config.beta_d)
        g1s = sp_soma1[:, None] * E_soma1[None, :]                       # (N1,K) soma-weight
        g1d = (sp_soma1 * config.gamma * hp_dend1)[:, None] * dmu1_atp   # (N1,K) dend-weight

        # ── L2 forward (presyn = o1) ──
        dend_in2_t = o1_f @ w_dend2.T
        soma_in2_t = o1_f @ w_soma2.T
        h2_carry, o2, v2_pre, h2_h, h2_h_prev, mu2_at_tp = TwoCompNeuron.forward_step(
            h2_carry, dend_in2_t, soma_in2_t, t, alpha_s, alpha_d, T_p2, config, alpha_w,
        )
        o2_f = o2.astype(jnp.float64)

        # ── L2 eligibility (presyn = o1) ──
        mu2, v2, h2, tp2, matp2, E_soma2_c, dmu2_c, dmu2_atp_c, w2c = h2_carry
        E_soma2 = TwoCompNeuron.update_somatic_eligibility(E_soma2_c, o1_f, alpha_s)
        dmu2, dmu2_atp = TwoCompNeuron.update_dendritic_eligibility(
            dmu2_c, dmu2_atp_c, o1_f, h2_h_prev, alpha_d,
        )
        h2_carry = (mu2, v2, h2, tp2, matp2, E_soma2, dmu2, dmu2_atp, w2c)

        # ── L2 surrogates ──
        sp_soma2 = surrogate_sigma(v2_pre + config.gamma * h2_h - config.v_th, config.beta_s)
        hp_dend2 = surrogate_sigma(mu2_at_tp - config.mu_th, config.beta_d)
        DGfac = sp_soma2 * config.gamma * hp_dend2   # (N2,) L2-dendrite branch factor

        # ── Readout (presyn = o2, with dropout) ──
        mask = random.bernoulli(drop_key, 1.0 - dropout_rate, (n_hidden2,)).astype(jnp.float64)
        o2_drop = o2_f * mask * dropout_scale
        r_carry, r_v, r_E = LINeuron.forward_step(r_carry, o2_drop, w_readout, alpha_m)

        # ── Nested frozen traces: L1 sensitivity filtered through L2 dendrite ──
        sens_soma = w_dend2[:, :, None] * g1s[None, :, :]   # (N2,N1,K)
        sens_dend = w_dend2[:, :, None] * g1d[None, :, :]
        P_soma, P_soma_atp = TwoCompNeuron.update_nested_dendritic_eligibility(
            P_soma, P_soma_atp, sens_soma, h2_h_prev, alpha_d,
        )
        P_dend, P_dend_atp = TwoCompNeuron.update_nested_dendritic_eligibility(
            P_dend, P_dend_atp, sens_dend, h2_h_prev, alpha_d,
        )

        # ── Accumulate ──
        A_r = A_r + r_E                                              # (N2,)
        C_soma2 = C_soma2 + sp_soma2[:, None] * E_soma2[None, :]     # (N2,N1)
        C_dend2 = C_dend2 + DGfac[:, None] * dmu2_atp               # (N2,N1)
        C_S2_soma = C_S2_soma + sp_soma2[:, None, None] * g1s[None, :, :]   # (N2,N1,K)
        C_S2_dend = C_S2_dend + sp_soma2[:, None, None] * g1d[None, :, :]
        C_D2_soma = C_D2_soma + DGfac[:, None, None] * P_soma_atp
        C_D2_dend = C_D2_dend + DGfac[:, None, None] * P_dend_atp

        nested = (P_soma, P_soma_atp, P_dend, P_dend_atp)
        A = (A_r, C_soma2, C_dend2, C_S2_soma, C_S2_dend, C_D2_soma, C_D2_dend)
        return (h1_carry, h2_carry, r_carry, nested, A), None

    init_carry = (h1_carry_init, h2_carry_init, r_carry_init, nested_init, A_init)
    scan_inputs = (dend_in1, soma_in1, x_input, time_indices, dropout_keys)
    final_carry, _ = lax.scan(step, init_carry, scan_inputs)

    _, _, r_carry_f, _, A_f = final_carry
    mean_voltage = r_carry_f[1] / T   # sum_v / T
    A_r, C_soma2, C_dend2, C_S2_soma, C_S2_dend, C_D2_soma, C_D2_dend = A_f
    return (mean_voltage, A_r, C_soma2, C_dend2,
            C_S2_soma, C_S2_dend, C_D2_soma, C_D2_dend)


def _predict_only_2l(
    x_input, w_dend1, w_soma1, w_dend2, w_soma2, w_readout,
    alpha_s, alpha_d, alpha_m, T_p1, T_p2, config, alpha_w,
):
    """Forward only — no gradient bookkeeping. Returns mean_voltage (J,)."""
    dend_in1 = x_input @ w_dend1.T
    soma_in1 = x_input @ w_soma1.T
    T = x_input.shape[0]
    n1, n2 = w_dend1.shape[0], w_dend2.shape[0]
    n_out = w_readout.shape[0]
    time_indices = jnp.arange(T, dtype=jnp.int32)

    def _zero_h(n, k):
        return (
            jnp.zeros(n), jnp.zeros(n),
            jnp.zeros(n, dtype=jnp.int32), jnp.zeros(n, dtype=jnp.int32),
            jnp.zeros(n), jnp.zeros(k),
            jnp.zeros((n, k)), jnp.zeros((n, k)), jnp.zeros(n),
        )

    def step(carry, inputs):
        h1_carry, h2_carry, r_carry = carry
        dend_in1_t, soma_in1_t, t = inputs

        h1_carry, o1, *_ = TwoCompNeuron.forward_step(
            h1_carry, dend_in1_t, soma_in1_t, t, alpha_s, alpha_d, T_p1, config, alpha_w,
        )
        o1_f = o1.astype(jnp.float64)
        h2_carry, o2, *_ = TwoCompNeuron.forward_step(
            h2_carry, o1_f @ w_dend2.T, o1_f @ w_soma2.T, t,
            alpha_s, alpha_d, T_p2, config, alpha_w,
        )
        r_carry, _, _ = LINeuron.forward_step(
            r_carry, o2.astype(jnp.float64), w_readout, alpha_m,
        )
        return (h1_carry, h2_carry, r_carry), None

    init = (_zero_h(n1, x_input.shape[1]), _zero_h(n2, n1),
            (jnp.zeros(n_out), jnp.zeros(n_out), jnp.zeros(n2)))
    final, _ = lax.scan(step, init, (dend_in1, soma_in1, time_indices))
    return final[2][1] / T   # readout sum_v / T


def _loss_and_grads_2l(
    mean_voltage, A_r, C_soma2, C_dend2,
    C_S2_soma, C_S2_dend, C_D2_soma, C_D2_dend,
    w_soma2, w_readout,
    target_smoothed, T, loss_temperature, loss_count_bias,
):
    """Loss + 5 weight gradients (w_dend1, w_soma1, w_dend2, w_soma2, w_readout)."""
    scaled_logits = mean_voltage / loss_temperature + loss_count_bias
    probs = jnp.exp(scaled_logits - jnp.max(scaled_logits))
    probs = probs / jnp.sum(probs)

    prediction = jnp.argmax(mean_voltage)
    loss = -jnp.sum(target_smoothed * jnp.log(probs + 1e-8))
    global_error = target_smoothed - probs            # (J,)
    G = global_error @ w_readout                       # (N2,) top-down learning signal

    grad_readout = (global_error[:, None] * A_r[None, :]) / T          # (J,N2)
    grad_soma2 = (G[:, None] * C_soma2) / (T * 8.0)                    # (N2,N1)
    grad_dend2 = (G[:, None] * C_dend2) / T                            # (N2,N1)

    # L1: combine L2-soma (current-time) and L2-dend (frozen t') branches.
    grad_soma1 = (
        jnp.einsum("m,mi,mik->ik", G, w_soma2, C_S2_soma)
        + jnp.einsum("m,mik->ik", G, C_D2_soma)
    ) / (T * 8.0)
    grad_dend1 = (
        jnp.einsum("m,mi,mik->ik", G, w_soma2, C_S2_dend)
        + jnp.einsum("m,mik->ik", G, C_D2_dend)
    ) / T

    return loss, prediction, grad_dend1, grad_soma1, grad_dend2, grad_soma2, grad_readout


def _apply_grads_2l(
    w_dend1, w_soma1, w_dend2, w_soma2, w_readout,
    g_dend1, g_soma1, g_dend2, g_soma2, g_readout,
    lr, clip_value, weight_decay,
):
    """SGD with decoupled weight decay for the 5 weight tensors."""
    def upd(w, g):
        g = jnp.clip(g, -clip_value, clip_value)
        return w + lr * g - lr * weight_decay * w
    return (upd(w_dend1, g_dend1), upd(w_soma1, g_soma1),
            upd(w_dend2, g_dend2), upd(w_soma2, g_soma2),
            upd(w_readout, g_readout))


def _adam_apply_2l(
    w_d1, w_s1, w_d2, w_s2, w_r,
    g_d1, g_s1, g_d2, g_s2, g_r,
    m_d1, m_s1, m_d2, m_s2, m_r,
    v_d1, v_s1, v_d2, v_s2, v_r,
    step, lr, beta1, beta2, eps, clip_value, weight_decay,
):
    """AdamW-style decoupled weight decay for the 5 weight tensors."""
    def update_one(w, g, m, v):
        g = jnp.clip(g, -clip_value, clip_value)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)
        w = w + lr * m_hat / (jnp.sqrt(v_hat) + eps) - lr * weight_decay * w
        return w, m, v

    w_d1, m_d1, v_d1 = update_one(w_d1, g_d1, m_d1, v_d1)
    w_s1, m_s1, v_s1 = update_one(w_s1, g_s1, m_s1, v_s1)
    w_d2, m_d2, v_d2 = update_one(w_d2, g_d2, m_d2, v_d2)
    w_s2, m_s2, v_s2 = update_one(w_s2, g_s2, m_s2, v_s2)
    w_r, m_r, v_r = update_one(w_r, g_r, m_r, v_r)
    return (w_d1, w_s1, w_d2, w_s2, w_r,
            m_d1, m_s1, m_d2, m_s2, m_r,
            v_d1, v_s1, v_d2, v_s2, v_r)


# ══════════════════════════════════════════════════════════════════════
#  vmap in_axes: which args get a batch dimension (0) vs stay shared (None)
#
#  For _forward_and_accum:
#    x_input → batched (B,T,K)       init carries → batched (B,...)
#    weights → shared                 config/alphas → shared
# ══════════════════════════════════════════════════════════════════════

_FWD_AXES = (
    0,                               # x_input
    None, None, None,                # w_dend, w_soma, w_readout
    None, None, None, None, None,    # alpha_s, alpha_d, alpha_m, T_p, config
    None,                            # alpha_w (shared)
    (0, 0, 0, 0, 0, 0, 0, 0, 0),    # h_carry_init (9-tuple, each batched)
    (0, 0, 0),                       # r_carry_init (3-tuple, each batched)
    0,                               # A_d_init
    0,                               # rng_key (per-sample)
    None,                            # dropout_rate (shared)
)

_PRED_AXES = (
    0,                            # x_input
    None, None, None,             # weights
    None, None, None, None, None, # alphas, T_p, config
    None,                         # alpha_w
)

_LOSS_AXES = (
    0, 0, 0, 0,                  # counts, A_r, A_s, A_d (per-sample)
    0,                            # target_smoothed (per-sample)
    None, None, None,             # T, loss_temperature, loss_count_bias
)


# ══════════════════════════════════════════════════════════════════════
#  Pre-compiled versions:
#    _*_single = jit(core_fn)                  → one sample
#    _*_batch  = jit(vmap(core_fn, in_axes=…)) → B samples in parallel
# ══════════════════════════════════════════════════════════════════════

_fwd_single = jit(_forward_and_accum)
_pred_single = jit(_predict_only)
_loss_single = jit(_loss_and_grads)
_apply = jit(_apply_grads)
_adam = jit(_adam_apply)

_fwd_batch = jit(vmap(_forward_and_accum, in_axes=_FWD_AXES))
_pred_batch = jit(vmap(_predict_only, in_axes=_PRED_AXES))
_loss_batch = jit(vmap(_loss_and_grads, in_axes=_LOSS_AXES))


# ── Two-layer vmap axes / pre-compiled versions ──

_FWD_AXES_2L = (
    0,                                  # x_input
    None, None, None, None, None,       # w_dend1, w_soma1, w_dend2, w_soma2, w_readout
    None, None, None,                   # alpha_s, alpha_d, alpha_m
    None, None, None, None,             # T_p1, T_p2, config, alpha_w
    (0,) * 9,                           # h1_carry_init
    (0,) * 9,                           # h2_carry_init
    (0, 0, 0),                          # r_carry_init
    (0, 0, 0, 0),                       # nested_init
    (0, 0, 0, 0, 0, 0, 0),             # A_init
    0,                                  # rng_key
    None,                               # dropout_rate
)

_PRED_AXES_2L = (
    0,                                  # x_input
    None, None, None, None, None,       # weights
    None, None, None,                   # alphas
    None, None, None, None,             # T_p1, T_p2, config, alpha_w
)

_LOSS_AXES_2L = (
    0, 0, 0, 0,                         # mean_voltage, A_r, C_soma2, C_dend2
    0, 0, 0, 0,                         # C_S2_soma, C_S2_dend, C_D2_soma, C_D2_dend
    None, None,                         # w_soma2, w_readout (shared)
    0,                                  # target_smoothed
    None, None, None,                   # T, loss_temperature, loss_count_bias
)

_fwd_single_2l = jit(_forward_and_accum_2l)
_pred_single_2l = jit(_predict_only_2l)
_loss_single_2l = jit(_loss_and_grads_2l)
_apply_2l = jit(_apply_grads_2l)
_adam_2l = jit(_adam_apply_2l)

_fwd_batch_2l = jit(vmap(_forward_and_accum_2l, in_axes=_FWD_AXES_2L))
_pred_batch_2l = jit(vmap(_predict_only_2l, in_axes=_PRED_AXES_2L))
_loss_batch_2l = jit(vmap(_loss_and_grads_2l, in_axes=_LOSS_AXES_2L))


# ══════════════════════════════════════════════════════════════════════
#  Network class — ties everything together
# ══════════════════════════════════════════════════════════════════════

class Network:
    def __init__(
        self,
        key: jnp.ndarray,
        n_inputs: int,
        n_hidden: int,
        n_outputs: int,
        config: NeuronConfig,
        optimizer: str = "sgd",
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_eps: float = 1e-8,
        dropout_rate: float = 0.0,
        weight_decay: float = 0.0,
        n_hidden2: int = 0,
    ):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_hidden2 = n_hidden2
        self.two_layer = n_hidden2 > 0
        self.n_outputs = n_outputs
        self.config = config
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay

        key_h, key_r, key_rng = random.split(key, 3)
        if self.two_layer:
            key_h1, key_h2 = random.split(key_h)
            self.hidden1 = TwoCompNeuron(key_h1, n_hidden, n_inputs, config)
            self.hidden2 = TwoCompNeuron(key_h2, n_hidden2, n_hidden, config)
            self.readout = LINeuron(key_r, n_outputs, n_hidden2, config)
        else:
            self.hidden = TwoCompNeuron(key_h, n_hidden, n_inputs, config)
            self.readout = LINeuron(key_r, n_outputs, n_hidden, config)
        self.rng_key = key_rng

        if optimizer == "adam":
            self.beta1 = beta1
            self.beta2 = beta2
            self.adam_eps = adam_eps
            self.adam_step = jnp.array(0, dtype=jnp.int32)
            if self.two_layer:
                self.m_dend1 = jnp.zeros_like(self.hidden1.w_dend)
                self.m_soma1 = jnp.zeros_like(self.hidden1.w_soma)
                self.m_dend2 = jnp.zeros_like(self.hidden2.w_dend)
                self.m_soma2 = jnp.zeros_like(self.hidden2.w_soma)
                self.m_readout = jnp.zeros_like(self.readout.w)
                self.v_dend1 = jnp.zeros_like(self.hidden1.w_dend)
                self.v_soma1 = jnp.zeros_like(self.hidden1.w_soma)
                self.v_dend2 = jnp.zeros_like(self.hidden2.w_dend)
                self.v_soma2 = jnp.zeros_like(self.hidden2.w_soma)
                self.v_readout = jnp.zeros_like(self.readout.w)
            else:
                self.m_dend = jnp.zeros_like(self.hidden.w_dend)
                self.m_soma = jnp.zeros_like(self.hidden.w_soma)
                self.m_readout = jnp.zeros_like(self.readout.w)
                self.v_dend = jnp.zeros_like(self.hidden.w_dend)
                self.v_soma = jnp.zeros_like(self.hidden.w_soma)
                self.v_readout = jnp.zeros_like(self.readout.w)

    # ── Helpers to build zero-initialized carries ──

    def _h_carry(self, B=None):
        """Hidden neuron carry. B=None → single sample, B=int → batched."""
        n, k = self.n_hidden, self.n_inputs
        s = (B, n) if B else (n,)
        sk = (B, k) if B else (k,)
        snk = (B, n, k) if B else (n, k)
        return (
            jnp.zeros(s), jnp.zeros(s),
            jnp.zeros(s, dtype=jnp.int32), jnp.zeros(s, dtype=jnp.int32),
            jnp.zeros(s), jnp.zeros(sk),
            jnp.zeros(snk), jnp.zeros(snk),
            jnp.zeros(s),  # w
        )

    def _r_carry(self, B=None):
        """Readout neuron carry."""
        j, n = self.n_outputs, self.n_hidden
        sj = (B, j) if B else (j,)
        sn = (B, n) if B else (n,)
        return (jnp.zeros(sj), jnp.zeros(sj), jnp.zeros(sn))

    def _A_d_zeros(self, B=None):
        """Dendritic accumulator zeros."""
        base = (self.n_outputs, self.n_hidden, self.n_inputs)
        return jnp.zeros((B,) + base if B else base)

    def _weights(self):
        return self.hidden.w_dend, self.hidden.w_soma, self.readout.w

    def _params(self):
        return (self.hidden.alpha_s, self.hidden.alpha_d, self.readout.alpha_m,
                self.hidden.T_p, self.config, self.hidden.alpha_w)

    # ── Two-layer carry / accumulator builders ──

    @staticmethod
    def _make_h_carry(n, k, B=None):
        """Generic 2-comp neuron carry for n neurons with k inputs."""
        s = (B, n) if B else (n,)
        sk = (B, k) if B else (k,)
        snk = (B, n, k) if B else (n, k)
        return (
            jnp.zeros(s), jnp.zeros(s),
            jnp.zeros(s, dtype=jnp.int32), jnp.zeros(s, dtype=jnp.int32),
            jnp.zeros(s), jnp.zeros(sk),
            jnp.zeros(snk), jnp.zeros(snk),
            jnp.zeros(s),  # w (adaptation)
        )

    def _r_carry2(self, B=None):
        """Readout carry when presynaptic source is L2 (n_hidden2 inputs)."""
        j, n = self.n_outputs, self.n_hidden2
        sj = (B, j) if B else (j,)
        sn = (B, n) if B else (n,)
        return (jnp.zeros(sj), jnp.zeros(sj), jnp.zeros(sn))

    def _nested_zeros(self, B=None):
        """Four (N2,N1,K) nested-trace tensors: P_soma, P_soma_atp, P_dend, P_dend_atp."""
        base = (self.n_hidden2, self.n_hidden, self.n_inputs)
        shape = (B,) + base if B else base
        return tuple(jnp.zeros(shape) for _ in range(4))

    def _A2_zeros(self, B=None):
        """Accumulators: A_r, C_soma2, C_dend2, C_S2_soma, C_S2_dend, C_D2_soma, C_D2_dend."""
        n1, n2, k = self.n_hidden, self.n_hidden2, self.n_inputs
        def z(*dims):
            return jnp.zeros((B,) + dims if B else dims)
        return (z(n2), z(n2, n1), z(n2, n1),
                z(n2, n1, k), z(n2, n1, k), z(n2, n1, k), z(n2, n1, k))

    def _weights2(self):
        return (self.hidden1.w_dend, self.hidden1.w_soma,
                self.hidden2.w_dend, self.hidden2.w_soma, self.readout.w)

    def _params2(self):
        return (self.hidden1.alpha_s, self.hidden1.alpha_d, self.readout.alpha_m,
                self.hidden1.T_p, self.hidden2.T_p, self.config, self.hidden1.alpha_w)

    def _smooth_targets(self, targets):
        """Scalar label or (B,) labels → smoothed one-hot vector(s)."""
        cfg = self.config
        one_hot = jnp.eye(self.n_outputs)[targets]
        return one_hot * (1 - cfg.loss_label_smoothing) + cfg.loss_label_smoothing / self.n_outputs

    def _update_weights(self, g_d, g_s, g_r, lr, clip_value):
        """Apply gradients using the configured optimizer (SGD or Adam).

        Both branches use decoupled weight decay (subtract lr·λ·w
        after the gradient step). For SGD this is equivalent to an
        L2 loss penalty (λ/2)·||w||²; for Adam this is the AdamW
        recipe — the decay does not pass through the 1/√v rescaling.
        """
        if self.optimizer == "adam":
            self.adam_step = self.adam_step + 1
            result = _adam(
                self.hidden.w_dend, self.hidden.w_soma, self.readout.w,
                g_d, g_s, g_r,
                self.m_dend, self.m_soma, self.m_readout,
                self.v_dend, self.v_soma, self.v_readout,
                self.adam_step, lr, self.beta1, self.beta2, self.adam_eps,
                clip_value, self.weight_decay,
            )
            (self.hidden.w_dend, self.hidden.w_soma, self.readout.w,
             self.m_dend, self.m_soma, self.m_readout,
             self.v_dend, self.v_soma, self.v_readout) = result
        else:
            self.hidden.w_dend, self.hidden.w_soma, self.readout.w = _apply(
                *self._weights(), g_d, g_s, g_r, lr, clip_value, self.weight_decay,
            )

    def _update_weights_2l(self, g_d1, g_s1, g_d2, g_s2, g_r, lr, clip_value):
        """Apply the 5 two-layer gradients with the configured optimizer."""
        if self.optimizer == "adam":
            self.adam_step = self.adam_step + 1
            result = _adam_2l(
                *self._weights2(),
                g_d1, g_s1, g_d2, g_s2, g_r,
                self.m_dend1, self.m_soma1, self.m_dend2, self.m_soma2, self.m_readout,
                self.v_dend1, self.v_soma1, self.v_dend2, self.v_soma2, self.v_readout,
                self.adam_step, lr, self.beta1, self.beta2, self.adam_eps,
                clip_value, self.weight_decay,
            )
            (self.hidden1.w_dend, self.hidden1.w_soma,
             self.hidden2.w_dend, self.hidden2.w_soma, self.readout.w,
             self.m_dend1, self.m_soma1, self.m_dend2, self.m_soma2, self.m_readout,
             self.v_dend1, self.v_soma1, self.v_dend2, self.v_soma2, self.v_readout) = result
        else:
            (self.hidden1.w_dend, self.hidden1.w_soma,
             self.hidden2.w_dend, self.hidden2.w_soma, self.readout.w) = _apply_2l(
                *self._weights2(), g_d1, g_s1, g_d2, g_s2, g_r,
                lr, clip_value, self.weight_decay,
            )

    # ── Single-sample API ──

    def _next_key(self):
        """Advance the PRNG and return a fresh subkey for dropout."""
        self.rng_key, subkey = random.split(self.rng_key)
        return subkey

    # ── Single-sample API ──

    def train_step(self, x_input, target, lr=1e-3, clip_value=1.0):
        """Train on one sample (with dropout during forward pass).
        Returns: (loss, prediction, grad_norms_dict)
        """
        T = x_input.shape[0]

        if self.two_layer:
            return self._train_step_2l(x_input, target, lr, clip_value)

        counts, A_r, A_s, A_d = _fwd_single(
            x_input, *self._weights(), *self._params(),
            self._h_carry(), self._r_carry(), self._A_d_zeros(),
            self._next_key(), self.dropout_rate,
        )

        loss, pred, g_r, g_s, g_d = _loss_single(
            counts, A_r, A_s, A_d,
            self._smooth_targets(target), T,
            self.config.loss_temperature, self.config.loss_count_bias,
        )

        gnorms = {
            "readout": float(jnp.linalg.norm(g_r)),
            "soma": float(jnp.linalg.norm(g_s)),
            "dend": float(jnp.linalg.norm(g_d)),
        }

        self._update_weights(g_d, g_s, g_r, lr, clip_value)
        return float(loss), int(pred), gnorms

    def _train_step_2l(self, x_input, target, lr, clip_value):
        """Two-layer single-sample training step."""
        T = x_input.shape[0]

        fwd = _fwd_single_2l(
            x_input, *self._weights2(), *self._params2(),
            self._make_h_carry(self.n_hidden, self.n_inputs),
            self._make_h_carry(self.n_hidden2, self.n_hidden),
            self._r_carry2(), self._nested_zeros(), self._A2_zeros(),
            self._next_key(), self.dropout_rate,
        )

        loss, pred, g_d1, g_s1, g_d2, g_s2, g_r = _loss_single_2l(
            *fwd, self.hidden2.w_soma, self.readout.w,
            self._smooth_targets(target), T,
            self.config.loss_temperature, self.config.loss_count_bias,
        )

        gnorms = {
            "readout": float(jnp.linalg.norm(g_r)),
            "soma2": float(jnp.linalg.norm(g_s2)),
            "dend2": float(jnp.linalg.norm(g_d2)),
            "soma1": float(jnp.linalg.norm(g_s1)),
            "dend1": float(jnp.linalg.norm(g_d1)),
        }

        self._update_weights_2l(g_d1, g_s1, g_d2, g_s2, g_r, lr, clip_value)
        return float(loss), int(pred), gnorms

    def predict(self, x_input):
        """Predict one sample (no dropout). x_input: (T,K) → int class label."""
        if self.two_layer:
            counts = _pred_single_2l(x_input, *self._weights2(), *self._params2())
        else:
            counts = _pred_single(x_input, *self._weights(), *self._params())
        return int(jnp.argmax(counts))

    # ── Batched API ──

    def batch_train_step(self, x_batch, targets, lr=1e-3, clip_value=1.0):
        """Train on B samples in parallel (with dropout).
        Returns: (mean_loss, predictions_array (B,), grad_norms_dict)
        """
        B = x_batch.shape[0]
        T = x_batch.shape[1]

        if self.two_layer:
            return self._batch_train_step_2l(x_batch, targets, lr, clip_value)

        batch_keys = random.split(self._next_key(), B)

        counts, A_r, A_s, A_d = _fwd_batch(
            x_batch, *self._weights(), *self._params(),
            self._h_carry(B), self._r_carry(B), self._A_d_zeros(B),
            batch_keys, self.dropout_rate,
        )

        losses, preds, g_r, g_s, g_d = _loss_batch(
            counts, A_r, A_s, A_d,
            self._smooth_targets(targets), T,
            self.config.loss_temperature, self.config.loss_count_bias,
        )

        g_r_avg = jnp.mean(g_r, axis=0)
        g_s_avg = jnp.mean(g_s, axis=0)
        g_d_avg = jnp.mean(g_d, axis=0)

        gnorms = {
            "readout": float(jnp.linalg.norm(g_r_avg)),
            "soma": float(jnp.linalg.norm(g_s_avg)),
            "dend": float(jnp.linalg.norm(g_d_avg)),
        }

        self._update_weights(g_d_avg, g_s_avg, g_r_avg, lr, clip_value)
        return float(jnp.mean(losses)), preds, gnorms

    def _batch_train_step_2l(self, x_batch, targets, lr, clip_value):
        """Two-layer batched training step."""
        B = x_batch.shape[0]
        T = x_batch.shape[1]
        batch_keys = random.split(self._next_key(), B)

        fwd = _fwd_batch_2l(
            x_batch, *self._weights2(), *self._params2(),
            self._make_h_carry(self.n_hidden, self.n_inputs, B),
            self._make_h_carry(self.n_hidden2, self.n_hidden, B),
            self._r_carry2(B), self._nested_zeros(B), self._A2_zeros(B),
            batch_keys, self.dropout_rate,
        )

        losses, preds, g_d1, g_s1, g_d2, g_s2, g_r = _loss_batch_2l(
            *fwd, self.hidden2.w_soma, self.readout.w,
            self._smooth_targets(targets), T,
            self.config.loss_temperature, self.config.loss_count_bias,
        )

        g_d1 = jnp.mean(g_d1, axis=0)
        g_s1 = jnp.mean(g_s1, axis=0)
        g_d2 = jnp.mean(g_d2, axis=0)
        g_s2 = jnp.mean(g_s2, axis=0)
        g_r = jnp.mean(g_r, axis=0)

        gnorms = {
            "readout": float(jnp.linalg.norm(g_r)),
            "soma2": float(jnp.linalg.norm(g_s2)),
            "dend2": float(jnp.linalg.norm(g_d2)),
            "soma1": float(jnp.linalg.norm(g_s1)),
            "dend1": float(jnp.linalg.norm(g_d1)),
        }

        self._update_weights_2l(g_d1, g_s1, g_d2, g_s2, g_r, lr, clip_value)
        return float(jnp.mean(losses)), preds, gnorms

    def batch_predict(self, x_batch):
        """Predict B samples in parallel. x_batch: (B,T,K) → (B,) int labels."""
        if self.two_layer:
            counts = _pred_batch_2l(x_batch, *self._weights2(), *self._params2())
        else:
            counts = _pred_batch(x_batch, *self._weights(), *self._params())
        return jnp.argmax(counts, axis=1)
