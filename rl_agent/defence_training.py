import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from math import comb as mcomb
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch_geometric.data import Data, Batch
import networkx as nx
import time


# --- Constants ---
QBER_HARD = 0.11          # FIXED: was 0.49, now matches the BB84 finite-key
                           # distillation cutoff used throughout the report.
QBER_WARN = 0.08
MAX_SKR   = 5_000_000.0
MAX_LATENCY_S = 0.0002
MAX_ENERGY_J  = 300e-12
MAX_DEPLETION_RATE = 1.0
HOP_MAX   = 30             # NEW: tune to typical anchor-to-anchor route
                            # length on your 162-node corridor. Used only by
                            # the hop penalty below.

# --- USE_CASE_WEIGHTS ---
USE_CASE_WEIGHTS = {
    "defence":    dict(w_skr=2.0, w_pool=0.8, w_margin=0.8, w_hops=0.2, w_switch=0.2,
                       C_sec=30., C_dep=60., C_warn=8., C_goal=40.,
                       w_latency=0.4, w_energy=0.4, w_congestion=0.2),
    "commercial": dict(w_skr=1.2, w_pool=0.4, w_margin=0.2, w_hops=0.3, w_switch=0.3,
                       C_sec=20., C_dep=50., C_warn=5., C_goal=25.,
                       w_latency=0.6, w_energy=0.6, w_congestion=0.3),
    "research":   dict(w_skr=1.0, w_pool=0.4, w_margin=0.8, w_hops=0.15, w_switch=0.3,
                       C_sec=20., C_dep=5.0, C_warn=5., C_goal=25.,
                       w_latency=0.3, w_energy=0.3, w_congestion=0.2),
}
# NEW: C_goal added per use-case, terminal bonus for actually reaching the
# destination. Tune relative to C_sec/C_dep so reaching the destination is
# clearly worth more than a few extra steps of per-hop reward farming.

USE_CASE_TRAINING = {
    "research": {
        "lr": 3e-4,
        "clip_eps": 0.25,
        "ent_coef_start": 0.1,
        "ent_coef_end": 0.05,
        "ent_decay_epochs": 200,
        "vf_coef": 0.5,
        "grad_clip": 0.5,
        "n_update_iters": 4,
        "mini_batch_size": 64,
        "lam": 0.95,
        "early_stop_patience": 999,
    },
    "commercial": {
        "lr": 5e-4,
        "clip_eps": 0.25,
        "ent_coef_start": 0.12,
        "ent_coef_end": 0.04,
        "ent_decay_epochs": 120,
        "vf_coef": 0.5,
        "grad_clip": 0.5,
        "n_update_iters": 6,
        "mini_batch_size": 64,
        "lam": 0.95,
        "early_stop_patience": 20,
    },
    "defence": {
        "lr": 8e-4,
        "clip_eps": 0.3,
        "ent_coef_start": 0.15,
        "ent_coef_end": 0.03,
        "ent_decay_epochs": 80,
        "vf_coef": 0.6,
        "grad_clip": 1.0,
        "n_update_iters": 8,
        "mini_batch_size": 32,
        "lam": 0.95,
        "early_stop_patience": 12,
    },
}

USE_CASE_STEPS = {
    "research": 576,
    "commercial": 288,
    "defence": 144,
}


def binary_entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def total_chain_noise(Q: float, p: int) -> float:
    n_links = p + 1
    w = 0.0
    for i in range(n_links // 2):
        w += mcomb(n_links, 2 * i + 1) * (Q ** (2 * i + 1)) * ((1 - Q) ** (n_links - 2 * i - 1))
    return w


class PPOBuffer:
    """Rollout buffer for PPO on-policy updates."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs_node_x      = []
        self.obs_ei          = []
        self.obs_ea          = []
        self.cur_idxs        = []
        self.nb_idxs         = []
        self.actions         = []
        self.log_probs       = []
        self.values          = []
        self.rewards         = []
        self.dones           = []
        self.graphs          = []
        self.nb_idxs_stored  = []

    def add(self, nx_, ei, ea, cur_idx, nb_idxs,
            action, log_prob, value, reward, done):
        self.obs_node_x.append(nx_)
        self.obs_ei.append(ei)
        self.obs_ea.append(ea)
        self.cur_idxs.append(cur_idx)
        self.nb_idxs.append(nb_idxs)
        self.actions.append(action)
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns(self, gamma=0.99, lam=0.95, device=None):
        T          = len(self.rewards)
        dev        = device if device is not None else self.values[0].device
        advantages = torch.zeros(T, device=dev)
        gae        = 0.
        for t in reversed(range(T)):
            next_val = (self.values[t + 1].to(dev) if t < T - 1
                        else torch.tensor(0., device=dev))
            delta    = (self.rewards[t]
                        + gamma * next_val * (1 - self.dones[t])
                        - self.values[t].to(dev))
            gae      = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages[t] = gae
        returns = advantages + torch.stack([v.to(dev) for v in self.values])
        return advantages, returns


def compute_reward(path, net_state, old_pools, new_pools,
                   switched, use_case="research", pool_cap=1e6,
                   cumulative_n_stn=0,
                   destination_node=None, next_node_in_path=None,
                   env_instance=None):
    if env_instance is None:
        raise ValueError("compute_reward must be called with env_instance")

    w = USE_CASE_WEIGHTS[use_case]
    path_links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    n_stn_hop = sum(
        1 for n in path[1:-1]
        if env_instance.net.config["node_types"].get(n, "stn") == "stn"
    )

    qbers = [
        net_state.get(lk, net_state.get((lk[1], lk[0]), {})).get('QBER', 0.5)
        for lk in path_links
    ]
    skrs = [
        net_state.get(lk, net_state.get((lk[1], lk[0]), {})).get('SKR', 0.)
        for lk in path_links
    ]

    # NEW: track QBER/SKR across the whole path so far, not just this hop.
    # Reset the rolling history whenever the env is at the start of a fresh
    # episode (path_so_far length <= 1, i.e. just the source node).
    if len(env_instance._path_so_far) <= 1:
        env_instance._path_qbers = []
        env_instance._path_skrs  = []
    if not hasattr(env_instance, '_path_qbers'):
        env_instance._path_qbers = []
    if not hasattr(env_instance, '_path_skrs'):
        env_instance._path_skrs = []
    env_instance._path_qbers.extend(qbers)
    env_instance._path_skrs.extend(skrs)

    valid_qbers = [q for q in qbers if q < 0.5]
    mean_Q      = float(np.mean(valid_qbers)) if valid_qbers else 0.5
    worst_qber  = max(valid_qbers) if valid_qbers else 0.5

    # Path-wide worst QBER, used only for the hard gate below, so a route
    # that was briefly noisy several hops ago still gets caught even if the
    # current hop looks clean.
    path_valid_qbers = [q for q in env_instance._path_qbers if q < 0.5]
    path_worst_qber  = max(path_valid_qbers) if path_valid_qbers else worst_qber

    w_q = total_chain_noise(mean_Q, cumulative_n_stn)

    if w_q >= QBER_HARD or worst_qber >= QBER_HARD or path_worst_qber >= QBER_HARD:
        return -w['C_sec'], {
            'security_hard': -w['C_sec'],
            'w_q': w_q,
            'total': -w['C_sec'],
        }

    r     = 0.0
    comps = {'w_q': w_q, 'n_stn_hop': n_stn_hop,
             'cumulative_n_stn': cumulative_n_stn}

    if w_q > QBER_WARN:
        pen = -2.0 * (w_q - QBER_WARN) / (QBER_HARD - QBER_WARN)
        r  += pen
        comps['security_soft'] = pen

    qber_r = w['w_margin'] * (1.0 - worst_qber / QBER_HARD)
    r     += qber_r
    comps['qber_margin'] = qber_r

    # Bottleneck SKR now taken over the whole path so far, not just this hop.
    btn_skr = min(env_instance._path_skrs) if env_instance._path_skrs else 0.
    skr_r   = w['w_skr'] * float(np.clip(btn_skr / MAX_SKR, 0., 1.))
    r      += skr_r
    comps['skr'] = skr_r

    current_full_path    = env_instance._path_so_far
    total_path_latency_s = 0.0
    total_path_energy_J  = 0.0

    if len(current_full_path) > 1:
        for i in range(len(current_full_path) - 1):
            node_u, node_v = current_full_path[i], current_full_path[i + 1]
            link_obj = env_instance.net.links.get(
                (node_u, node_v),
                env_instance.net.links.get((node_v, node_u))
            )
            if link_obj:
                total_path_latency_s += link_obj.latency_s()
                total_path_energy_J  += link_obj.energy_per_bit_J()

    latency_pen = -w['w_latency'] * float(np.clip(total_path_latency_s / MAX_LATENCY_S, 0., 1.))
    r          += latency_pen
    comps['latency'] = latency_pen

    energy_pen = -w['w_energy'] * float(np.clip(total_path_energy_J / MAX_ENERGY_J, 0., 1.))
    r         += energy_pen
    comps['energy'] = energy_pen

    # NEW: hop-count penalty (w_hops was previously defined but unused).
    hop_pen = -w['w_hops'] * float(np.clip(len(current_full_path) / HOP_MAX, 0., 1.))
    r      += hop_pen
    comps['hops'] = hop_pen

    pool_ratios = []
    for lk in path_links:
        pool = new_pools.get(lk, new_pools.get((lk[1], lk[0]), 0.))
        pool_ratios.append(pool / pool_cap)
    min_pr       = min(pool_ratios) if pool_ratios else 0.
    pool_level_r = w['w_pool'] * float(np.clip(min_pr, 0., 1.))
    r           += pool_level_r
    comps['pool_level'] = pool_level_r

    pool_delta_ratios = []
    for lk in path_links:
        new_val = new_pools.get(lk, new_pools.get((lk[1], lk[0]), 0.))
        old_val = old_pools.get(lk, old_pools.get((lk[1], lk[0]), 0.))
        pool_delta_ratios.append((new_val - old_val) / pool_cap)
    mean_delta   = float(np.mean(pool_delta_ratios)) if pool_delta_ratios else 0.
    pool_delta_r = w['w_pool'] * 0.2 * float(np.clip(mean_delta, -1., 1.))
    r           += pool_delta_r
    comps['pool_delta'] = pool_delta_r

    depletion_rate_l    = max(0.0, -mean_delta)
    congestion_pen      = -w['w_congestion'] * (np.clip(depletion_rate_l / MAX_DEPLETION_RATE, 0., 1.) ** 2)
    r                  += congestion_pen
    comps['congestion_penalty'] = congestion_pen

    if min_pr <= 0.:
        r -= w['C_dep']
        comps['depletion'] = -w['C_dep']
    elif min_pr < 0.1:
        pen = -w['C_warn'] * (1. - min_pr / 0.1)
        r  += pen
        comps['depletion'] = pen
    else:
        comps['depletion'] = 0.

    if switched and w_q < QBER_WARN:
        r -= w['w_switch']
        comps['switch'] = -w['w_switch']
    else:
        comps['switch'] = 0.

    # NEW: terminal bonus for actually reaching the destination. Without
    # this, a policy that never completes the route can out-score one that
    # does, since completion only ever ends reward accumulation early.
    if destination_node is not None and next_node_in_path == destination_node:
        r += w['C_goal']
        comps['goal_bonus'] = w['C_goal']
    else:
        comps['goal_bonus'] = 0.

    comps['total'] = r
    return r, comps


def _nb(name):
    import sys
    return getattr(sys.modules['__main__'], name)


def _quick_eval(model, net, use_case, n_runs=3, n_steps=144, season='monsoon'):
    """Evaluate PPO+GNN and heuristics over n_runs episodes.
    NOTE: season is now passed through explicitly and threaded into both
    the environment construction and the Max-SKR heuristic's own state
    query, so heuristics decide under the same physics the environment
    will actually execute and reward under.
    """
    def _run_policy(policy_fn, model_to_eval=None):
        rewards = []
        for seed in range(n_runs):
            np.random.seed(seed)
            e = _nb('QKDRoutingEnv')(net, use_case=use_case,
                                     terminate_on_depletion=False,
                                     season=season)
            obs, _ = e.reset()
            total_r = 0.
            for _ in range(n_steps):
                cur = e._cur_node
                nb  = e.adj[cur]
                if not nb:
                    break
                if model_to_eval is not None:
                    nx_, ei, ea   = e.get_graph_data()
                    cur_idx       = e.node2idx[cur]
                    nb_idxs       = [e.node2idx[n] for n in nb]
                    dest_node_idx = e.node2idx[e.dest]
                    with torch.no_grad():
                        action, _, _, _ = model_to_eval.get_action(
                            nx_, ei, ea, cur_idx, nb_idxs,
                            dest_node_idx, deterministic=True)
                else:
                    action = policy_fn(e)
                obs, r, term, trunc, _ = e.step(action)
                total_r += r
                if term or trunc:
                    break
            rewards.append(total_r)
        return float(np.mean(rewards))

    def random_policy(e):
        nb = e.adj[e._cur_node]
        return np.random.randint(len(nb)) if nb else 0

    def shortest_len(e):
        cur, dst, nb = e._cur_node, e.dest, e.adj[e._cur_node]
        if not nb: return 0
        try:
            path = nx.shortest_path(e.net.graph, cur, dst, weight='weight')
            if len(path) > 1 and path[1] in nb:
                return nb.index(path[1])
        except: pass
        return 0

    def dijkstra_hop(e):
        cur, dst, nb = e._cur_node, e.dest, e.adj[e._cur_node]
        if not nb: return 0
        try:
            path = nx.shortest_path(e.net.graph, cur, dst)
            if len(path) > 1 and path[1] in nb:
                return nb.index(path[1])
        except: pass
        return 0

    def max_skr(e):
        cur, nb = e._cur_node, e.adj[e._cur_node]
        if not nb: return 0
        ns = e.net.get_state(e._t, season=e.season)   # FIXED: was e.net.get_state(e._t)
        return max(range(len(nb)),
                   key=lambda i: ns.get(
                       (cur, nb[i]),
                       ns.get((nb[i], cur), {})).get('SKR', 0.))

    return {
        'Random':       _run_policy(random_policy),
        'Shortest-Len': _run_policy(shortest_len),
        'Dijkstra-Hop': _run_policy(dijkstra_hop),
        'Max-SKR':      _run_policy(max_skr),
        'PPO+GNN':      _run_policy(None, model_to_eval=model),
    }


def train_ppo_with_eval(env, model, net, use_case, n_epochs,
                        eval_every=50, season='monsoon', **kwargs):
    """
    PPO training loop with batched GNN forward and device-aware tensor handling.
    Pass device=torch.device('cuda') (or 'cpu') via kwargs.
    NOTE: season is now an explicit parameter, threaded into mid-train eval
    so heuristic baselines are compared under the same conditions training
    ran under.
    """
    # ── Resolve device ────────────────────────────────────────────────
    device = kwargs.get('device', next(model.parameters()).device)

    hp               = USE_CASE_TRAINING[use_case]
    lr               = kwargs.get('lr',               hp['lr'])
    clip_eps         = kwargs.get('clip_eps',         hp['clip_eps'])
    ent_coef_start   = kwargs.get('ent_coef_start',   hp['ent_coef_start'])
    ent_coef_end     = kwargs.get('ent_coef_end',     hp['ent_coef_end'])
    ent_decay_epochs = kwargs.get('ent_decay_epochs', hp['ent_decay_epochs'])
    vf_coef          = kwargs.get('vf_coef',          hp['vf_coef'])
    grad_clip        = kwargs.get('grad_clip',        hp['grad_clip'])
    n_update_iters   = kwargs.get('n_update_iters',   hp['n_update_iters'])
    mini_batch_size  = kwargs.get('mini_batch_size',  hp['mini_batch_size'])
    gamma            = kwargs.get('gamma',            0.99)
    lam              = kwargs.get('lam',              hp['lam'])
    steps_per_epoch  = kwargs.get('steps_per_epoch',  USE_CASE_STEPS[use_case])

    # ── Optimizer + scheduler (created once) ─────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    warmup    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                         total_iters=10)
    cosine    = CosineAnnealingLR(optimizer, T_max=max(n_epochs - 10, 1),
                                  eta_min=lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[10])

    all_logs = {k: [] for k in ['rewards', 'wq', 'skr', 'pool',
                                 'policy_loss', 'value_loss']}
    best_avg_reward  = -np.inf
    patience_counter = 0
    obs, _ = env.reset()

    print(f"Training PPO+GNN [BATCHED-GNN, PERSISTENT-OPT]  "
          f"|  {n_epochs} epochs × {steps_per_epoch} steps")
    print(f"Use-case: {use_case}  |  Season: {season}  |  Clip ε={clip_eps}  |  LR={lr}")
    print(f"Mini-batch: {mini_batch_size}  |  Update iters: {n_update_iters}")
    print(f"Entropy: {ent_coef_start} → {ent_coef_end} over {ent_decay_epochs} epochs")
    print(f"GAE λ={lam}")
    print(f"Device: {device}")
    print("—" * 60)

    for epoch in range(n_epochs):
        buf            = PPOBuffer()
        ep_rewards     = []
        ep_wq, ep_skr, ep_pool = [], [], []
        step_durations = []

        current_ent_coef = ent_coef_start - (
            (ent_coef_start - ent_coef_end) *
            min(1.0, epoch / ent_decay_epochs))

        # ── Rollout ───────────────────────────────────────────────────
        model.eval()
        for _ in range(steps_per_epoch):
            nx_, ei, ea = env.get_graph_data()

            assert not torch.isnan(nx_).any(), f"NaN in node features at node {env._cur_node}"
            assert not torch.isinf(nx_).any(), f"Inf in node features at node {env._cur_node}"
            assert not torch.isnan(ea).any(),  f"NaN in edge attrs at node {env._cur_node}"
            assert not torch.isinf(ea).any(),  f"Inf in edge attrs at node {env._cur_node}"
            assert ei.max() < nx_.shape[0],    f"Edge index OOB: {ei.max()} >= {nx_.shape[0]}"

            cur_idx       = env.node2idx[env._cur_node]
            nb_idxs       = [env.node2idx[n] for n in env.adj[env._cur_node]]
            dest_node_idx = env.node2idx[env.dest]

            if not nb_idxs:
                obs, _ = env.reset()
                continue

            # Cast graph tensors to device before model sees them
            nx_d = nx_.to(device)
            ei_d = ei.to(device)
            ea_d = ea.to(device)

            with torch.no_grad():
                action, log_prob, entropy, value = model.get_action(
                    nx_d, ei_d, ea_d, cur_idx, nb_idxs, dest_node_idx)

            t0 = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            step_durations.append(time.time() - t0)

            buf.add(nx_d, ei_d, ea_d, cur_idx, nb_idxs,
                    action, log_prob, value, reward,
                    float(terminated or truncated))
            buf.graphs.append(Data(
                x          = nx_d.clone(),
                edge_index = ei_d.clone(),
                edge_attr  = ea_d.clone(),
                cur_idx    = torch.tensor([cur_idx], dtype=torch.long, device=device),
                num_nodes  = nx_d.shape[0],
            ))
            buf.nb_idxs_stored.append(list(nb_idxs))

            ep_rewards.append(reward)
            ep_wq.append(info.get('w_q',         0.))
            ep_skr.append(info.get('skr',         0.))
            ep_pool.append(info.get('pool_level',  1.))

            if terminated or truncated:
                obs, _ = env.reset()

        # ── Returns & advantages ──────────────────────────────────────
        advantages, returns = buf.compute_returns(gamma, lam, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── PPO update ────────────────────────────────────────────────
        model.train()
        ep_pol_loss, ep_val_loss = [], []
        buf_size = len(buf.rewards)

        for _ in range(n_update_iters):
            perm = np.random.permutation(buf_size)
            for start in range(0, buf_size, mini_batch_size):
                batch_idx  = perm[start: start + mini_batch_size]
                n          = len(batch_idx)

                graph_list = [buf.graphs[i] for i in batch_idx]
                pyg_batch  = Batch.from_data_list(graph_list).to(device)  # ← whole batch to device

                offsets, offset = [], 0
                for g in graph_list:
                    offsets.append(offset)
                    offset += g.num_nodes

                pyg_batch.cur_idxs = torch.tensor(
                    [graph_list[j].cur_idx.item() + offsets[j] for j in range(n)],
                    dtype=torch.long, device=device)                       # ← device
                pyg_batch.nb_idxs  = [
                    [i_ + offsets[j] for i_ in buf.nb_idxs_stored[batch_idx[j]]]
                    for j in range(n)
                ]
                pyg_batch.dest_idxs = torch.tensor(
                    [env.node2idx[env.dest] + offsets[j] for j in range(n)],
                    dtype=torch.long, device=device)                       # ← device

                logits_list, values_b = model.forward_batch(pyg_batch)

                max_nb     = max(len(l) for l in logits_list)
                logits_pad = torch.stack([
                    F.pad(l, (0, max_nb - len(l)), value=-1e9)
                    for l in logits_list
                ])                                                         # already on device (from model)

                # ── All rollout tensors cast to device ────────────────
                old_lps = torch.stack(
                    [buf.log_probs[i] for i in batch_idx]
                ).to(device)

                acts = torch.tensor(
                    [buf.actions[i] % len(buf.nb_idxs_stored[i])
                     for i in batch_idx],
                    dtype=torch.long, device=device)                       # ← device

                advs = advantages[batch_idx].to(device)
                rets = returns[batch_idx].to(device)

                dist      = Categorical(logits=logits_pad)
                new_lps   = dist.log_prob(acts)
                entropies = dist.entropy()

                ratio = torch.exp(new_lps - old_lps)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs

                pol_loss = -torch.min(surr1, surr2).mean()
                val_loss = F.mse_loss(values_b.squeeze(-1), rets.detach().squeeze(-1))
                ent_loss = entropies.mean()
                loss     = pol_loss + vf_coef * val_loss - current_ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                ep_pol_loss.append(pol_loss.item())
                ep_val_loss.append(val_loss.item())

        scheduler.step()

        # ── Logging ───────────────────────────────────────────────────
        mean_r    = float(np.mean(ep_rewards))
        mean_wq   = float(np.mean(ep_wq))
        mean_skr  = float(np.mean(ep_skr))
        mean_pool = float(np.mean(ep_pool))
        mean_step = np.mean(step_durations) if step_durations else 0.0

        all_logs['rewards'].append(mean_r)
        all_logs['wq'].append(mean_wq)
        all_logs['skr'].append(mean_skr)
        all_logs['pool'].append(mean_pool)
        all_logs['policy_loss'].append(float(np.mean(ep_pol_loss)))
        all_logs['value_loss'].append(float(np.mean(ep_val_loss)))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"R={mean_r:+7.3f} | w(q)={mean_wq:.4f} | "
                  f"SKR={mean_skr:.3f} | Pool={mean_pool:.3f} | "
                  f"ent={current_ent_coef:.4f} | "
                  f"LR={scheduler.get_last_lr()[0]:.5f} | "
                  f"StepTime={mean_step:.4f}s")

        # ── Mid-training eval ─────────────────────────────────────────
        if (epoch + 1) % eval_every == 0 or (epoch + 1) == n_epochs:
            print(f"\n{'—'*54}")
            print(f"  Mid-train eval @ epoch {epoch+1}/{n_epochs} "
                  f"(3 runs × 144 steps, no early termination)")
            print(f"{'—'*54}")
            model.eval()
            comp = _quick_eval(model, net, use_case, n_runs=3, n_steps=144,
                               season=season)
            for name, val in comp.items():
                marker = " ◄ PPO" if name == 'PPO+GNN' else ""
                print(f"  {name:15s}: {val:+.3f}{marker}")
            print(f"{'—'*54}\n")
            model.train()

    return all_logs