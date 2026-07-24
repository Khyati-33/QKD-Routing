import torch
import time
import numpy as np
import importlib, sys
import networkx as nx

# ── Fresh load of defence_training ───────────────────────────────────────────
if 'defence_training' in sys.modules:
    del sys.modules['defence_training']
import defence_training

# ── Load new GNNActorCritic with forward_batch ────────────────────────────────
with open('/content/gnn_model.py', 'r') as f:
    exec(compile(f.read(), 'gnn_model.py', 'exec'), globals())
assert hasattr(GNNActorCritic, 'forward_batch'), "forward_batch missing"
print("✓ GNNActorCritic loaded with forward_batch")

# ── Apply batched patch ───────────────────────────────────────────────────────
with open('/content/batched_ppo_patch.py', 'r') as f:
    exec(compile(f.read(), 'batched_ppo_patch.py', 'exec'), globals())

# ══════════════════════════════════════════════════════════════════════════════
# MID-TRAINING HEURISTIC COMPARISON
# Injected into train_ppo via monkey-patch on PPOBuffer so we don't need to
# modify defence_training.py. Instead we wrap train_ppo to add eval every 50.
# ══════════════════════════════════════════════════════════════════════════════

def _quick_eval(model, net, use_case, n_runs=3, n_steps=144):
    """
    Run n_runs episodes for each policy and return mean rewards.
    Uses 144 steps (6h window) for speed — enough for a relative comparison.
    """
    results = {}

    def _run_policy(policy_fn, model_to_eval=None):
        rewards = []
        for seed in range(n_runs):
            np.random.seed(seed)
            e = QKDRoutingEnv(net, use_case=use_case, terminate_on_depletion=True)
            obs, _ = e.reset()
            total_r = 0.
            for _ in range(n_steps):
                cur    = e._cur_node
                nb     = e.adj[cur]
                if not nb: break
                if model_to_eval is not None:
                    nx_, ei, ea = e.get_graph_data()
                    cur_idx = e.node2idx[cur]
                    nb_idxs = [e.node2idx[n] for n in nb]
                    with torch.no_grad():
                        action, _, _, _ = model_to_eval.get_action(
                            nx_, ei, ea, cur_idx, nb_idxs, deterministic=True)
                else:
                    action = policy_fn(e)
                obs, r, term, trunc, _ = e.step(action)
                total_r += r
                if term or trunc: break
            rewards.append(total_r)
        return float(np.mean(rewards))

    # ── Heuristics ────────────────────────────────────────────────────────────
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
        ns = e.net.get_state(e._t)
        return max(range(len(nb)),
                   key=lambda i: ns.get((cur, nb[i]),
                                  ns.get((nb[i], cur), {})).get('SKR', 0.))

    results['Random']       = _run_policy(random_policy)
    results['Shortest-Len'] = _run_policy(shortest_len)
    results['Dijkstra-Hop'] = _run_policy(dijkstra_hop)
    results['Max-SKR']      = _run_policy(max_skr)
    results['PPO+GNN']      = _run_policy(None, model_to_eval=model)
    return results


def _train_ppo_with_eval(env, model, net, use_case, n_epochs, eval_every=50,
                          **kwargs):
    """
    Wraps defence_training.train_ppo to inject heuristic comparison
    every eval_every epochs by splitting training into chunks.
    """
    all_logs = {k: [] for k in ['rewards','wq','skr','pool',
                                 'policy_loss','value_loss']}
    epoch_done = 0

    while epoch_done < n_epochs:
        chunk = min(eval_every, n_epochs - epoch_done)
        chunk_logs = defence_training.train_ppo(
            env, model,
            n_epochs    = chunk,
            use_case    = use_case,
            **kwargs
        )
        for k in all_logs:
            all_logs[k].extend(chunk_logs[k])
        epoch_done += chunk

        # ── Mid-training comparison ───────────────────────────────────────────
        print(f"\n{'─'*52}")
        print(f"  Mid-train eval @ epoch {epoch_done}/{n_epochs} "
              f"(3 runs × 144 steps each)")
        print(f"{'─'*52}")
        model.eval()
        comp = _quick_eval(model, net, use_case, n_runs=3, n_steps=144)
        for name, val in comp.items():
            marker = " ◄ PPO" if name == 'PPO+GNN' else ""
            print(f"  {name:15s}: {val:+.3f}{marker}")
        print(f"{'─'*52}\n")
        model.train()

        if epoch_done >= n_epochs:
            break

    return all_logs


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT + MODEL
# ══════════════════════════════════════════════════════════════════════════════

N_EPOCHS   = 200
USE_CASE   = "defence"

print(f"\n--- Training GNNEncoder for {USE_CASE.upper()} ---")
env_train_gnn_defence = QKDRoutingEnv(
    net, use_case=USE_CASE, terminate_on_depletion=False)

model_gnn_defence = GNNActorCritic(
    node_dim       = env_train_gnn_defence.node_feat_dim,
    edge_dim       = env_train_gnn_defence.edge_feat_dim,
    hidden         = 64,
    gnn_out        = 32,
    drop_edge_rate = 0.2
)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

defence_params = defence_training.USE_CASE_TRAINING[USE_CASE]
start_time     = time.time()

logs_gnn_defence = _train_ppo_with_eval(
    env          = env_train_gnn_defence,
    model        = model_gnn_defence,
    net          = net,
    use_case     = USE_CASE,
    n_epochs     = N_EPOCHS,
    eval_every   = 50,
    # ── all defence hyperparams passed through ────────────────────────────────
    steps_per_epoch     = defence_training.USE_CASE_STEPS[USE_CASE],
    lr                  = defence_params["lr"],
    clip_eps            = defence_params["clip_eps"],
    ent_coef_start      = defence_params["ent_coef_start"],
    ent_coef_end        = defence_params["ent_coef_end"],
    ent_decay_epochs    = defence_params["ent_decay_epochs"],
    vf_coef             = defence_params["vf_coef"],
    grad_clip           = defence_params["grad_clip"],
    n_update_iters      = defence_params["n_update_iters"],
    mini_batch_size     = defence_params["mini_batch_size"],
    lam                 = defence_params["lam"],
    early_stop_patience = defence_params["early_stop_patience"],
)

print(f"\n✓ GNNEncoder ({USE_CASE}) training complete in "
      f"{time.time() - start_time:.2f} seconds")
