"""
Batched PPO minibatch loop patch.

Run this cell ONCE before the training cell.
To disable: restart kernel and skip this cell.

Requires:
  - gnn_model.py already run (GNNActorCritic with forward_batch defined)
  - torch_geometric installed: pip install torch-geometric

Monkey-patches defence_training.train_ppo to use forward_batch()
inside the update loop, giving a true batched GNN forward pass
(one GNN call per minibatch instead of 64 serial calls).

Expected speedup: 5-8x on the PPO update loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data, Batch
import numpy as np
import defence_training


def _build_pyg_data(node_x, edge_index, edge_attr, cur_idx, nb_idxs):
    """
    Pack a single rollout observation into a PyG Data object.
    Called during rollout collection to pre-build Data objects for the buffer.
    cur_idx and nb_idxs stored as plain Python attributes (not tensors)
    so Batch.from_data_list handles them correctly.
    """
    return Data(
        x          = node_x.clone(),
        edge_index = edge_index.clone(),
        edge_attr  = edge_attr.clone(),
        cur_idx    = torch.tensor([cur_idx], dtype=torch.long),
        # nb_idxs stored as list — Batch won't auto-collate but we handle it
        num_nodes  = node_x.shape[0],
    ), nb_idxs   # return separately; stored in buf.nb_idxs_stored


def _build_batch(buf, batch_idx):
    """
    Build a PyG Batch from stored Data objects for a minibatch.
    Attaches nb_idxs as a plain list attribute after batching.
    """
    graph_list = [buf.graphs[i] for i in batch_idx]
    batched    = Batch.from_data_list(graph_list)

    # Compute per-graph node offsets so forward_batch can convert
    # local buffer indices → global (offset) indices in the merged graph.
    # PyG offsets edge_index automatically; we need to offset cur_idx too.
    offsets = []
    offset  = 0
    for g in graph_list:
        offsets.append(offset)
        offset += g.num_nodes

    # Override cur_idxs with offset-adjusted global indices
    batched.cur_idxs = torch.tensor(
        [graph_list[j].cur_idx.item() + offsets[j]
         for j in range(len(batch_idx))],
        dtype=torch.long
    )

    # Attach nb_idxs as list of lists with global offsets applied
    batched.nb_idxs = [
        [idx + offsets[j] for idx in buf.nb_idxs_stored[batch_idx[j]]]
        for j in range(len(batch_idx))
    ]

    return batched


def _batched_train_ppo(env, model,
                       n_epochs=200, steps_per_epoch=None,
                       use_case="research",
                       lr=None, clip_eps=None,
                       ent_coef_start=None, ent_coef_end=None,
                       ent_decay_epochs=None, vf_coef=None,
                       grad_clip=None, n_update_iters=None,
                       mini_batch_size=None, gamma=0.99,
                       lam=None, early_stop_patience=None):

    # ── Resolve hyperparameters ───────────────────────────────────────────────
    hp                  = defence_training.USE_CASE_TRAINING[use_case]
    lr                  = lr                  or hp['lr']
    clip_eps            = clip_eps            or hp['clip_eps']
    ent_coef_start      = ent_coef_start      or hp['ent_coef_start']
    ent_coef_end        = ent_coef_end        or hp['ent_coef_end']
    ent_decay_epochs    = ent_decay_epochs    or hp['ent_decay_epochs']
    vf_coef             = vf_coef             or hp['vf_coef']
    grad_clip           = grad_clip           or hp['grad_clip']
    n_update_iters      = n_update_iters      or hp['n_update_iters']
    mini_batch_size     = mini_batch_size     or hp['mini_batch_size']
    lam                 = lam                 or hp['lam']
    early_stop_patience = early_stop_patience or hp['early_stop_patience']
    steps_per_epoch     = steps_per_epoch     or defence_training.USE_CASE_STEPS[use_case]

    PPOBuffer = defence_training._nb('PPOBuffer')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # T_max fixed at 200 so LR decay is consistent regardless of n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200, eta_min=lr * 0.01)

    epoch_rewards, epoch_wq             = [], []
    epoch_skr,     epoch_pool           = [], []
    policy_losses, value_losses         = [], []
    patience_count = 0
    obs, _ = env.reset()

    print(f"Training PPO+GNN [BATCHED-GNN]  |  {n_epochs} epochs × {steps_per_epoch} steps")
    print(f"Use-case: {use_case}  |  Clip ε={clip_eps}  |  LR={lr}")
    print(f"Mini-batch: {mini_batch_size}  |  Update iters: {n_update_iters}")
    print(f"Entropy: {ent_coef_start} → {ent_coef_end} over {ent_decay_epochs} epochs")
    print(f"GAE λ={lam}  |  Early-stop patience={early_stop_patience}")
    print("—" * 60)

    for epoch in range(n_epochs):
        buf        = PPOBuffer()
        # Extra storage for PyG Data objects and nb_idxs
        buf.graphs         = []
        buf.nb_idxs_stored = []

        ep_rewards = []
        ep_wq, ep_skr, ep_pool = [], [], []

        current_ent_coef = ent_coef_start - (
            (ent_coef_start - ent_coef_end) *
            min(1.0, epoch / ent_decay_epochs))

        # ── Rollout collection ────────────────────────────────────────────────
        model.eval()
        for _ in range(steps_per_epoch):
            nx_, ei, ea = env.get_graph_data()
            cur_idx = env.node2idx[env._cur_node]
            nb_idxs = [env.node2idx[n] for n in env.adj[env._cur_node]]
            if not nb_idxs:
                obs, _ = env.reset()
                continue

            with torch.no_grad():
                action, log_prob, entropy, value = model.get_action(
                    nx_, ei, ea, cur_idx, nb_idxs)

            obs, reward, terminated, truncated, info = env.step(action)

            # Store in PPOBuffer as usual
            buf.add(nx_, ei, ea, cur_idx, nb_idxs,
                    action, log_prob, value, reward,
                    float(terminated or truncated))

            # Also store PyG Data object for batched update
            data, nb = _build_pyg_data(nx_, ei, ea, cur_idx, nb_idxs)
            buf.graphs.append(data)
            buf.nb_idxs_stored.append(nb)

            ep_rewards.append(reward)
            ep_wq.append(info.get('w_q',        0.))
            ep_skr.append(info.get('skr',        0.))
            ep_pool.append(info.get('pool_level', 1.))

            if terminated or truncated:
                obs, _ = env.reset()

        # ── Returns & advantages ──────────────────────────────────────────────
        advantages, returns = buf.compute_returns(gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── BATCHED GNN PPO update ────────────────────────────────────────────
        model.train()
        ep_pol_loss, ep_val_loss = [], []
        buf_size = len(buf.rewards)

        for _ in range(n_update_iters):
            idx = np.random.permutation(buf_size)
            for start in range(0, buf_size, mini_batch_size):
                batch_idx = idx[start: start + mini_batch_size]
                n         = len(batch_idx)

                # ── Build PyG batch — one GNN call for all n graphs ───────────
                pyg_batch = _build_batch(buf, batch_idx)

                old_lps = torch.stack(
                    [buf.log_probs[i] for i in batch_idx])
                acts    = torch.tensor(
                    [buf.actions[i] % len(buf.nb_idxs_stored[i])
                     for i in batch_idx], dtype=torch.long)
                advs    = advantages[batch_idx]
                rets    = returns[batch_idx]

                # ── Single batched GNN forward ────────────────────────────────
                logits_list, values_b = model.forward_batch(pyg_batch)

                # ── Pad logits to max_nb for vectorised loss ──────────────────
                max_nb     = max(len(l) for l in logits_list)
                logits_pad = torch.stack([
                    F.pad(l, (0, max_nb - len(l)), value=-1e9)
                    for l in logits_list
                ])   # (n, max_nb)

                # ── Vectorised PPO loss ───────────────────────────────────────
                dist      = Categorical(logits=logits_pad)
                new_lps   = dist.log_prob(acts)
                entropies = dist.entropy()

                ratio = torch.exp(new_lps - old_lps)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs

                pol_loss = -torch.min(surr1, surr2).mean()
                val_loss = F.mse_loss(values_b, rets.detach())
                ent_loss = entropies.mean()

                loss = pol_loss + vf_coef * val_loss - current_ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                ep_pol_loss.append(pol_loss.item())
                ep_val_loss.append(val_loss.item())

        scheduler.step()

        # ── Logging ───────────────────────────────────────────────────────────
        mean_r    = float(np.mean(ep_rewards))
        mean_wq   = float(np.mean(ep_wq))
        mean_skr  = float(np.mean(ep_skr))
        mean_pool = float(np.mean(ep_pool))

        epoch_rewards.append(mean_r);  epoch_wq.append(mean_wq)
        epoch_skr.append(mean_skr);    epoch_pool.append(mean_pool)
        policy_losses.append(float(np.mean(ep_pol_loss)))
        value_losses.append(float(np.mean(ep_val_loss)))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"R={mean_r:+7.3f} | w(q)={mean_wq:.4f} | "
                  f"SKR={mean_skr:.3f} | Pool={mean_pool:.3f} | "
                  f"ent={current_ent_coef:.4f} | "
                  f"LR={scheduler.get_last_lr()[0]:.5f}")

        if len(epoch_rewards) >= early_stop_patience * 2:
            recent = float(np.mean(epoch_rewards[-early_stop_patience:]))
            prior  = float(np.mean(
                epoch_rewards[-early_stop_patience*2:-early_stop_patience]))
            if recent > 0 and (recent - prior) < 0.05:
                patience_count += 1
                if patience_count >= 3:
                    print(f"\n✓ Early stop at epoch {epoch+1}: "
                          f"reward plateaued at {recent:.3f}")
                    break
            else:
                patience_count = 0

    return {
        'rewards':     epoch_rewards,
        'wq':          epoch_wq,
        'skr':         epoch_skr,
        'pool':        epoch_pool,
        'policy_loss': policy_losses,
        'value_loss':  value_losses,
    }


# ── Monkey-patch ──────────────────────────────────────────────────────────────
defence_training.train_ppo = _batched_train_ppo
print("✓ Batched-GNN PPO patch applied.")
print("  GNN forward: 1 call per minibatch (was 64 serial calls).")
print("  To revert: restart kernel and skip this cell.")
