"""
rl_agent/reward.py
==================
STN-constrained, path-aware physics-driven reward function for the QKD routing agent.

Design principles
-----------------
1. STN Physics Compliance: The hard security constraint is evaluated on the 
   accumulated chain noise w(q), not individual link QBERs. Crossing the 11% 
   asymptotic security limit triggers an early-exit return.
2. Path-Level Evaluation: Quantifies end-to-end performance using the global 
   STN secret key block size rather than a simple bottleneck link heuristic.
3. Use-case Modularity: Adjusts performance weights (security margin vs throughput) 
   dynamically per deployment mode (defence, commercial, research).
4. Continuous Gradients: Implements a soft warning zone for the accumulated noise 
   w(q) to provide feedback to the RL agent before terminal invalidation occurs.
"""

from __future__ import annotations
import numpy as np
from math import comb
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal

# ── Physical constants ────────────────────────────────────────────────────────
W_Q_SECURITY_LIMIT: float = 0.11   # Asymptotic limit for total accumulated chain noise
W_Q_WARNING_MARGIN: float = 0.08   # Operational warning threshold for total chain noise
WARNING_ZONE_WIDTH: float = W_Q_SECURITY_LIMIT - W_Q_WARNING_MARGIN  # 0.03

# ── Weight presets per use-case ───────────────────────────────────────────────
USE_CASE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "defence": {
        "w_skr":             0.5,
        "w_pool":            0.6,
        "w_security_margin": 1.0,   # Prioritize staying far below the STN noise limit
        "w_noise_penalty":   0.8,   # High penalty for selecting paths that rapidly degrade w(q)
        "w_switch":          0.2,
        "security_penalty":  40.0,
        "depletion_penalty": 60.0,
        "warning_penalty":   10.0,
    },
    "commercial": {
        "w_skr":             1.4,   # Focus heavily on raw secret key generation capacity
        "w_pool":            0.4,
        "w_security_margin": 0.2,
        "w_noise_penalty":   0.3,
        "w_switch":          0.4,   # Minimize route disruptions to satisfy SLAs
        "security_penalty":  25.0,
        "depletion_penalty": 50.0,
        "warning_penalty":   5.0,
    },
    "research": {
        "w_skr":             1.0,
        "w_pool":            0.5,
        "w_security_margin": 0.5,
        "w_noise_penalty":   0.5,
        "w_switch":          0.3,
        "security_penalty":  25.0,
        "depletion_penalty": 50.0,
        "warning_penalty":   5.0,
    },
}

@dataclass
class RewardComponents:
    """Per-term breakdown of one STN reward computation for training diagnostic tracking."""
    security_hard:    float = 0.0  # Non-zero only when total chain noise w(q) >= 0.11
    security_soft:    float = 0.0  # Soft gradient tracking inside warning zone [0.08, 0.11)
    security_margin:  float = 0.0  # Reward for operating safely below the security limit
    skr:              float = 0.0  # Normalized end-to-end STN secret key generation performance
    pool:             float = 0.0  # Normalized minimum pool ratio across path
    depletion:        float = 0.0  # Key asset exhaustion penalty
    noise_accumulation: float = 0.0 # Physics-driven path degradation penalty
    switch:           float = 0.0  # Routing transition cost
    total:            float = 0.0  # Aggregated scalar reward value

class SecurityConstrainedReward:
    """
    Multi-objective, path-aware reward for the STN routing MDP.
    """
    def __init__(self,
                 use_case:           Literal['defence', 'commercial', 'research'] = 'research',
                 max_skr_bps:        float = 16594.0,
                 pool_capacity_bits: float = 1e6,
                 **weight_overrides):

        if use_case not in USE_CASE_WEIGHTS:
            raise ValueError(f"use_case must be one of {list(USE_CASE_WEIGHTS)}")

        self.use_case      = use_case
        self.max_skr       = max_skr_bps
        self.pool_capacity = pool_capacity_bits

        w = dict(USE_CASE_WEIGHTS[use_case])
        w.update(weight_overrides)

        self.w_skr             = w['w_skr']
        self.w_pool            = w['w_pool']
        self.w_security_margin = w['w_security_margin']
        self.w_noise_penalty   = w['w_noise_penalty']
        self.w_switch          = w['w_switch']
        self.security_penalty  = w['security_penalty']
        self.depletion_penalty = w['depletion_penalty']
        self.warning_penalty   = w['warning_penalty']

    def _path_links(self, path: List[str]) -> List[Tuple[str, str]]:
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    def compute_total_chain_noise(self, Q_avg: float, p: int) -> float:
        """
        Computes the expected accumulated total noise w(q) for a chain of p STNs
        assuming average depolarizing noise per link Q_avg.
        
        An error occurs if and only if an odd number of link-level errors occur 
        across the p + 1 links.
        """
        n_links = p + 1
        w = 0.0
        for i in range((n_links // 2) + 1):
            k = 2 * i + 1
            if k <= n_links:
                w += comb(n_links, k) * (Q_avg ** k) * ((1.0 - Q_avg) ** (n_links - k))
        return w

    def _compute(self,
                 path:        List[str],
                 link_states: Dict,
                 old_pools:   Dict[Tuple[str, str], float],
                 new_pools:   Dict[Tuple[str, str], float],
                 switched:    bool) -> RewardComponents:
        
        c = RewardComponents()
        links = self._path_links(path)
        p = len(path) - 2  # Number of STN relay nodes (excluding Alice and Bob)

        if not links:
            return c

        # ── 1. Calculate Average Link QBER and Total Accumulated Noise ────────
        link_qbers = [link_states[lk]['QBER'] for lk in links if lk in link_states]
        Q_avg = np.mean(link_qbers) if link_qbers else 0.0
        w_q = self.compute_total_chain_noise(Q_avg, p)

        # ── 2. Hard Security Constraint Check ─────────────────────────────────
        if w_q >= W_Q_SECURITY_LIMIT:
            c.security_hard = -self.security_penalty
            c.total = c.security_hard
            return c

        # ── 3. Soft Warning Gradient ──────────────────────────────────────────
        if w_q > W_Q_WARNING_MARGIN:
            penetration = (w_q - W_Q_WARNING_MARGIN) / WARNING_ZONE_WIDTH
            c.security_soft = -2.0 * penetration

        # ── 4. Security Margin Bonus ──────────────────────────────────────────
        margin = (W_Q_SECURITY_LIMIT - w_q) / W_Q_SECURITY_LIMIT
        c.security_margin = self.w_security_margin * margin

        # ── 5. STN Path-Level Secret Key Rate Evaluation ──────────────────────
        # Realized key rate is limited by the physical bottleneck link capacity 
        # discounted by the global end-to-end STN extraction cost.
        raw_bottleneck = min(link_states[lk]['SKR'] for lk in links if lk in link_states)
        
        # Binary entropy function h(x)
        w_q_clipped = np.clip(w_q, 1e-9, 0.5 - 1e-9)
        h_wq = -w_q_clipped * np.log2(w_q_clipped) - (1.0 - w_q_clipped) * np.log2(1.0 - w_q_clipped)
        stn_efficiency_factor = max(0.0, 1.0 - 2.0 * h_wq)
        
        realized_stn_skr = raw_bottleneck * stn_efficiency_factor
        c.skr = self.w_skr * np.clip(realized_stn_skr / self.max_skr, 0.0, 1.0)

        # ── 6. Key Pool Asset Tracking ────────────────────────────────────────
        path_pools = [new_pools[lk] for lk in links if lk in new_pools]
        min_pool_val = min(path_pools) if path_pools else 0.0
        min_pool_ratio = min_pool_val / self.pool_capacity
        c.pool = self.w_pool * np.clip(min_pool_ratio, 0.0, 1.0)

        # ── 7. Pool Depletion Penalties ───────────────────────────────────────
        any_depleted = any(new_pools[lk] <= 0 for lk in links if lk in new_pools)
        any_critical = any(new_pools[lk] < 0.1 * self.pool_capacity for lk in links if lk in new_pools)

        if any_depleted:
            c.depletion = -self.depletion_penalty
        elif any_critical:
            c.depletion = -self.warning_penalty * (1.0 - (min_pool_ratio / 0.1))

        # ── 8. Physics-Driven Noise Accumulation Penalty ──────────────────────
        # Replaces empirical hop penalties with a structural degradation penalty 
        # proportional to how much noise the chain introduces compared to a direct link.
        w_q_direct = self.compute_total_chain_noise(Q_avg, 0)
        added_noise = max(0.0, w_q - w_q_direct)
        c.noise_accumulation = -self.w_noise_penalty * (added_noise / W_Q_SECURITY_LIMIT)

        # ── 9. Route Switching Cost ───────────────────────────────────────────
        if switched and w_q < W_Q_WARNING_MARGIN:
            c.switch = -self.w_switch

        # ── Complete Aggregation ──────────────────────────────────────────────
        c.total = (c.security_hard + c.security_soft + c.security_margin +
                   c.skr + c.pool + c.depletion + c.noise_accumulation + c.switch)
        return c

    def __call__(self, path: List[str], link_states: Dict, old_pools: Dict, new_pools: Dict, switched: bool) -> float:
        return self._compute(path, link_states, old_pools, new_pools, switched).total

    def decompose(self, path: List[str], link_states: Dict, old_pools: Dict, new_pools: Dict, switched: bool) -> RewardComponents:
        return self._compute(path, link_states, old_pools, new_pools, switched)
