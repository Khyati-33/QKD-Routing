"""
rl_agent/environment.py
=======================
STN-Aware, gym-compliant Markov Decision Process Environment for QKD Routing.
"""

from __future__ import annotations
import gym
from gym import spaces
import numpy as np
from typing import Dict, List, Tuple, Any
from rl_agent.reward import SecurityConstrainedReward, W_Q_SECURITY_LIMIT

class STNQKDRoutingEnv(gym.Env):
    """
    Custom Gym environment representing a multi-hop Simplified Trusted Node (STN) 
    network topology.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 topology_config: Dict[str, Any],
                 channel_layer: Any,
                 use_case: str = 'research',
                 pool_capacity_bits: float = 1e6,
                 max_hops: int = 10,
                 terminate_on_depletion: bool = True):
        super().__init__()
        
        self.config = topology_config
        self.channels = channel_layer
        self.pool_capacity = pool_capacity_bits
        self.max_hops = max_hops
        self.terminate_on_depletion = terminate_on_depletion

        # Extract infrastructure details
        self.nodes: List[str] = sorted(list(self.config['nodes'].keys()))
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.links: List[Tuple[str, str]] = self.config['links']
        
        # System endpoints
        self.alice = self.config['endpoints']['alice']
        self.bob = self.config['endpoints']['bob']

        # Core State Tracking Storage
        self.current_time_step = 0
        self.current_path: List[str] = [self.alice, self.bob]
        self.key_pools: Dict[Tuple[str, str], float] = {tuple(lk): pool_capacity_bits for lk in self.links}
        
        # Instantiate Reward Calculator
        self.reward_engine = SecurityConstrainedReward(
            use_case=use_case,
            max_skr_bps=16594.0,
            pool_capacity_bits=pool_capacity_bits
        )

        # ── Define Observation Space ──────────────────────────────────────────
        # Feature count per link = 14 (11 base channel dimensions + 3 global STN metrics)
        self.features_per_link = 14
        num_links = len(self.links)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_links * self.features_per_link + len(self.nodes),),
            dtype=np.float32
        )

        # ── Define Action Space ───────────────────────────────────────────────
        # Action space mapped as choosing a valid neighbor or an index from an explicit path table
        self.action_space = spaces.Discrete(len(self.nodes))

    def _get_network_state(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Queries physical channel layer for current QBER and raw SKR markers."""
        state_dict = {}
        for lk in self.links:
            link_tuple = tuple(lk)
            # Simulated link conditions based on step execution time
            state_dict[link_tuple] = self.channels.get_link_metrics(link_tuple, self.current_time_step)
        return state_dict

    def _build_observation_vector(self, link_states: Dict) -> np.ndarray:
        """
        Serializes multi-hop physical metrics combined with global STN features 
        into a structural 1D observation vector.
        """
        obs_buffer = []
        p_chain = max(0, len(self.current_path) - 2)

        # Calculate path baseline error features for uniform conditioning
        path_links = [(self.current_path[i], self.current_path[i+1]) for i in range(len(self.current_path)-1)]
        path_qbers = [link_states[lk]['QBER'] for lk in path_links if lk in link_states]
        Q_avg_path = np.mean(path_qbers) if path_qbers else 0.0

        for lk in self.links:
            link_tuple = tuple(lk)
            metrics = link_states[link_tuple]
            
            qber = metrics['QBER']
            skr = metrics['SKR']
            pool_ratio = self.key_pools[link_tuple] / self.pool_capacity

            # 1-11: Channel features
            obs_buffer.extend([
                float(metrics.get('link_type_id', 0.0)),
                float(metrics.get('transmittance', 1.0)),
                float(qber),
                float(skr / 16594.0),
                float((0.11 - qber) / 0.11),
                float(pool_ratio),
                1.0 if qber < 0.11 else 0.0,
                float(metrics.get('sigma_R2', 0.0)),
                float(metrics.get('sun_elevation', 0.0)),
                float(metrics.get('eta_atm', 1.0)),
                float(metrics.get('pointing_error', 0.0))
            ])

            # 12-14: STN features
            # Predict global impact if this specific link is added to the active chain
            hypothetical_p = p_chain + 1
            w_q_hypothetical = self.reward_engine.compute_total_chain_noise(
                np.mean([Q_avg_path, qber]), hypothetical_p
            )
            
            obs_buffer.extend([
                float(w_q_hypothetical / W_Q_SECURITY_LIMIT),             # 12: Normalized total chain noise
                float(max(0.0, W_Q_SECURITY_LIMIT - w_q_hypothetical)),  # 13: Remaining noise budget
                float(hypothetical_p / self.max_hops)                    # 14: Path sizing growth
            ])

        # Topographic contextual encoding (Active Path Nodes)
        node_occupancy = np.zeros(len(self.nodes), dtype=np.float32)
        for node in self.current_path:
            if node in self.node_to_idx:
                node_occupancy[self.node_to_idx[node]] = 1.0
                
        return np.concatenate([np.array(obs_buffer, dtype=np.float32), node_occupancy])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
            
        self.current_time_step = 0
        self.current_path = [self.alice, self.bob]
        self.key_pools = {tuple(lk): self.pool_capacity for lk in self.links}
        
        link_states = self._get_network_state()
        return self._build_observation_vector(link_states)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes a step in the environment by evaluating the routing path 
        against the global STN noise bounds.
        """
        self.current_time_step += 1
        target_node = self.nodes[action]

        # ── Path Construction Logic ──────────────────────────────────────────
        old_path = list(self.current_path)
        
        # Simple dynamic modifier: reconstruct path from choice
        if target_node not in [self.alice, self.bob]:
            self.current_path = [self.alice, target_node, self.bob]
        else:
            self.current_path = [self.alice, self.bob]

        switched = (self.current_path != old_path)
        link_states = self._get_network_state()
        
        # Save historical snapshots for consumption differential calculations
        old_pools = dict(self.key_pools)
        
        # ── Key Pool Generation & Consumption Math ────────────────────────────
        active_links = [(self.current_path[i], self.current_path[i+1]) for i in range(len(self.current_path)-1)]
        
        # 1. Generation Phase (All links generate keys independently)
        for lk in self.links:
            link_tuple = tuple(lk)
            gen_keys = link_states[link_tuple]['SKR'] * 1.0  # Assumes 1-second step interval
            self.key_pools[link_tuple] = min(self.pool_capacity, self.key_pools[link_tuple] + gen_keys)

        # 2. Consumption Phase (Active path consumes keys uniformly to route downstream data)
        consumption_load = 500.0  # Constant demand matching simulation profiling
        for lk in active_links:
            if lk in self.key_pools:
                self.key_pools[lk] = max(0.0, self.key_pools[lk] - consumption_load)
            elif (lk[1], lk[0]) in self.key_pools:
                rev_lk = (lk[1], lk[0])
                self.key_pools[rev_lk] = max(0.0, self.key_pools[rev_lk] - consumption_load)

        # ── Evaluate Global Path Security Status ──────────────────────────────
        link_qbers = [link_states[lk]['QBER'] for lk in active_links if lk in link_states]
        Q_avg = np.mean(link_qbers) if link_qbers else 0.0
        p_chain = max(0, len(self.current_path) - 2)
        
        w_q_total = self.reward_engine.compute_total_chain_noise(Q_avg, p_chain)
        stn_cryptographically_secure = (w_q_total < W_Q_SECURITY_LIMIT)

        # Check for key pool exhaustion
        pool_depleted = any(v <= 0 for v in self.key_pools.values())

        # ── Termination Criteria ──────────────────────────────────────────────
        terminated = (not stn_cryptographically_secure) or (pool_depleted and self.terminate_on_depletion)
        truncated = (self.current_time_step >= 200)  # Time horizon truncation boundary

        # Reward Processing
        reward = self.reward_engine(
            path=self.current_path,
            link_states=link_states,
            old_pools=old_pools,
            new_pools=self.key_pools,
            switched=switched
        )

        # Diagnostic metadata logging
        decomposition = self.reward_engine.decompose(
            path=self.current_path, link_states=link_states,
            old_pools=old_pools, new_pools=self.key_pools, switched=switched
        )
        
        info = {
            "step": self.current_time_step,
            "total_chain_noise": w_q_total,
            "is_secure": stn_cryptographically_secure,
            "decomposition": decomposition.__dict__
        }

        next_obs = self._build_observation_vector(link_states)
        return next_obs, float(reward), (terminated or truncated), info

    def render(self, mode="human"):
        print(f"Step: {self.current_time_step} | Path: {self.current_path} | Pools: {[self.key_pools[tuple(k)] for k in self.links]}")
