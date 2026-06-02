# network_layer/topology.py

from physical_layer.fiber_channel import FiberLink
from physical_layer.fso_channel import FSOLink

class HybridQKDNetwork:
    """
    Mixed fiber + FSO trusted relay network.
    Manages per-link physical state and
    exposes a uniform interface to the routing layer.
    """

    def __init__(self, config):
        self.nodes = config.nodes
        self.links = self._build_links(config.link_configs)
        
        # Track active operational states to prevent double sampling inconsistencies
        self.current_conditions = {}
        
        # Explicitly support bidirectional lookups for key pools
        self.key_pools = {}
        for (i, j) in self.links:
            self.key_pools[(i, j)] = config.initial_key_pool
            self.key_pools[(j, i)] = config.initial_key_pool
            
        self.max_pool_capacity = getattr(config, 'max_pool_capacity', 1e6)

    def _build_links(self, link_configs):
        links = {}
        for cfg in link_configs:
            i, j = cfg['nodes']
            # Pass shared RNG down if provided in configuration
            link_rng = cfg.get('params', {}).get('rng', None)
            
            if cfg['type'] == 'fiber':
                links[(i, j)] = FiberLink(i, j, cfg['length_km'], **cfg['params'])
            elif cfg['type'] == 'fso':
                links[(i, j)] = FSOLink(i, j, cfg['length_km'], **cfg['params'])
                
        return links

    def get_network_state(self, time: float) -> dict:
        """
        Sample all link states at current time.
        Caches condition states to ensure execution consistency across layers.
        """
        states = {}
        for (i, j), link in self.links.items():
            conditions = link.sample_conditions(time)
            
            # Cache conditions to eliminate double-sampling mismatch
            self.current_conditions[(i, j)] = conditions
            
            states[(i, j)] = link.get_rl_state(
                conditions,
                self.key_pools[(i, j)]
            )
            states[(i, j)]['conditions'] = conditions
            states[(i, j)]['link_type_str'] = link.link_type()
        return states

    def update_key_pools(self, time: float, dt: float, traffic: dict):
        """
        Update all key pools using cached execution states.
        """
        for (i, j), link in self.links.items():
            # Fix: Retrieve cached conditions instead of re-sampling randomly
            conditions = self.current_conditions.get((i, j))
            if conditions is None:
                conditions = link.sample_conditions(time)
                
            # Safely calculate Secure Key Rate (SKR)
            SKR = link.SKR_finite(conditions) if hasattr(link, 'SKR_finite') else 0.0

            # Symmetric bidirectional key generation update
            new_keys = SKR * dt
            for direction in [(i, j), (j, i)]:
                # Restrict bounds to maximum storage capacities
                capacity = getattr(link, 'n_block', self.max_pool_capacity)
                self.key_pools[direction] = min(self.key_pools[direction] + new_keys, capacity)

                # Deduct keys consumed by directional routed traffic
                consumed = traffic.get(direction, 0)
                self.key_pools[direction] = max(0, self.key_pools[direction] - consumed)

    def get_link_type_mask(self):
        """
        Returns which links are FSO vs fiber.
        """
        mask = {}
        for (i, j), link in self.links.items():
            mask[(i, j)] = link.link_type()
            mask[(j, i)] = link.link_type()
        return mask
