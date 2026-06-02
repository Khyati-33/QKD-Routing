import gymnasium as gym
from network_layer.topology import HybridQKDNetwork

class QKDRoutingEnv(gym.Env):
    def __init__(self, config, dt=300):   # dt=300s = 5-min timestep
        self.network = HybridQKDNetwork(config)
        self.dt = dt
        self.t = 0.0
        # Observation: concatenated get_rl_state() from all links
        # Action: integer index into ROUTES list

    def step(self, action):
        route = ROUTES[action]
        old_state = self.network.get_network_state(self.t)
        self.network.update_key_pools(self.t, self.dt, traffic={route: 1000})
        self.t += self.dt
        new_state = self.network.get_network_state(self.t)
        obs = self._flatten_state(new_state)
        reward = self.reward_fn(old_state, new_state, action)
        done = self.t >= 86400   # one full day
        return obs, reward, done, False, {}

    def reset(self, seed=None):
        self.t = 0.0
        self.network.reset_pools()
        return self._flatten_state(self.network.get_network_state(0)), {}
