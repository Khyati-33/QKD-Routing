import numpy as np

class FiberLink(QKDLinkBase):
    def __init__(self, node_i, node_j, length_km,
                 alpha_db_per_km=0.2,
                 eta_r=0.9,
                 raman_coefficient=1e-7,
                 classical_power_dbm=0,
                 temperature_k=293,
                 pmd_coef=0.1, # ps/sqrt(km)
                 rng=None,
                 **kwargs):

        super().__init__(node_i, node_j, length_km, **kwargs)
        self.alpha_db = alpha_db_per_km
        self.alpha_lin = 10**(alpha_db_per_km / 10) # linear loss per km
        self.alpha_np = alpha_db_per_km * 0.1151   # absorption coefficient in Np/km
        self.eta_r = eta_r
        self.raman_coeff = raman_coefficient
        self.classical_power = 10**(classical_power_dbm/10) * 1e-3
        self.temperature = temperature_k
        self.pmd_coef = pmd_coef
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    def link_type(self):
        return 'fiber'

    def transmittance(self, conditions: dict) -> float:
        tk = conditions.get('temperature_k', self.temperature)
        temp_factor = 1.0 - 1e-5 * abs(tk - self.temperature)

        # Bug fix: self.L used directly as km instead of dividing by 1e3
        eta_fiber = 10**(-self.alpha_db * self.L / 10)
        
        # Polarization alignment efficiency penalty from PMD
        pmd_delay = self.pmd_coef * np.sqrt(self.L)
        eta_pol = np.cos(conditions.get('pol_drift', 0.0))**2

        return eta_fiber * self.eta_r * self.eta_d * temp_factor * eta_pol

    def noise_rate(self, conditions: dict) -> float:
        ch_count = conditions.get('classical_channels', 1)
        
        # Bug fix: use effective length Leff due to classical pump attenuation
        L_eff = (1 - np.exp(-self.alpha_np * self.L)) / self.alpha_np
        
        raman_noise = self.raman_coeff * self.classical_power * L_eff * ch_count
        return self.dark_count + raman_noise

    def sample_conditions(self, time: float) -> dict:
        hour = (time / 3600) % 24
        
        # Deterministic cycle + stochastic temperature jitter
        t_base = self.temperature + 5 * np.sin(2 * np.pi * hour / 24)
        t_jitter = t_base + float(self.rng.normal(0, 0.2))
        
        # Polarization state rotation drift driven by thermal changes
        pol_drift = float(self.rng.normal(0.0, 0.05 * (t_jitter - self.temperature)))

        return {
            'temperature_k': t_jitter,
            'classical_channels': 1,
            'pol_drift': pol_drift
        }

    def get_rl_state(self, conditions: dict, residual_keys: float) -> dict:
        base_state = super().get_rl_state(conditions, residual_keys)
        base_state.update({
            'temperature_k': float(conditions.get('temperature_k', self.temperature)),
            'pol_drift': float(conditions.get('pol_drift', 0.0))
        })
        return base_state
