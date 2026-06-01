# physical_layer/fiber_channel.py

class FiberLink(QKDLinkBase):
    """
    Optical fiber QKD link.

    Loss: exponential Beer-Lambert attenuation.
    Noise: detector dark counts + Raman scattering
           from co-propagating classical channels.
    Dynamics: essentially static — slow thermal drift only.
    Protocol: decoy-state BB84 assumed.
    """

    def __init__(self, node_i, node_j, length_km,
                 alpha_db_per_km=0.2,        # Standard SMF-28 fiber
                 eta_r=0.9,                  # Receiver coupling efficiency
                 raman_coefficient=1e-7,     # Raman noise coefficient
                 classical_power_dbm=0,      # Co-propagating classical channel power
                 temperature_k=293,          # Operating temperature
                 **kwargs):

        super().__init__(node_i, node_j, length_km, **kwargs)
        self.alpha = alpha_db_per_km
        self.eta_r = eta_r
        self.raman_coeff = raman_coefficient
        self.classical_power = 10**(classical_power_dbm/10) * 1e-3  # Watts
        self.temperature = temperature_k

    def link_type(self):
        return 'fiber'

    def transmittance(self, conditions: dict) -> float:
        """
        eta = 10^(-alpha * L/10) * eta_r * eta_d
        Conditions can include temperature-induced drift.
        """
        # Temperature causes slow refractive index drift
        temp_factor = 1.0 - 1e-5 * abs(
            conditions.get('temperature_k', self.temperature)
            - self.temperature
        )

        eta_fiber = 10**(-self.alpha * self.L/1e3 / 10)
        return eta_fiber * self.eta_r * self.eta_d * temp_factor

    def noise_rate(self, conditions: dict) -> float:
        """
        Fiber noise sources:
        1. Dark counts (detector)
        2. Raman scattering from co-propagating classical light
           R_raman = raman_coeff * P_classical * L * delta_lambda
        """
        # Raman noise — most significant in WDM deployments
        raman_noise = (self.raman_coeff *
                       self.classical_power *
                       self.L/1e3 *
                       conditions.get('classical_channels', 1))

        return self.dark_count + raman_noise

    def sample_conditions(self, time: float) -> dict:
        """
        Fiber conditions change slowly.
        Temperature follows a gentle diurnal cycle.
        """
        import numpy as np
        hour = (time / 3600) % 24
        temp = self.temperature + 5 * np.sin(2 * np.pi * hour / 24)
        return {
            'temperature_k':    temp,
            'classical_channels': 1,    # number of co-propagating channels
        }
