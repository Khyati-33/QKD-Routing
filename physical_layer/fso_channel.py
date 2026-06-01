# physical_layer/fso_channel.py

class FSOLink(QKDLinkBase):
    """
    Free-space optical QKD link.

    Loss: diffraction + turbulence + atmospheric absorption.
    Noise: solar/urban background + dark counts.
    Dynamics: highly variable — diurnal, weather, pointing errors.
    Protocol: BB84 or entanglement-based (configurable).
    """

    def __init__(self, node_i, node_j, length_km,
                 wavelength_nm=785,
                 beam_waist_mm=80,
                 receiver_radius_mm=200,
                 eta_g=0.85,
                 eta_r=0.5,
                 delta_lambda_nm=0.1,
                 tau_ns=0.5,
                 H_sun=1.5e3,
                 a_E=0.2,
                 FOV=100e-6,
                 pointing_error_rad=1e-6,    # RMS pointing jitter
                 **kwargs):

        super().__init__(node_i, node_j, length_km, **kwargs)
        self.lambda_m = wavelength_nm * 1e-9
        self.W0 = beam_waist_mm * 1e-3
        self.R = receiver_radius_mm * 1e-3
        self.eta_g = eta_g
        self.eta_r = eta_r
        self.delta_lambda = delta_lambda_nm * 1e-9
        self.tau = tau_ns * 1e-9
        self.H_sun = H_sun
        self.a_E = a_E
        self.FOV = FOV
        self.pointing_error = pointing_error_rad

    def link_type(self):
        return 'fso'

    def transmittance(self, conditions: dict) -> float:
        """
        eta = eta_diff * eta_atm * eta_turb * eta_point * eta_g * eta_r * eta_d
        """
        Cn2     = conditions.get('Cn2', 1e-14)
        eta_atm = conditions.get('eta_atm', 0.9)

        # 1. Diffraction loss — Gaussian beam spreading
        z_R   = np.pi * self.W0**2 / self.lambda_m
        W_L   = self.W0 * np.sqrt(1 + (self.L / z_R)**2)
        eta_diff = 1 - np.exp(-2 * self.R**2 / W_L**2)

        # 2. Turbulence — Rytov variance based Strehl ratio
        k = 2 * np.pi / self.lambda_m
        sigma_R2 = 1.23 * Cn2 * k**(7/6) * self.L**(11/6)
        eta_turb = np.exp(-sigma_R2 / 2)

        # 3. Pointing error loss
        # Gaussian approximation: eta_point = exp(-theta^2 / theta_div^2)
        theta_div = self.lambda_m / (np.pi * self.W0)
        eta_point = np.exp(
            -self.pointing_error**2 / theta_div**2
        )

        return (eta_diff * eta_atm * eta_turb *
                eta_point * self.eta_g * self.eta_r * self.eta_d)

    def noise_rate(self, conditions: dict) -> float:
        """
        FSO noise: solar background + dark counts.
        Background dominates during daytime.
        """
        time_of_day = conditions.get('time_of_day', 12)
        sun_elevation = conditions.get('sun_elevation', 1.0)  # 0=night, 1=noon

        h = 6.626e-34
        c = 3e8
        E_photon = h * c / self.lambda_m
        A_r = np.pi * self.R**2

        R_bg = (self.H_sun * sun_elevation * self.a_E *
                self.eta_d * self.eta_r *
                A_r * self.delta_lambda *
                self.FOV**2 * self.tau / E_photon)

        return self.dark_count + R_bg

    def sample_conditions(self, time: float) -> dict:
        """
        FSO conditions vary significantly with time.
        """
        import numpy as np
        rng = np.random.default_rng(int(time * 1000) % 2**32)

        hour = (time / 3600) % 24

        # Diurnal Cn2 — peaks at midday
        Cn2_mean = 1e-14 * np.exp(-((hour - 13)**2) / 18)
        Cn2_mean = max(Cn2_mean, 1e-17)
        Cn2 = float(rng.lognormal(np.log(Cn2_mean), 0.5))

        # Atmospheric transmission — weather dependent
        eta_atm = float(np.clip(rng.normal(0.9, 0.02), 0.1, 1.0))

        # Sun elevation angle
        sun_elev = max(0, np.sin(np.pi * (hour - 6) / 12))

        # Pointing error jitter (wind, thermal expansion)
        point_jitter = float(abs(rng.normal(
            self.pointing_error,
            self.pointing_error * 0.1
        )))

        return {
            'Cn2':            Cn2,
            'eta_atm':        eta_atm,
            'sun_elevation':  sun_elev,
            'time_of_day':    hour,
            'pointing_error': point_jitter,
            'sigma_R2':       1.23 * Cn2 * (2*np.pi/self.lambda_m)**(7/6) * self.L**(11/6),
        }

    def get_rl_state(self, conditions: dict, residual_keys: float) -> dict:
        """
        FSO gets extra state variables the fiber links don't have.
        The routing agent can use these to anticipate FSO degradation.
        """
        base_state = super().get_rl_state(conditions, residual_keys)

        # FSO-specific additions
        base_state.update({
            'sigma_R2_norm':  float(np.clip(
                                conditions.get('sigma_R2', 0) / 5, 0, 1)),
            'sun_elevation':  float(conditions.get('sun_elevation', 0)),
            'eta_atm':        float(conditions.get('eta_atm', 1.0)),
            'pointing_error': float(np.clip(
                                conditions.get('pointing_error', 0) / 1e-5, 0, 1)),
        })
        return base_state
