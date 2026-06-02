import numpy as np

class FSOLink(QKDLinkBase):
    def __init__(self, node_i, node_j, length_km,
                 wavelength_nm=785,
                 beam_waist_mm=80,
                 receiver_radius_mm=200,
                 eta_g=0.85,
                 eta_r=0.5,
                 delta_lambda_nm=0.1,
                 tau_ns=0.5,
                 a_E=0.2,
                 FOV=100e-6,
                 pointing_error_rad=1e-6,
                 rng=None,
                 **kwargs):

        super().__init__(node_i, node_j, length_km, **kwargs)
        self.lam = wavelength_nm * 1e-9
        self.W0 = beam_waist_mm * 1e-3
        self.R = receiver_radius_mm * 1e-3
        self.eta_g = eta_g
        self.eta_r = eta_r
        self.d_lam = delta_lambda_nm * 1e-9
        self.tau = tau_ns * 1e-9
        self.a_E = a_E
        self.FOV = FOV
        self.pt_err = pointing_error_rad
        
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self.H_sun = self._calc_solar_radiance()

    def _calc_solar_radiance(self) -> float:
        # Planck's Law for 5778K blackbody solar radiance in W/(m^2 * nm * sr)
        h, c, kb, T = 6.626e-34, 3e8, 1.38e-23, 5778
        expr = (h * c) / (self.lam * kb * T)
        if expr > 700:
            return 0.0
        return ((2 * h * c**2) / (self.lam**5 * (np.exp(expr) - 1))) * 1e-9

    def link_type(self):
        return 'fso'

    def transmittance(self, conditions: dict) -> float:
        Cn2 = conditions.get('Cn2', 1e-14)
        eta_atm = conditions.get('eta_atm', 0.9)
        jitter = conditions.get('pointing_error', self.pt_err)

        # Geometric diffraction
        z_R = np.pi * self.W0**2 / self.lam
        W_L = self.W0 * np.sqrt(1 + (self.L / z_R)**2)
        eta_diff = 1 - np.exp(-2 * self.R**2 / W_L**2)

        # Gamma-Gamma turbulence fading
        k = 2 * np.pi / self.lam
        sigma_R2 = 1.23 * Cn2 * k**(7/6) * self.L**(11/6)
        eta_turb = self._sample_fading(sigma_R2) if sigma_R2 > 0 else 1.0

        # Pointing error
        div = self.lam / (np.pi * self.W0)
        eta_point = np.exp(-jitter**2 / div**2)

        return eta_diff * eta_atm * eta_turb * eta_point * self.eta_g * self.eta_r * self.eta_d

    def _sample_fading(self, sigma_R2: float) -> float:
        # Andrews & Phillips aperture averaging model
        k, Dr = 2 * np.pi / self.lam, 2 * self.R
        A = 1 / (1 + 1.062 * ((Dr**2 * k) / (4 * self.L))**1.2)

        a_scint = 0.49 * sigma_R2 / (1 + 1.11 * (sigma_R2**0.857))**1.17
        b_scint = A * (0.51 * sigma_R2 / (1 + 0.69 * (sigma_R2**0.857))**0.833)

        alpha = max(1.0 / (np.exp(a_scint) - 1.0), 1e-4)
        beta = max(1.0 / (np.exp(b_scint) - 1.0), 1e-4)

        return float(self.rng.gamma(alpha, 1.0/alpha) * self.rng.gamma(beta, 1.0/beta))

    def noise_rate(self, conditions: dict) -> float:
        sun_elev = conditions.get('sun_elevation', 1.0)
        E_photon = (6.626e-34 * 3e8) / self.lam
        
        omega = np.pi * (self.FOV / 2)**2
        P_bg = self.H_sun * sun_elev * self.a_E * (np.pi * self.R**2) * omega * (self.d_lam * 1e9)
        R_bg = (P_bg * self.eta_d * self.eta_r * self.tau) / E_photon

        return self.dark_count + R_bg

    def sample_conditions(self, time: float) -> dict:
        hour = (time / 3600) % 24

        Cn2_mean = max(1e-14 * np.exp(-((hour - 13)**2) / 18), 1e-17)
        Cn2 = float(self.rng.lognormal(np.log(Cn2_mean), 0.5))
        eta_atm = float(np.clip(self.rng.normal(0.9, 0.02), 0.1, 1.0))
        sun_elev = max(0, np.sin(np.pi * (hour - 6) / 12))
        
        jitter = float(abs(self.rng.normal(self.pt_err, self.pt_err * 0.1)))

        return {
            'Cn2': Cn2,
            'eta_atm': eta_atm,
            'sun_elevation': sun_elev,
            'time_of_day': hour,
            'pointing_error': jitter,
            'sigma_R2': 1.23 * Cn2 * (2*np.pi/self.lam)**(7/6) * self.L**(11/6),
        }

    def get_rl_state(self, conditions: dict, residual_keys: float) -> dict:
        base_state = super().get_rl_state(conditions, residual_keys)
        base_state.update({
            'sigma_R2_norm': float(np.clip(conditions.get('sigma_R2', 0) / 5, 0, 1)),
            'sun_elevation': float(conditions.get('sun_elevation', 0)),
            'eta_atm': float(conditions.get('eta_atm', 1.0)),
            'pointing_error': float(np.clip(conditions.get('pointing_error', 0) / 1e-5, 0, 1)),
        })
        return base_state
