# physical_layer/base_channel.py

from abc import ABC, abstractmethod
import numpy as np

class QKDLinkBase(ABC):
    """
    Abstract base class for all QKD link types.
    The routing layer only ever talks to this interface —
    it never needs to know if a link is fiber or FSO.
    """

    def __init__(self, node_i, node_j, length_km,
                 eta_d=0.6,
                 dark_count=1e-6,
                 visibility=0.98,
                 pulse_rate_MHz=100,
                 f_ec=1.16,
                 epsilon_sec=1e-10,
                 n_block=1e6):

        self.node_i = node_i
        self.node_j = node_j
        self.L = length_km * 1e3       # meters
        self.eta_d = eta_d
        self.dark_count = dark_count
        self.visibility = visibility
        self.pulse_rate = pulse_rate_MHz * 1e6
        self.f_ec = f_ec
        self.epsilon_sec = epsilon_sec
        self.n_block = n_block

    # --- Interface every link type must implement ---

    @abstractmethod
    def transmittance(self, conditions: dict) -> float:
        """
        End-to-end channel transmittance given current
        environmental conditions. Conditions dict is
        link-type specific (Cn2 for FSO, temperature for fiber).
        """
        pass

    @abstractmethod
    def noise_rate(self, conditions: dict) -> float:
        """
        Total noise photon rate [per pulse].
        Sources differ between FSO and fiber.
        """
        pass

    @abstractmethod
    def link_type(self) -> str:
        pass

    @abstractmethod
    def sample_conditions(self, time: float) -> dict:
        """
        Sample current environmental conditions at time t.
        FSO: returns Cn2, eta_atm, pointing error.
        Fiber: returns temperature, Raman noise level.
        """
        pass

    # --- Shared computation for all link types ---

    @staticmethod
    def binary_entropy(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -p * np.log2(p) - (1-p) * np.log2(1-p)

    def QBER(self, conditions: dict) -> float:
        """
        Total QBER from first principles.
        Optical + noise + memory decoherence terms.
        Shared formula, but transmittance and noise_rate
        are link-type specific.
        """
        eta = self.transmittance(conditions)
        d   = self.noise_rate(conditions)

        e_opt   = (1 - self.visibility) / 2
        e_noise = d / (eta + d + 1e-12)
        return np.clip(e_opt + e_noise, 0, 0.5)

    def SKR_asymptotic(self, conditions: dict) -> float:
        """
        Asymptotic secret key rate [bits/pulse].
        Shared formula across link types.
        """
        eta = self.transmittance(conditions)
        e   = self.QBER(conditions)

        if e >= 0.11:
            return 0.0

        r = eta * max(0, 1 - (1 + self.f_ec) * self.binary_entropy(e))
        return r

    def SKR_finite(self, conditions: dict) -> float:
        """
        Finite-key SKR with composable security correction.
        """
        eta = self.transmittance(conditions)
        e   = self.QBER(conditions)

        if e >= 0.11:
            return 0.0

        delta = 7 * np.sqrt(np.log2(2.0 / self.epsilon_sec))
        correction = delta / np.sqrt(self.n_block)

        r = max(0.0,
                eta * (1 - (1 + self.f_ec) * self.binary_entropy(e))
                - correction)
        return r * self.pulse_rate   # bits/second

    def get_rl_state(self, conditions: dict, residual_keys: float) -> dict:
        """
        Physics-grounded state vector for RL agent.
        Same structure regardless of link type —
        routing layer gets a uniform observation.
        """
        eta = self.transmittance(conditions)
        e   = self.QBER(conditions)
        SKR = self.SKR_finite(conditions)

        return {
            'link_type':         0 if self.link_type() == 'fiber' else 1,
            'transmittance':     float(eta),
            'QBER':              float(e),
            'SKR_norm':          float(np.clip(SKR / 1e6, 0, 1)),
            'qber_margin':       float(np.clip((0.11 - e) / 0.11, 0, 1)),
            'residual_keys_norm':float(np.clip(residual_keys / 1e6, 0, 1)),
            'is_secure':         float(e < 0.11),
        }
