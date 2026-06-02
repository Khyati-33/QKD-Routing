# device_profiles.py

from dataclasses import dataclass

@dataclass
class DetectorProfile:
    name: str
    # Efficiency
    eta_d: float          # detection efficiency
    # Noise
    dark_count_hz: float  # dark count rate [Hz]
    dead_time_ns: float   # detector dead time [ns]
    timing_jitter_ps: float  # timing jitter [ps]
    # Spectral
    wavelength_nm: float  # operating wavelength
    # Type flag for RL state
    detector_type: str    # 'SNSPD' or 'APD'


@dataclass
class QKDSourceProfile:
    name: str
    pulse_rate_MHz: float
    mean_photon_number: float   # mu — for decoy state
    protocol: str               # 'BB84_decoy', 'CV_QKD', 'E91'
    visibility: float           # interference visibility
    modulation_error: float     # extinction ratio imperfection


# Concrete device presets 

# ID Quantique Clavis3 — APD based, telecom wavelength
IDQ_CLAVIS3_DETECTOR = DetectorProfile(
    name='IDQ_Clavis3_InGaAs_APD',
    eta_d=0.25,
    dark_count_hz=100,
    dead_time_ns=10000,     # 10 µs gating dead time
    timing_jitter_ps=400,
    wavelength_nm=1310,
    detector_type='APD'
)

# Toshiba QKD — SNSPD, 1550 nm
TOSHIBA_SNSPD = DetectorProfile(
    name='Toshiba_SNSPD',
    eta_d=0.85,
    dark_count_hz=10,
    dead_time_ns=5,
    timing_jitter_ps=50,
    wavelength_nm=1550,
    detector_type='SNSPD'
)

# Generic Si-APD for 785 nm FSO (lower cost, short range)
SI_APD_785 = DetectorProfile(
    name='Generic_Si_APD_785nm',
    eta_d=0.60,
    dark_count_hz=500,
    dead_time_ns=50,
    timing_jitter_ps=500,
    wavelength_nm=785,
    detector_type='APD'
)

# QuintessenceLabs CV-QKD source
QLABS_CV_SOURCE = QKDSourceProfile(
    name='QuintessenceLabs_CV',
    pulse_rate_MHz=2500,
    mean_photon_number=10.0,   # coherent state, high mu
    protocol='CV_QKD',
    visibility=0.99,
    modulation_error=0.002
)

# Standard decoy-state BB84 source
BB84_DECOY_SOURCE = QKDSourceProfile(
    name='Standard_BB84_Decoy',
    pulse_rate_MHz=100,
    mean_photon_number=0.5,
    protocol='BB84_decoy',
    visibility=0.98,
    modulation_error=0.005
)
