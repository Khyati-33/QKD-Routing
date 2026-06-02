# turbulence_regimes.py — sketch

@dataclass
class TurbulenceRegime:
    name: str                    # e.g. 'urban_rooftop_summer_noon'
    Cn2_mean: float              # structure constant [m^(-2/3)]
    Cn2_log_std: float           # log-normal spread
    wind_speed_ms: float         # for beam wander (Taylor's frozen turbulence)
    coherence_time_ms: float     # how fast fading changes (correlation time)
    inner_scale_mm: float        # l0 — inner turbulence scale
    outer_scale_m: float         # L0 — outer turbulence scale
    notes: str

# validated against literature Cn2 tables
URBAN_ROOFTOP_NIGHT  = TurbulenceRegime('urban_rooftop_night',   1e-17, 0.3, 2.0,  100, 1, 10, 'Clear, calm')
URBAN_ROOFTOP_DAWN   = TurbulenceRegime('urban_rooftop_dawn',    1e-15, 0.5, 3.0,  50,  1, 10, 'Thermal mixing starts')
URBAN_ROOFTOP_MIDDAY = TurbulenceRegime('urban_rooftop_midday',  1e-13, 0.8, 8.0,  10,  1, 10, 'Strong convection')
MARITIME_LINK        = TurbulenceRegime('maritime',              5e-15, 0.4, 6.0,  30,  2, 50, 'Sea spray, stable')
UAV_GROUND           = TurbulenceRegime('uav_ground',            1e-14, 0.6, 10.0, 20,  1, 5,  'Platform vibration')
