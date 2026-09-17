import numpy as np

AIR_DENSITY = 1.225

def get_induced_velocity(disk_loading: float) -> float:
    """Vi = sqrt(DL / (2 * rho))"""
    if disk_loading <= 0:
        raise ValueError("disk loading은 0보다 커야 합니다.")
    return np.sqrt(disk_loading / (2.0 * AIR_DENSITY))

def nondim_velocity(velocity_mps: np.ndarray, disk_loading: float) -> np.ndarray:
    vi = get_induced_velocity(disk_loading)
    return np.asarray(velocity_mps) / vi

def dimensional_velocity(velocity_nd: np.ndarray, disk_loading: float) -> np.ndarray:
    vi = get_induced_velocity(disk_loading)
    return np.asarray(velocity_nd) * vi
