from pathlib import Path

import numpy as np


TARGET_SCALE = 100.0
DEFAULT_SHAPE_ZTHETA = (256, 256)


def _first_numeric(row, keys, default=None):
    for key in keys:
        if key in row.index and not np.isnan(row[key]):
            return float(row[key])
    if default is None:
        raise KeyError(f"다음 열 중 하나가 필요합니다: {keys}")
    return float(default)


def rotor_spacing_from_row(row):
    """인접한 개별 로터 중심 사이 거리 L [m]를 읽는다."""
    return _first_numeric(row, ("L", "spacing", "rotor_spacing", "rotor_spacing_m"))


def rotor_diameter_from_row(row):
    """개별 로터 한 개의 직경 D [m]를 읽는다."""
    return _first_numeric(row, ("D", "diameter", "rotor_diameter", "rotor_diameter_m"))


def outer_radius_square_quadrotor(rotor_spacing_m, rotor_diameter_m):
    """정사각형 4로터 전체를 감싸는 외접원 반경 R_D [m].

    L은 인접 로터 중심 간 거리, D는 개별 로터 직경이다.
    R_D = L/sqrt(2) + D/2
    """
    l_value = float(rotor_spacing_m)
    d_value = float(rotor_diameter_m)
    if l_value <= 0 or d_value <= 0:
        raise ValueError("L과 D는 0보다 커야 합니다.")
    return l_value / np.sqrt(2.0) + d_value / 2.0


def l_over_d_from_row(row):
    """L과 D로 간격비 L/D를 계산한다."""
    return rotor_spacing_from_row(row) / rotor_diameter_from_row(row)


def disk_radius_from_row(row):
    """정사각형으로 배치된 4개 로터 전체의 외접원 반경 R_D [m]를 계산한다."""
    return outer_radius_square_quadrotor(
        rotor_spacing_from_row(row), rotor_diameter_from_row(row)
    )


def disk_loading_from_row(row):
    # 속도 무차원화/복원에만 사용한다. 네트워크 입력변수에는 포함하지 않는다.
    return _first_numeric(row, ("load", "disk_loading", "DL"))


def center_from_row(row):
    x = _first_numeric(row, ("center_x", "x_center", "cx"), default=0.0)
    y = _first_numeric(row, ("center_y", "y_center", "cy"), default=0.0)
    return x, y


def rotor_ground_z_from_row(row):
    """로터면과 지면의 실제 OpenFOAM z 좌표 [m]를 읽는다."""
    rotor_z = _first_numeric(row, ("rotor_z", "rotor_z_m", "z_rotor"))
    ground_z = _first_numeric(row, ("ground_z", "ground_z_m", "z_ground"))
    if np.isclose(rotor_z, ground_z):
        raise ValueError("rotor_z와 ground_z는 서로 달라야 합니다.")
    return rotor_z, ground_z


def height_over_rd_from_row(row):
    rotor_z, ground_z = rotor_ground_z_from_row(row)
    return abs(rotor_z - ground_z) / disk_radius_from_row(row)


def _token(value):
    return f"{float(value):.4f}".replace("-", "m").replace(".", "p")


def make_case_id(l_over_d, disk_radius_m, height_over_rd):
    return (
        f"LoD{_token(l_over_d)}_RD{_token(disk_radius_m)}_"
        f"HoRD{_token(height_over_rd)}"
    )


def make_input_surface(l_over_d, height_over_rd, shape_ztheta):
    """원통 전개면용 2D U-Net 입력을 만든다.

    CH0: L/D (케이스별 스칼라 값)
    CH1: H/R_D (로터면-지면 거리/외접반경)
    CH2: sin(theta)
    CH3: cos(theta)
    CH4: s/R_D (로터면에서 지면 방향으로 잰 거리)

    반환 shape: (5, Nz, Ntheta)
    """
    nz, ntheta = map(int, shape_ztheta)
    if nz % 16 or ntheta % 16:
        raise ValueError("Nz와 Ntheta는 4단계 U-Net을 위해 16의 배수여야 합니다.")

    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False, dtype=np.float32)
    if height_over_rd <= 0:
        raise ValueError("H/R_D는 0보다 커야 합니다.")
    s_over_rd = np.linspace(0.0, height_over_rd, nz, dtype=np.float32)

    image = np.empty((5, nz, ntheta), dtype=np.float32)
    image[0].fill(float(l_over_d))
    image[1].fill(float(height_over_rd))
    image[2] = np.broadcast_to(np.sin(theta)[None, :], (nz, ntheta))
    image[3] = np.broadcast_to(np.cos(theta)[None, :], (nz, ntheta))
    image[4] = np.broadcast_to(s_over_rd[:, None], (nz, ntheta))
    return image


def load_case_input(input_path):
    data = np.load(input_path)
    return {
        "l_over_d": float(data["l_over_d"]),
        "rotor_spacing_m": float(data["rotor_spacing_m"]),
        "rotor_diameter_m": float(data["rotor_diameter_m"]),
        "disk_radius_m": float(data["disk_radius_m"]),
        "disk_loading": float(data["disk_loading"]),
        "cylinder_radius_m": float(data["cylinder_radius_m"]),
        "center_xy_m": tuple(float(x) for x in data["center_xy_m"]),
        "rotor_z_m": float(data["rotor_z_m"]),
        "ground_z_m": float(data["ground_z_m"]),
        "height_m": float(data["height_m"]),
        "height_over_rd": float(data["height_over_rd"]),
        "shape_ztheta": tuple(int(x) for x in data["shape_ztheta"]),
    }


def find_case_ids(data_root):
    root = Path(data_root)
    return sorted(
        p.name[len("input_"):-len(".npz")]
        for p in (root / "inputs").glob("input_*.npz")
    )
