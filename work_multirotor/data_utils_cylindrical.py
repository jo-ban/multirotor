from pathlib import Path

import numpy as np


COORDINATE_SYSTEM = "absolute_z_over_rd_v1"
INPUT_CHANNELS = ("L/D", "sin(theta)", "cos(theta)", "z/R_D")
TARGET_SCALE = 100.0
DEFAULT_SHAPE_ZTHETA = (256, 256)
FIXED_RD_M = 9.3101751


def geometry_from_rd_ratio(rd_m, l_over_d):
    """Keep RD fixed; infer L and D from q=L/D for a square quadrotor."""
    rd_m, l_over_d = float(rd_m), float(l_over_d)
    if not np.isfinite([rd_m, l_over_d]).all() or rd_m <= 0 or l_over_d <= 0:
        raise ValueError("RD and L/D must be finite and positive")
    diameter = rd_m / (l_over_d / np.sqrt(2.0) + 0.5)
    return l_over_d * diameter, diameter


def _geometry_value(row, names, default=None):
    values = [float(row[name]) for name in names
              if name in row and str(row[name]).strip().lower() not in ("", "nan", "<na>", "none")]
    if not values:
        return default
    if not np.isfinite(values).all() or not np.allclose(values, values[0], rtol=1e-6, atol=1e-8):
        raise ValueError(f"Invalid or conflicting geometry columns: {names}")
    return values[0]


def geometry_from_row(row):
    """Preferred CSV: RD,L_over_D (also R_D and L/D); RD may be omitted."""
    rd = _geometry_value(row, ("RD", "R_D", "rd_m", "disk_radius_m"), FIXED_RD_M)
    ratio = _geometry_value(row, ("L_over_D", "L/D", "l_over_d"))
    old_l = _geometry_value(row, ("L", "spacing", "rotor_spacing", "rotor_spacing_m"))
    old_d = _geometry_value(row, ("D", "diameter", "rotor_diameter", "rotor_diameter_m"))
    if ratio is None:
        if old_l is None or old_d is None or old_l <= 0 or old_d <= 0:
            raise ValueError("CSV requires L_over_D (or L/D); RD defaults to 9.3101751 m")
        ratio = old_l / old_d
    spacing, diameter = geometry_from_rd_ratio(rd, ratio)
    for supplied, inferred in ((old_l, spacing), (old_d, diameter)):
        if supplied is not None and not np.isclose(supplied, inferred, rtol=1e-6, atol=1e-7):
            raise ValueError("L/D columns conflict with RD and L_over_D; use the RD,L_over_D input format")
    return rd, ratio, spacing, diameter


def cartesian_to_cylindrical_velocity(u, v, w, theta_rad):
    """Convert Cartesian velocity components to cylindrical components about z."""
    u = np.asarray(u)
    v = np.asarray(v)
    w = np.asarray(w)
    theta_rad = np.asarray(theta_rad)
    if u.shape != v.shape or u.shape != w.shape:
        raise ValueError("u, v and w must have identical shapes")
    if theta_rad.ndim != 1 or u.ndim < 1 or u.shape[-1] != theta_rad.size:
        raise ValueError("theta_rad must be one-dimensional and match the last velocity axis")
    angle_shape = (1,) * (u.ndim - 1) + (theta_rad.size,)
    cos_theta = np.cos(theta_rad).reshape(angle_shape)
    sin_theta = np.sin(theta_rad).reshape(angle_shape)
    return {
        "u_r": u * cos_theta + v * sin_theta,
        "u_theta": -u * sin_theta + v * cos_theta,
        "u_z": w,
    }


def _first_numeric(row, keys, default=None):
    for key in keys:
        if key in row.index and not np.isnan(row[key]):
            return float(row[key])
    if default is None:
        raise KeyError(f"다음 열 중 하나가 필요합니다: {keys}")
    return float(default)


def rotor_spacing_from_row(row):
    """인접한 개별 로터 중심 사이 거리 L [m]를 읽는다."""
    return geometry_from_row(row)[2]


def rotor_diameter_from_row(row):
    """개별 로터 한 개의 직경 D [m]를 읽는다."""
    return geometry_from_row(row)[3]


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
    return geometry_from_row(row)[1]


def disk_radius_from_row(row):
    """정사각형으로 배치된 4개 로터 전체의 외접원 반경 R_D [m]를 계산한다."""
    return geometry_from_row(row)[0]


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


def make_input_surface(l_over_d, z_over_rd, shape_ztheta):
    """원통 전개면용 2D U-Net 입력을 만든다.

    CH0: L/D (케이스별 스칼라 값)
    CH1: sin(theta)
    CH2: cos(theta)
    CH3: z/R_D (OpenFOAM 실제 z 좌표 / 외접반경)

    반환 shape: (4, Nz, Ntheta). 행은 ground_z부터 mesh z_max까지 증가한다.
    """
    nz, ntheta = map(int, shape_ztheta)
    if nz < 16 or ntheta < 16 or nz % 16 or ntheta % 16:
        raise ValueError("Nz와 Ntheta는 4단계 U-Net을 위해 16의 배수여야 합니다.")

    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False, dtype=np.float32)
    z_over_rd = np.asarray(z_over_rd, dtype=np.float32)
    if z_over_rd.shape != (nz,) or not np.isfinite(z_over_rd).all() or not np.all(np.diff(z_over_rd) > 0):
        raise ValueError("z/R_D must be a finite, strictly increasing vector of length Nz")

    image = np.empty((4, nz, ntheta), dtype=np.float32)
    image[0].fill(float(l_over_d))
    image[1] = np.broadcast_to(np.sin(theta)[None, :], (nz, ntheta))
    image[2] = np.broadcast_to(np.cos(theta)[None, :], (nz, ntheta))
    image[3] = np.broadcast_to(z_over_rd[:, None], (nz, ntheta))
    return image


def load_case_input(input_path):
    with np.load(input_path) as archive:
        data = {key: archive[key] for key in archive.files}
    if str(data.get("coordinate_system", "")) != COORDINATE_SYSTEM:
        raise ValueError(f"{input_path}: re-extract ground-to-mesh-top z/R_D data into a new directory")
    return {
        "z_m": np.asarray(data["z_m"], dtype=np.float32),
        "z_over_rd": np.asarray(data["z_over_rd"], dtype=np.float32),
        "z_max_m": float(data["z_max_m"]),
        "source_folder": str(data["source_folder"]),
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


def case_ids_for_rows(data_root, rows):
    """Resolve validation identifiers only; physical inputs come from extracted NPZs."""
    records = [(case_id, load_case_input(Path(data_root) / "inputs" / f"input_{case_id}.npz"))
               for case_id in find_case_ids(data_root)]
    selected = []
    for _, row in rows.iterrows():
        explicit_id = str(row.get("case_id", "")).strip()
        folder = str(row.get("folder", "")).strip().replace("\\", "/")
        if explicit_id.lower() in ("", "nan", "<na>"):
            explicit_id = ""
        matches = [case_id for case_id, meta in records
                   if (case_id == explicit_id if explicit_id else meta["source_folder"] == folder)]
        if len(matches) != 1:
            raise ValueError(f"Expected one extracted case for {explicit_id or folder}, found {len(matches)}. Specify case_id for ambiguous folders.")
        selected.extend(matches)
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate CSV cases; training and validation must not share a case")
    return selected


def validation_ids_from_frame(data_root, dataframe):
    if "type" not in dataframe:
        return []
    labels = dataframe["type"].astype(str).str.strip().str.upper()
    # A repeated identifier must not silently appear in both train and validation.
    if "folder" in dataframe:
        keys = dataframe["folder"].astype(str).str.strip().str.replace("\\", "/", regex=False)
        if "case_id" in dataframe:
            ids = dataframe["case_id"].fillna("").astype(str).str.strip()
            keys = ids.where(ids != "", keys)
        if keys.duplicated().any():
            raise ValueError("Duplicate CSV identifiers; use unique folder or case_id values")
    return case_ids_for_rows(data_root, dataframe[labels == "V"])


def validate_checkpoint(checkpoint, expected_rd=None):
    if (checkpoint.get("coordinate_system") != COORDINATE_SYSTEM
            or tuple(checkpoint.get("input_channels", ())) != INPUT_CHANNELS
            or checkpoint.get("in_channels") != 4
            or checkpoint["state_dict"]["enc1.block.0.conv.weight"].shape[1] != 4):
        raise ValueError("A newly trained 4-channel z/R_D checkpoint is required; old s/R_D and 5-channel weights are incompatible")
    if expected_rd is not None and "disk_radius_m" in checkpoint:
        if not np.isclose(float(checkpoint["disk_radius_m"]), expected_rd, rtol=1e-6, atol=1e-7):
            raise ValueError("Prediction RD differs from the fixed RD used for training")


def plot_extent(theta_deg, z_m):
    """imshow bounds at half-grid edges so pixel centers match the sample coordinates."""
    dtheta = float(theta_deg[1] - theta_deg[0])
    dz = float(z_m[1] - z_m[0])
    return (float(theta_deg[0]) - dtheta / 2, float(theta_deg[-1]) + dtheta / 2,
            float(z_m[0]) - dz / 2, float(z_m[-1]) + dz / 2)
