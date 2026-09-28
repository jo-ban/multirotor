import argparse
import hashlib
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import pyvista as pv

from data_utils_cylindrical import (
    COORDINATE_SYSTEM,
    center_from_row,
    disk_loading_from_row,
    disk_loading_from_diameter,
    make_case_id,
    geometry_from_row,
    geometry_from_rd_ratio,
    rotor_ground_z_from_row,
)
from normalization import nondim_velocity


# 원통 표면 설정 -------------------------------------------------------------
CYLINDER_RADIUS_OVER_RD = 2.0      # 원통 반경 / 4로터 전체 외접원 반경 R_D
RESOLUTION_Z_THETA = (256, 256)    # (Nz, Ntheta), 각 값은 16의 배수 권장
FAIL_ON_INVALID_POINTS = True       # 원통 일부가 CFD 도메인 밖이면 즉시 중단


def _folder_name(value):
    if isinstance(value, (int, float, np.number)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _make_cylinder_points(center_x, center_y, disk_radius_m, ground_z_m, z_max_m, nz, ntheta):
    """지면부터 메시 최대 z까지 증가하는 r=2R_D 원통 좌표를 만든다."""
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False, dtype=np.float64)
    if not np.isfinite([ground_z_m, z_max_m, disk_radius_m]).all() or z_max_m <= ground_z_m or disk_radius_m <= 0:
        raise ValueError("Require finite ground_z < mesh z_max and R_D > 0")
    z_m = np.linspace(ground_z_m, z_max_m, nz, dtype=np.float64)
    z_over_rd = z_m / disk_radius_m
    theta_grid, z_grid = np.meshgrid(theta, z_m, indexing="xy")

    cylinder_radius_m = CYLINDER_RADIUS_OVER_RD * disk_radius_m
    x = center_x + cylinder_radius_m * np.cos(theta_grid)
    y = center_y + cylinder_radius_m * np.sin(theta_grid)
    points = np.column_stack((x.ravel(), y.ravel(), z_grid.ravel()))
    return points, theta, z_m, z_over_rd, cylinder_radius_m


def extract_velocity_case(
    case_path,
    output_root,
    disk_radius_m,
    l_over_d,
    disk_loading,
    rotor_z_m,
    ground_z_m,
    center_x=0.0,
    center_y=0.0,
    source_folder=None,
):
    rotor_spacing_m, rotor_diameter_m = geometry_from_rd_ratio(disk_radius_m, l_over_d)
    # Retain the legacy argument for callers, but always use fixed-total-thrust DL.
    disk_loading = disk_loading_from_diameter(rotor_diameter_m)

    foam_file = case_path / f"{case_path.name}.foam"
    if not foam_file.exists():
        foam_file.touch()

    reader = pv.OpenFOAMReader(str(foam_file))
    reader.cell_to_point_creation = True
    if not reader.time_values:
        raise RuntimeError(f"time directory를 찾을 수 없습니다: {case_path}")
    reader.set_active_time_value(max(reader.time_values))

    multi_block = reader.read()
    mesh = multi_block["internalMesh"] if "internalMesh" in multi_block.keys() else multi_block[0]
    mesh = mesh.cell_data_to_point_data()

    z_min_m, z_max_m = map(float, mesh.bounds[4:6])
    if not z_min_m <= ground_z_m < z_max_m:
        raise ValueError(f"ground_z={ground_z_m} must lie in mesh z bounds [{z_min_m}, {z_max_m})")
    nz, ntheta = map(int, RESOLUTION_Z_THETA)
    points, theta, z_m, z_over_rd, cylinder_radius_m = _make_cylinder_points(
        center_x, center_y, disk_radius_m, ground_z_m, z_max_m, nz, ntheta
    )
    sampled = pv.PolyData(points).sample(mesh)
    if "U" not in sampled.point_data:
        raise KeyError("OpenFOAM 결과에서 point field 'U'를 찾지 못했습니다.")

    velocity_raw = np.asarray(sampled.point_data["U"], dtype=np.float32)[:, :3]
    valid = np.asarray(
        sampled.point_data.get("vtkValidPointMask", np.ones(len(velocity_raw))), dtype=bool
    )
    invalid_count = int((~valid).sum())
    if invalid_count:
        message = (
            f"원통 표면의 {invalid_count}/{len(valid)}개 점이 CFD 도메인 밖입니다. "
            "중심, R_D, rotor_z, ground_z와 계산영역을 확인하세요."
        )
        if FAIL_ON_INVALID_POINTS:
            raise RuntimeError(message)
        print(f"  [경고] {message} 해당 값을 0으로 저장합니다.")
        velocity_raw[~valid] = 0.0

    velocity_zt = velocity_raw.reshape((nz, ntheta, 3), order="C")
    velocity_nd = nondim_velocity(velocity_zt, disk_loading).astype(np.float32)

    out = Path(output_root)
    for directory in ("inputs", "targets_u", "targets_v", "targets_w"):
        (out / directory).mkdir(parents=True, exist_ok=True)

    height_over_rd = abs(rotor_z_m - ground_z_m) / disk_radius_m
    source_folder = str(source_folder if source_folder is not None else case_path.name).strip().replace("\\", "/")
    identity = repr((source_folder, rotor_spacing_m, rotor_diameter_m, rotor_z_m, ground_z_m, z_max_m, disk_loading, center_x, center_y))
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    case_id = make_case_id(l_over_d, disk_radius_m, height_over_rd) + "_ZRD_" + suffix
    np.savez_compressed(
        out / "inputs" / f"input_{case_id}.npz",
        coordinate_system=COORDINATE_SYSTEM,
        source_folder=source_folder,
        z_max_m=np.float64(z_max_m),
        l_over_d=np.float32(l_over_d),
        rotor_spacing_m=np.float32(rotor_spacing_m),
        rotor_diameter_m=np.float32(rotor_diameter_m),
        disk_radius_m=np.float64(disk_radius_m),
        disk_loading=np.float32(disk_loading),
        cylinder_radius_m=np.float32(cylinder_radius_m),
        center_xy_m=np.asarray((center_x, center_y), dtype=np.float32),
        rotor_z_m=np.float32(rotor_z_m),
        ground_z_m=np.float32(ground_z_m),
        height_m=np.float32(abs(rotor_z_m - ground_z_m)),
        height_over_rd=np.float32(height_over_rd),
        shape_ztheta=np.asarray((nz, ntheta), dtype=np.int32),
        theta_rad=theta.astype(np.float32),
        z_m=z_m.astype(np.float32),
        z_over_rd=z_over_rd.astype(np.float32),
    )
    for index, component in enumerate(("u", "v", "w")):
        np.save(out / f"targets_{component}" / f"{component}_{case_id}.npy", velocity_nd[..., index])

    print(
        f"  [완료] {case_id} | surface(z,theta)={velocity_nd.shape[:2]} | "
        f"r={CYLINDER_RADIUS_OVER_RD:.1f}R_D={cylinder_radius_m:.4f} m | "
        f"ground_z={ground_z_m:.4f} m -> mesh z_max={z_max_m:.4f} m | "
        f"D={rotor_diameter_m:.4f} m | DL={disk_loading:.4f} N/m^2 (total thrust 30000 N)"
    )


def main():
    parser = argparse.ArgumentParser(description="Extract and nondimensionalize OpenFOAM velocity fields")
    parser.add_argument("--csv", type=Path, default=Path("cases.csv"))
    parser.add_argument("--data-root", type=Path, help="Root for relative CSV folder values (default: CSV directory)")
    parser.add_argument("--output-root", type=Path, default=Path("dataset_zrd"))
    args = parser.parse_args()
    csv_path = args.csv.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else csv_path.parent
    if not csv_path.exists():
        raise FileNotFoundError("cases.csv 파일이 없습니다.")

    dataframe = pd.read_csv(csv_path, dtype={"folder": str})
    if "folder" not in dataframe.columns:
        raise KeyError("cases.csv에 folder 열이 필요합니다.")

    if dataframe.empty:
        raise ValueError("CSV contains no cases")
    radii = [geometry_from_row(row)[0] for _, row in dataframe.iterrows()]
    if not np.allclose(radii, radii[0], rtol=1e-6, atol=1e-7):
        raise ValueError("All cases in this fixed-RD extraction must use the same RD")
    failures = []
    for _, row in dataframe.iterrows():
        try:
            folder = _folder_name(row["folder"]).strip()
            if not folder or pd.isna(row["folder"]):
                raise ValueError("CSV folder is empty")
            if PureWindowsPath(folder).drive and Path(folder).anchor == "":
                raise ValueError("Windows path cannot be used here; use a relative CSV folder path")
            case_path = data_root / Path(folder.replace("\\", "/"))
            if not case_path.is_dir():
                raise FileNotFoundError(f"Case directory does not exist: {case_path}")
            center_x, center_y = center_from_row(row)
            rotor_z, ground_z = rotor_ground_z_from_row(row)
            rd, ratio, _, _ = geometry_from_row(row)
            extract_velocity_case(
                case_path=case_path,
                output_root=args.output_root,
                disk_radius_m=rd,
                l_over_d=ratio,
                disk_loading=disk_loading_from_row(row),
                rotor_z_m=rotor_z,
                ground_z_m=ground_z,
                center_x=center_x,
                center_y=center_y,
                source_folder=folder,
            )
        except Exception as exc:
            print(f"  [에러] {row.get('folder')}: {exc}")
            failures.append(str(row.get("folder")))
    if failures:
        raise RuntimeError(f"Extraction failed for {len(failures)} case(s): {', '.join(failures)}")


if __name__ == "__main__":
    main()
