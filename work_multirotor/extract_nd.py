import argparse
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import pyvista as pv

from data_utils_cylindrical import (
    center_from_row,
    disk_loading_from_row,
    make_case_id,
    rotor_diameter_from_row,
    rotor_ground_z_from_row,
    rotor_spacing_from_row,
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


def _make_cylinder_points(center_x, center_y, disk_radius_m, rotor_z_m, ground_z_m, nz, ntheta):
    """로터면부터 지면까지 r=2R_D 원통 옆면의 점을 만든다."""
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False, dtype=np.float64)
    z_m = np.linspace(rotor_z_m, ground_z_m, nz, dtype=np.float64)
    height_m = abs(rotor_z_m - ground_z_m)
    s_over_rd = np.linspace(0.0, height_m / disk_radius_m, nz, dtype=np.float64)
    theta_grid, z_grid = np.meshgrid(theta, z_m, indexing="xy")

    cylinder_radius_m = CYLINDER_RADIUS_OVER_RD * disk_radius_m
    x = center_x + cylinder_radius_m * np.cos(theta_grid)
    y = center_y + cylinder_radius_m * np.sin(theta_grid)
    points = np.column_stack((x.ravel(), y.ravel(), z_grid.ravel()))
    return points, theta, z_m, s_over_rd, cylinder_radius_m


def extract_velocity_case(
    case_path,
    output_root,
    rotor_spacing_m,
    rotor_diameter_m,
    disk_loading,
    rotor_z_m,
    ground_z_m,
    center_x=0.0,
    center_y=0.0,
):
    l_over_d = rotor_spacing_m / rotor_diameter_m
    disk_radius_m = rotor_spacing_m / np.sqrt(2.0) + rotor_diameter_m / 2.0
    if disk_radius_m <= 0:
        raise ValueError("멀티로터 외접원 반경 R_D는 0보다 커야 합니다.")

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

    nz, ntheta = map(int, RESOLUTION_Z_THETA)
    points, theta, z_m, s_over_rd, cylinder_radius_m = _make_cylinder_points(
        center_x, center_y, disk_radius_m, rotor_z_m, ground_z_m, nz, ntheta
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
    case_id = make_case_id(l_over_d, disk_radius_m, height_over_rd)
    np.savez_compressed(
        out / "inputs" / f"input_{case_id}.npz",
        l_over_d=np.float32(l_over_d),
        rotor_spacing_m=np.float32(rotor_spacing_m),
        rotor_diameter_m=np.float32(rotor_diameter_m),
        disk_radius_m=np.float32(disk_radius_m),
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
        s_over_rd=s_over_rd.astype(np.float32),
    )
    for index, component in enumerate(("u", "v", "w")):
        np.save(out / f"targets_{component}" / f"{component}_{case_id}.npy", velocity_nd[..., index])

    print(
        f"  [완료] {case_id} | surface(z,theta)={velocity_nd.shape[:2]} | "
        f"r={CYLINDER_RADIUS_OVER_RD:.1f}R_D={cylinder_radius_m:.4f} m | "
        f"rotor_z={rotor_z_m:.4f} m -> ground_z={ground_z_m:.4f} m"
    )


def main():
    parser = argparse.ArgumentParser(description="Extract and nondimensionalize OpenFOAM velocity fields")
    parser.add_argument("--csv", type=Path, default=Path("cases.csv"))
    parser.add_argument("--data-root", type=Path, help="Root for relative CSV folder values (default: CSV directory)")
    parser.add_argument("--output-root", type=Path, default=Path("dataset_nd"))
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
            extract_velocity_case(
                case_path=case_path,
                output_root=args.output_root,
                rotor_spacing_m=rotor_spacing_from_row(row),
                rotor_diameter_m=rotor_diameter_from_row(row),
                disk_loading=disk_loading_from_row(row),
                rotor_z_m=rotor_z,
                ground_z_m=ground_z,
                center_x=center_x,
                center_y=center_y,
            )
        except Exception as exc:
            print(f"  [에러] {row.get('folder')}: {exc}")
            failures.append(str(row.get("folder")))
    if failures:
        raise RuntimeError(f"Extraction failed for {len(failures)} case(s): {', '.join(failures)}")


if __name__ == "__main__":
    main()
