import argparse
import hashlib
import tempfile
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import pyvista as pv

from data_utils_cylindrical import (
    COORDINATE_SYSTEM,
    CYLINDER_RADIUS_OVER_RD,
    SA_RADIUS_OVER_RD,
    CYLINDER_RADII_OVER_RD,
    load_case_input,
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
RESOLUTION_Z_THETA = (256, 256)    # (Nz, Ntheta), 각 값은 16의 배수 권장
FAIL_ON_INVALID_POINTS = True       # 원통 일부가 CFD 도메인 밖이면 즉시 중단


def _folder_name(value):
    if isinstance(value, (int, float, np.number)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _make_cylinder_points(center_x, center_y, disk_radius_m, ground_z_m, z_max_m, nz, ntheta, radius_over_rd=CYLINDER_RADIUS_OVER_RD):
    """지면부터 메시 최대 z까지 증가하는 r=2R_D 원통 좌표를 만든다."""
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False, dtype=np.float64)
    if not np.isfinite([ground_z_m, z_max_m, disk_radius_m]).all() or z_max_m <= ground_z_m or disk_radius_m <= 0:
        raise ValueError("Require finite ground_z < mesh z_max and R_D > 0")
    z_m = np.linspace(ground_z_m, z_max_m, nz, dtype=np.float64)
    z_over_rd = z_m / disk_radius_m
    theta_grid, z_grid = np.meshgrid(theta, z_m, indexing="xy")

    cylinder_radius_m = radius_over_rd * disk_radius_m
    x = center_x + cylinder_radius_m * np.cos(theta_grid)
    y = center_y + cylinder_radius_m * np.sin(theta_grid)
    points = np.column_stack((x.ravel(), y.ravel(), z_grid.ravel()))
    return points, theta, z_m, z_over_rd, cylinder_radius_m


def _extract_surface(
    mesh,
    radius_over_rd,
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

    z_min_m, z_max_m = map(float, mesh.bounds[4:6])
    if not z_min_m <= ground_z_m < z_max_m:
        raise ValueError(f"ground_z={ground_z_m} must lie in mesh z bounds [{z_min_m}, {z_max_m})")
    nz, ntheta = map(int, RESOLUTION_Z_THETA)
    out = Path(output_root)
    height_over_rd = abs(rotor_z_m - ground_z_m) / disk_radius_m
    source_folder = str(source_folder if source_folder is not None else case_path.name).strip().replace("\\", "/")
    identity = repr((source_folder, rotor_spacing_m, rotor_diameter_m, rotor_z_m, ground_z_m, z_max_m, disk_loading, center_x, center_y))
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    case_id = make_case_id(l_over_d, disk_radius_m, height_over_rd) + "_ZRD_" + suffix
    input_path = out / "inputs" / f"input_{case_id}.npz"
    if input_path.exists():
        meta = load_case_input(input_path)
        if not np.isclose(meta["radius_over_rd"], radius_over_rd):
            raise ValueError("FATO and SA must use separate output roots")
        for component in ("u", "v", "w"):
            target = out / f"targets_{component}" / f"{component}_{case_id}.npy"
            if not target.is_file() or np.load(target, mmap_mode="r").shape != (nz, ntheta):
                raise RuntimeError(f"Incomplete existing case; preserved without overwriting: {target}")
        print(f"  [재사용] {case_id} | r={radius_over_rd:g}RD")
        return case_id
    for path in (out / "inputs").glob("input_*.npz"):
        if load_case_input(path)["source_folder"] == source_folder:
            raise RuntimeError(f"Existing source case has different metadata; preserved: {path}")
    points, theta, z_m, z_over_rd, cylinder_radius_m = _make_cylinder_points(
        center_x, center_y, disk_radius_m, ground_z_m, z_max_m, nz, ntheta, radius_over_rd
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
        f"r={radius_over_rd:.1f}R_D={cylinder_radius_m:.4f} m | "
        f"ground_z={ground_z_m:.4f} m -> mesh z_max={z_max_m:.4f} m | "
        f"D={rotor_diameter_m:.4f} m | DL={disk_loading:.4f} N/m^2 (total thrust 30000 N)"
    )

    return case_id


def _read_velocity_mesh(case_path):
    """Read U once; adapt flattened Drive exports only in a temporary case."""
    case_path = Path(case_path)
    numeric = []
    for path in case_path.iterdir():
        try:
            time = float(path.name)
        except ValueError:
            continue
        if path.is_dir() and ((path / "U").is_file() or (path / "U.gz").is_file()):
            numeric.append((time, path))
    candidates = (case_path / "constant/polyMesh", case_path / "polyMesh",
                  case_path / "constant/constant/polyMesh")
    mesh_root = next((p for p in candidates if (p / "points").is_file()), None)
    if mesh_root is None:
        # Also permits the existing mocked reader tests.
        reader = pv.OpenFOAMReader(str(case_path / f"{case_path.name}.foam"))
        reader.cell_to_point_creation = True
        if not reader.time_values:
            raise RuntimeError(f"time directory를 찾을 수 없습니다: {case_path}")
        reader.set_active_time_value(max(reader.time_values))
        blocks = reader.read()
    else:
        work = Path("/content/multirotor_runs")
        temporary_root = work if work.is_dir() else None
        with tempfile.TemporaryDirectory(prefix="foam_", dir=temporary_root) as folder:
            root = Path(folder)
            (root / "constant").mkdir()
            (root / "constant/polyMesh").symlink_to(mesh_root.resolve(), target_is_directory=True)
            if numeric:
                for _, path in numeric:
                    (root / path.name).symlink_to(path.resolve(), target_is_directory=True)
            else:
                velocity = next((p for p in (case_path / "constant/U", case_path / "U",
                                case_path / "constant/constant/U",
                                case_path / "constant/U.gz", case_path / "U.gz",
                                case_path / "constant/constant/U.gz") if p.is_file()), None)
                if velocity is None:
                    raise RuntimeError(f"OpenFOAM U field is missing: {case_path}")
                (root / "0").mkdir()
                (root / "0" / velocity.name).symlink_to(velocity.resolve())
            foam = root / "case.foam"
            foam.touch()
            reader = pv.OpenFOAMReader(str(foam))
            reader.cell_to_point_creation = True
            reader.skip_zero_time = False
            if not reader.time_values:
                raise RuntimeError(f"time directory를 찾을 수 없습니다: {case_path}")
            reader.set_active_time_value(max(reader.time_values))
            blocks = reader.read()
    mesh = blocks["internalMesh"] if "internalMesh" in blocks.keys() else blocks[0]
    return mesh.cell_data_to_point_data()


def extract_velocity_case(case_path, output_root, disk_radius_m, l_over_d, disk_loading,
                          rotor_z_m, ground_z_m, center_x=0.0, center_y=0.0,
                          source_folder=None, radii_over_rd=None, sa_output_root=None):
    radii = list(CYLINDER_RADII_OVER_RD if radii_over_rd is None else radii_over_rd)
    if not radii or len(set(radii)) != len(radii) or any(r not in CYLINDER_RADII_OVER_RD for r in radii):
        raise ValueError("Choose unique radii from the FATO/SA radius list")
    fato_root = Path(output_root)
    sa_root = Path(sa_output_root) if sa_output_root is not None else fato_root.with_name("dataset_zrd_sa")
    if fato_root.resolve() == sa_root.resolve():
        raise ValueError("FATO and SA output roots must differ")
    mesh = _read_velocity_mesh(case_path)
    return [_extract_surface(mesh, ratio, Path(case_path),
                fato_root if ratio == CYLINDER_RADIUS_OVER_RD else sa_root,
                disk_radius_m, l_over_d, disk_loading, rotor_z_m, ground_z_m,
                center_x, center_y, source_folder) for ratio in radii]


def main():
    parser = argparse.ArgumentParser(description="Extract and nondimensionalize OpenFOAM velocity fields")
    parser.add_argument("--csv", type=Path, default=Path("cases.csv"))
    parser.add_argument("--data-root", type=Path, help="Root for relative CSV folder values (default: CSV directory)")
    parser.add_argument("--output-root", type=Path, default=Path("dataset_zrd"))
    parser.add_argument("--sa-output-root", type=Path, help="Separate SA dataset root")
    parser.add_argument("--radii-over-rd", nargs="+", type=float, choices=CYLINDER_RADII_OVER_RD, default=CYLINDER_RADII_OVER_RD)
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
    folders = dataframe['folder'].fillna('').str.strip().str.replace('\\', '/', regex=False)
    if (folders == '').any() or folders.duplicated().any():
        raise ValueError('CSV folder contains empty or duplicate cases')
    print(f'CSV 케이스: {len(dataframe)}개 | {csv_path}', flush=True)
    failures = []
    for number, (_, row) in enumerate(dataframe.iterrows(), start=1):
        print(f'[{number}/{len(dataframe)}] 추출 시작: {row["folder"]}', flush=True)
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
                radii_over_rd=args.radii_over_rd,
                sa_output_root=args.sa_output_root,
            )
        except Exception as exc:
            print(f"  [에러] {row.get('folder')}: {exc}")
            failures.append(str(row.get("folder")))
    if failures:
        raise RuntimeError(f"Extraction failed for {len(failures)} case(s): {', '.join(failures)}")
    print(f'전체 추출 완료: {len(dataframe)}/{len(dataframe)}개', flush=True)


if __name__ == "__main__":
    main()
