import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_utils_cylindrical import (
    DEFAULT_SHAPE_ZTHETA,
    TARGET_SCALE,
    COORDINATE_SYSTEM,
    INPUT_CHANNELS,
    validate_checkpoint,
    plot_extent,
    cartesian_to_cylindrical_velocity,
    make_input_surface,
    FIXED_RD_M,
    geometry_from_rd_ratio,
)
from model_cylindrical import CylindricalUNet2D
from normalization import dimensional_velocity
from plot_hover import attach_velocity_hover


MODEL_PATHS = {c: Path(f"rotor_unet_cyl2d_{c}.pth") for c in ("u", "v", "w")}


def load_component_model(component, device, model_dir=None, expected_rd=None):
    path = MODEL_PATHS[component] if model_dir is None else Path(model_dir) / f"rotor_unet_cyl2d_{component}.pth"
    if not path.exists():
        raise FileNotFoundError(f"{path}가 없습니다. train_nd.py로 먼저 학습하세요.")
    checkpoint = torch.load(path, map_location=device)
    validate_checkpoint(checkpoint, expected_rd)
    model = CylindricalUNet2D(
        in_channels=4,
        out_channels=1,
        base_channels=int(checkpoint.get("base_channels", 16)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, float(checkpoint.get("target_scale", TARGET_SCALE))


def predict_cylindrical_surface(
    disk_radius_m,
    l_over_d,
    disk_loading,
    rotor_z_m,
    ground_z_m,
    z_max_m,
    shape_ztheta=DEFAULT_SHAPE_ZTHETA,
    save_path="prediction_cylinder_r2RD.npz",
    show_plot=True,
    model_dir=None,
    plot_path=None,
    backend="TkAgg",
):
    """고정 RD와 L/D로 형상을 계산하고 r=2RD 원통면의 U_r/U_theta/U_z를 예측한다."""
    if show_plot:
        try:
            plt.switch_backend(backend)
            probe = plt.figure()
            plt.close(probe)
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "Plot window unavailable. Install python3-tk and use a desktop/X11 session. "
                "See README_LINUX.md. For numerical output only, use --no-show."
            ) from exc
    else:
        plt.switch_backend("Agg")
    rotor_spacing_m, rotor_diameter_m = geometry_from_rd_ratio(disk_radius_m, l_over_d)
    height_m = abs(float(rotor_z_m) - float(ground_z_m))
    if height_m <= 0:
        raise ValueError("rotor_z_m과 ground_z_m은 서로 달라야 합니다.")
    height_over_rd = height_m / disk_radius_m
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not np.isfinite([ground_z_m, z_max_m]).all() or z_max_m <= ground_z_m:
        raise ValueError("Require finite --z-max > --ground-z")
    nz, ntheta = map(int, shape_ztheta)
    z_m = np.linspace(ground_z_m, z_max_m, nz, dtype=np.float32)
    z_over_rd = z_m / disk_radius_m
    input_surface = make_input_surface(l_over_d, z_over_rd, shape_ztheta)
    inputs = torch.from_numpy(input_surface[None, ...]).to(device)

    prediction_nd = {}
    with torch.no_grad():
        for component in ("u", "v", "w"):
            model, scale = load_component_model(component, device, model_dir, disk_radius_m)
            prediction_nd[component] = (
                model(inputs)[0, 0].cpu().numpy() / scale
            ).astype(np.float32)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    prediction = {
        c: dimensional_velocity(prediction_nd[c], disk_loading).astype(np.float32)
        for c in ("u", "v", "w")
    }
    magnitude = np.sqrt(
        prediction["u"] ** 2 + prediction["v"] ** 2 + prediction["w"] ** 2
    ).astype(np.float32)

    nz, ntheta = map(int, shape_ztheta)
    theta_rad = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False, dtype=np.float32)
    theta_deg = np.rad2deg(theta_rad).astype(np.float32)

    # 모델은 Cartesian (u, v, w)를 예측하므로, 축이 z인 원통좌표로 변환한다.
    cylindrical = {
        key: value.astype(np.float32)
        for key, value in cartesian_to_cylindrical_velocity(
            prediction["u"], prediction["v"], prediction["w"], theta_rad
        ).items()
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        u_mps=prediction["u"],
        v_mps=prediction["v"],
        w_mps=prediction["w"],
        u_r_mps=cylindrical["u_r"],
        u_theta_mps=cylindrical["u_theta"],
        u_z_mps=cylindrical["u_z"],
        radial_mps=cylindrical["u_r"],
        tangential_mps=cylindrical["u_theta"],
        magnitude_mps=magnitude,
        theta_deg=theta_deg,
        coordinate_system=COORDINATE_SYSTEM,
        velocity_coordinate_system="cylindrical_z_axis",
        input_channels=np.asarray(INPUT_CHANNELS),
        z_over_rd=z_over_rd,
        z_max_m=np.float64(z_max_m),
        z_m=z_m,
        rotor_z_m=np.float32(rotor_z_m),
        ground_z_m=np.float32(ground_z_m),
        height_m=np.float32(height_m),
        height_over_rd=np.float32(height_over_rd),
        rotor_spacing_m=np.float32(rotor_spacing_m),
        rotor_diameter_m=np.float32(rotor_diameter_m),
        l_over_d=np.float32(l_over_d),
        disk_radius_m=np.float64(disk_radius_m),
        cylinder_radius_m=np.float32(2.0 * disk_radius_m),
        disk_loading=np.float32(disk_loading),
    )
    print(f"r=2R_D 원통 표면 예측 결과 저장: {save_path}")

    if show_plot or plot_path is not None:
        extent = plot_extent(theta_deg, z_m)
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        images = []
        for ax, data, title in zip(
            axes.flat,
            (cylindrical["u_r"], cylindrical["u_theta"], cylindrical["u_z"], magnitude),
            (r"$U_r$ (radial)", r"$U_\theta$ (azimuthal)", r"$U_z$ (axial)", "Magnitude"),
        ):
            image = ax.imshow(data, origin="lower", extent=extent, aspect="auto", cmap="turbo")
            images.append(image)
            ax.set_title(title)
            ax.set_xlabel("azimuth theta [deg]")
            ax.set_ylabel("z [m]")
            ax.set_ylim(float(z_m[0]), float(z_m[-1]))
            fig.colorbar(image, ax=ax, label="m/s")
        plt.suptitle(f"Cylinder r=2R_D prediction | L/D={l_over_d:.3f}")
        plt.tight_layout()
        if plot_path is not None:
            Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=150)
        if show_plot:
            attach_velocity_hover(fig, images, {**cylindrical, "magnitude": magnitude},
                                  theta_deg, z_m)
            plt.show(block=True)
        plt.close(fig)

    return cylindrical["u_r"], cylindrical["u_theta"], cylindrical["u_z"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a cylindrical velocity field")
    parser.add_argument("--model-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("prediction_cylinder_r2RD.npz"))
    parser.add_argument("--rd", type=float, default=FIXED_RD_M, help="Fixed outer radius RD [m] (default: 9.3101751)")
    parser.add_argument("--l-over-d", type=float, default=1.5, help="Rotor spacing / rotor diameter")
    parser.add_argument("--disk-loading", type=float, required=True, help="Actual disk loading [N/m^2]")
    parser.add_argument("--rotor-z", type=float, default=9.3101752)
    parser.add_argument("--ground-z", type=float, required=True, help="Actual ground z [m]")
    parser.add_argument("--z-max", type=float, required=True, help="Mesh maximum z [m]; use the value reported by extraction")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--backend", default="TkAgg", choices=("TkAgg", "QtAgg"),
                        help="Interactive GUI backend (default: TkAgg)")
    parser.add_argument("--plot-output", type=Path, help="Optional plot file; no PNG is saved by default")
    args = parser.parse_args()
    predict_cylindrical_surface(
        disk_radius_m=args.rd,
        l_over_d=args.l_over_d,
        disk_loading=args.disk_loading,
        rotor_z_m=args.rotor_z,
        ground_z_m=args.ground_z,
        z_max_m=args.z_max,
        shape_ztheta=(256, 256),
        save_path=args.output,
        model_dir=args.model_dir,
        show_plot=not args.no_show,
        plot_path=args.plot_output,
        backend=args.backend,
    )
