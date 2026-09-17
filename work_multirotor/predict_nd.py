import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_utils_cylindrical import (
    DEFAULT_SHAPE_ZTHETA,
    TARGET_SCALE,
    make_input_surface,
    outer_radius_square_quadrotor,
)
from model_cylindrical import CylindricalUNet2D
from normalization import dimensional_velocity


MODEL_PATHS = {c: Path(f"rotor_unet_cyl2d_{c}.pth") for c in ("u", "v", "w")}


def load_component_model(component, device, model_dir=None):
    path = MODEL_PATHS[component] if model_dir is None else Path(model_dir) / f"rotor_unet_cyl2d_{component}.pth"
    if not path.exists():
        raise FileNotFoundError(f"{path}가 없습니다. train_nd.py로 먼저 학습하세요.")
    checkpoint = torch.load(path, map_location=device)
    model = CylindricalUNet2D(
        in_channels=int(checkpoint.get("in_channels", 5)),
        out_channels=1,
        base_channels=int(checkpoint.get("base_channels", 16)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, float(checkpoint.get("target_scale", TARGET_SCALE))


def predict_cylindrical_surface(
    rotor_spacing_m,
    rotor_diameter_m,
    disk_loading,
    rotor_z_m,
    ground_z_m,
    shape_ztheta=DEFAULT_SHAPE_ZTHETA,
    save_path="prediction_cylinder_r2RD.npz",
    show_plot=True,
    model_dir=None,
    plot_path=None,
):
    """L과 D로 외접반경 R_D를 계산하고 r=2R_D 원통면의 U/V/W를 예측한다."""
    l_over_d = float(rotor_spacing_m) / float(rotor_diameter_m)
    disk_radius_m = outer_radius_square_quadrotor(rotor_spacing_m, rotor_diameter_m)
    height_m = abs(float(rotor_z_m) - float(ground_z_m))
    if height_m <= 0:
        raise ValueError("rotor_z_m과 ground_z_m은 서로 달라야 합니다.")
    height_over_rd = height_m / disk_radius_m
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_surface = make_input_surface(l_over_d, height_over_rd, shape_ztheta)
    inputs = torch.from_numpy(input_surface[None, ...]).to(device)

    prediction_nd = {}
    with torch.no_grad():
        for component in ("u", "v", "w"):
            model, scale = load_component_model(component, device, model_dir)
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
    s_over_rd = np.linspace(0.0, height_over_rd, nz, dtype=np.float32)
    z_m = np.linspace(rotor_z_m, ground_z_m, nz, dtype=np.float32)

    # 후처리 편의를 위해 원통좌표 속도도 함께 저장한다.
    cos_theta = np.cos(theta_rad)[None, :]
    sin_theta = np.sin(theta_rad)[None, :]
    radial = prediction["u"] * cos_theta + prediction["v"] * sin_theta
    tangential = -prediction["u"] * sin_theta + prediction["v"] * cos_theta

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        u_mps=prediction["u"],
        v_mps=prediction["v"],
        w_mps=prediction["w"],
        radial_mps=radial.astype(np.float32),
        tangential_mps=tangential.astype(np.float32),
        magnitude_mps=magnitude,
        theta_deg=theta_deg,
        s_over_rd=s_over_rd,
        z_m=z_m,
        rotor_z_m=np.float32(rotor_z_m),
        ground_z_m=np.float32(ground_z_m),
        height_m=np.float32(height_m),
        height_over_rd=np.float32(height_over_rd),
        rotor_spacing_m=np.float32(rotor_spacing_m),
        rotor_diameter_m=np.float32(rotor_diameter_m),
        l_over_d=np.float32(l_over_d),
        disk_radius_m=np.float32(disk_radius_m),
        cylinder_radius_m=np.float32(2.0 * disk_radius_m),
        disk_loading=np.float32(disk_loading),
    )
    print(f"r=2R_D 원통 표면 예측 결과 저장: {save_path}")

    if show_plot or plot_path is not None:
        extent = (0.0, 360.0, 0.0, height_over_rd)
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, data, title in zip(
            axes.flat,
            (prediction["u"], prediction["v"], prediction["w"], magnitude),
            ("U", "V", "W", "Magnitude"),
        ):
            image = ax.imshow(data, origin="lower", extent=extent, aspect="auto", cmap="turbo")
            ax.set_title(title)
            ax.set_xlabel("azimuth theta [deg]")
            ax.set_ylabel("s/R_D (rotor to ground)")
            fig.colorbar(image, ax=ax, label="m/s")
        plt.suptitle(f"Cylinder r=2R_D prediction | L/D={l_over_d:.3f}")
        plt.tight_layout()
        if plot_path is not None:
            Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=150)
        if show_plot:
            plt.show()
        plt.close(fig)

    return prediction["u"], prediction["v"], prediction["w"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a cylindrical velocity field")
    parser.add_argument("--model-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("prediction_cylinder_r2RD.npz"))
    parser.add_argument("--rotor-spacing", type=float, default=1.25)
    parser.add_argument("--rotor-diameter", type=float, default=1.0)
    parser.add_argument("--disk-loading", type=float, default=153.22)
    parser.add_argument("--rotor-z", type=float, default=2.0)
    parser.add_argument("--ground-z", type=float, default=0.0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    predict_cylindrical_surface(
        rotor_spacing_m=args.rotor_spacing,
        rotor_diameter_m=args.rotor_diameter,
        disk_loading=args.disk_loading,
        rotor_z_m=args.rotor_z,
        ground_z_m=args.ground_z,
        shape_ztheta=(256, 256),
        save_path=args.output,
        model_dir=args.model_dir,
        show_plot=not args.no_show,
        plot_path=args.output.with_suffix(".png"),
    )
