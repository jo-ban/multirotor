import argparse
from pathlib import Path

import numpy as np

from data_utils_cylindrical import (
    validation_ids_from_frame,
    TARGET_SCALE,
    cartesian_to_cylindrical_velocity,
    validate_checkpoint,
    plot_extent,
    load_case_input,
    make_input_surface,
)
from normalization import dimensional_velocity


DATA_ROOT = Path("./dataset_zrd")
MODEL_PATHS = {c: Path(f"rotor_unet_cyl2d_{c}.pth") for c in ("u", "v", "w")}


def validation_case_ids(csv_path="./cases.csv", data_root=DATA_ROOT):
    import pandas as pd

    dataframe = pd.read_csv(csv_path, dtype={"folder": str})
    return validation_ids_from_frame(data_root, dataframe)


def load_model(component, device, model_dir=None, expected_rd=None):
    import torch
    from model_cylindrical import CylindricalUNet2D

    path = MODEL_PATHS[component] if model_dir is None else Path(model_dir) / f"rotor_unet_cyl2d_{component}.pth"
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


def evaluate_case(case_id, device, show_plot=True, data_root=DATA_ROOT, model_dir=None, output_dir=None):
    import matplotlib.pyplot as plt
    import torch

    data_root = Path(data_root)
    meta = load_case_input(data_root / "inputs" / f"input_{case_id}.npz")
    input_array = make_input_surface(
        meta["l_over_d"], meta["z_over_rd"], meta["shape_ztheta"]
    )
    inputs = torch.from_numpy(input_array[None, ...]).to(device)
    cfd_nd = {
        c: np.load(data_root / f"targets_{c}" / f"{c}_{case_id}.npy")
        for c in ("u", "v", "w")
    }

    ai_nd = {}
    with torch.no_grad():
        for component in ("u", "v", "w"):
            model, scale = load_model(component, device, model_dir, meta["disk_radius_m"])
            ai_nd[component] = model(inputs)[0, 0].cpu().numpy() / scale
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    dl = meta["disk_loading"]
    cfd = {c: dimensional_velocity(cfd_nd[c], dl) for c in ("u", "v", "w")}
    ai = {c: dimensional_velocity(ai_nd[c], dl) for c in ("u", "v", "w")}
    theta_rad = np.linspace(0, 2 * np.pi, meta["shape_ztheta"][1], endpoint=False)
    cfd_cyl = cartesian_to_cylindrical_velocity(cfd["u"], cfd["v"], cfd["w"], theta_rad)
    ai_cyl = cartesian_to_cylindrical_velocity(ai["u"], ai["v"], ai["w"], theta_rad)
    maes = {
        c: float(np.mean(np.abs(cfd_cyl[c] - ai_cyl[c])))
        for c in ("u_r", "u_theta", "u_z")
    }
    magnitude_cfd = np.sqrt(cfd["u"] ** 2 + cfd["v"] ** 2 + cfd["w"] ** 2)
    magnitude_ai = np.sqrt(ai["u"] ** 2 + ai["v"] ** 2 + ai["w"] ** 2)
    magnitude_error = np.abs(magnitude_cfd - magnitude_ai)
    magnitude_mae = float(np.mean(magnitude_error))

    print(
        f"{case_id} | U_r MAE={maes['u_r']:.5f} | "
        f"U_theta MAE={maes['u_theta']:.5f} | U_z MAE={maes['u_z']:.5f} | "
        f"|U| MAE={magnitude_mae:.5f} m/s"
    )

    from visualize_validation import comparison_fields, export_html
    cfd_fields = {**cfd_cyl, "magnitude": magnitude_cfd}
    ai_fields = {**ai_cyl, "magnitude": magnitude_ai}
    fields = comparison_fields(cfd_fields, ai_fields)
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        comparison_path = destination / f"comparison_{case_id}.npz"
        np.savez_compressed(
            comparison_path, case_id=case_id,
            theta_deg=np.rad2deg(theta_rad), z_m=meta["z_m"],
            cylinder_radius_m=meta["cylinder_radius_m"],
            **{f"cfd_{key}": value for key, value in cfd_fields.items()},
            **{f"prediction_{key}": value for key, value in ai_fields.items()},
        )
        html_path = destination / f"comparison_{case_id}.html"
        export_html(comparison_path, html_path)
        print(f"Comparison HTML saved: {html_path}")

    if show_plot or output_dir is not None:
        theta_deg = np.linspace(0, 360, meta["shape_ztheta"][1], endpoint=False)
        extent = plot_extent(theta_deg, meta["z_m"])
        fig, axes = plt.subplots(4, 3, figsize=(17, 15))
        labels = ("U_r (radial)", "U_theta (azimuthal)", "U_z (axial)", "|U| (speed)")
        for row, (label, values) in enumerate(zip(labels, fields)):
            limit = max(float(np.max(np.abs(values[0]))), float(np.max(np.abs(values[1]))), 1e-9)
            for col, (data, title) in enumerate(zip(values, ("CFD", "Prediction", "Absolute error"))):
                ax = axes[row, col]
                is_error = col == 2
                image = ax.imshow(
                    data, origin="lower", extent=extent, aspect="auto",
                    cmap="Reds" if is_error else ("turbo" if row == 3 else "RdBu_r"),
                    vmin=0 if is_error or row == 3 else -limit,
                    vmax=max(float(data.max()), 1e-9) if is_error else limit,
                )
                ax.set_title(f"{title} · {label}")
                ax.set_xlabel("azimuth theta [deg]")
                ax.set_ylabel("z [m]")
                ax.set_ylim(float(meta["z_m"][0]), float(meta["z_m"][-1]))
                fig.colorbar(image, ax=ax, label="m/s")
        plt.suptitle(f"{case_id} | cylinder r=2R_D | magnitude MAE={magnitude_mae:.5f} m/s")
        plt.tight_layout()
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(output_dir) / f"error_{case_id}.png", dpi=150)
        if show_plot:
            plt.show()
        plt.close(fig)

    return {
        "case_id": case_id,
        **{f"mae_{c}": maes[c] for c in maes},
        "mae_magnitude": magnitude_mae,
    }


def regenerate_html(output_dir):
    """Read saved comparison arrays and overwrite only their HTML files."""
    from visualize_validation import export_html

    paths = sorted(Path(output_dir).glob("comparison_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No saved comparison_*.npz in {output_dir}; HTML-only regeneration requires existing evaluation results."
        )
    for path in paths:
        output = path.with_suffix(".html")
        export_html(path, output)
        print(f"Comparison HTML saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate validation cases")
    parser.add_argument("--csv", type=Path, default=Path("cases.csv"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--model-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--html-only", action="store_true",
                        help="Regenerate only HTML from comparison_*.npz in --output-dir; no inference or other outputs")
    args = parser.parse_args()
    if args.html_only:
        regenerate_html(args.output_dir)
        raise SystemExit(0)
    import torch
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    case_ids = validation_case_ids(args.csv, args.data_root)
    if not case_ids:
        raise RuntimeError("cases.csv에서 type='V' 검증 케이스를 찾지 못했습니다.")
    results = [evaluate_case(case_id, device, show_plot=not args.no_show,
                             data_root=args.data_root, model_dir=args.model_dir,
                             output_dir=args.output_dir) for case_id in case_ids]
    output = args.output_dir / "validation_errors_cyl2rd.csv"
    pd.DataFrame(results).to_csv(output, index=False)
    print(f"검증 결과 저장: {output}")
