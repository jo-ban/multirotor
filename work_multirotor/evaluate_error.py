import argparse
from pathlib import Path

import numpy as np

from data_utils_cylindrical import (
    validation_ids_from_frame,
    TARGET_SCALE,
    CYLINDER_RADIUS_OVER_RD,
    SA_RADIUS_OVER_RD,
    cartesian_to_cylindrical_velocity,
    validate_checkpoint,
    load_case_input,
    make_input_surface,
)
from joint_checkpoint import load_joint_model
from normalization import dimensional_velocity


DATA_ROOT = Path("./dataset_zrd")
MODEL_PATHS = {c: Path(f"rotor_unet_cyl2d_{c}.pth") for c in ("u", "v", "w")}


def validation_case_ids(csv_path="./cases.csv", data_root=DATA_ROOT):
    import pandas as pd

    dataframe = pd.read_csv(csv_path, dtype={"folder": str})
    return validation_ids_from_frame(data_root, dataframe)


def load_model(component, device, model_dir=None, expected_rd=None,
               expected_radius_over_rd=CYLINDER_RADIUS_OVER_RD):
    import torch
    from model_cylindrical import CylindricalUNet2D

    path = MODEL_PATHS[component] if model_dir is None else Path(model_dir) / f"rotor_unet_cyl2d_{component}.pth"
    checkpoint = torch.load(path, map_location=device)
    validate_checkpoint(checkpoint, expected_rd, expected_radius_over_rd)
    if checkpoint["state_dict"]["out_conv.weight"].shape[0] != 1:
        raise ValueError("Legacy component loader requires one Cartesian channel; use load_joint_model for joint cylindrical weights")
    model = CylindricalUNet2D(
        in_channels=4,
        out_channels=1,
        base_channels=int(checkpoint.get("base_channels", 16)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, float(checkpoint.get("target_scale", TARGET_SCALE))


def evaluate_case(case_id, device, show_plot=True, data_root=DATA_ROOT, model_dir=None, output_dir=None,
                  return_payload=False):
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

    model, scale = load_joint_model(device, model_dir, meta["disk_radius_m"], meta["radius_over_rd"])
    with torch.no_grad():
        ai_nd = model(inputs)[0].cpu().numpy() / scale
    del model
    dl = meta["disk_loading"]
    cfd = {c: dimensional_velocity(cfd_nd[c], dl) for c in ("u", "v", "w")}
    theta_rad = np.linspace(0, 2 * np.pi, meta["shape_ztheta"][1], endpoint=False)
    cfd_cyl = cartesian_to_cylindrical_velocity(cfd["u"], cfd["v"], cfd["w"], theta_rad)
    ai_cyl = {key: dimensional_velocity(ai_nd[i], dl)
              for i, key in enumerate(("u_r", "u_theta", "u_z"))}
    magnitude_cfd = np.sqrt(sum(field ** 2 for field in cfd_cyl.values()))
    magnitude_ai = np.sqrt(sum(field ** 2 for field in ai_cyl.values()))
    from visualize_validation import export_html, build_static_figure
    from display_analysis import error_statistics, metrics_row, point_label
    cfd_fields = {**cfd_cyl, "magnitude": magnitude_cfd}
    ai_fields = {**ai_cyl, "magnitude": magnitude_ai}
    comparison = dict(theta_deg=np.rad2deg(theta_rad), z_m=meta["z_m"],
                      radius_m=meta["cylinder_radius_m"], rd_m=meta["disk_radius_m"],
                      cfd=cfd_fields, prediction=ai_fields, case_id=case_id)
    stats, _, bounds = error_statistics(**{k: v for k, v in comparison.items() if k != "case_id"},
                                        radius_over_rd=meta["radius_over_rd"])
    print(f"{case_id} | display/metrics z={bounds} m; zero CFD excluded from point rate")
    for key, values in stats.items():
        print(key, point_label(values["max_error"], "error"),
              point_label(values["max_percent"], "percent"),
              f"zero references excluded={values['zero_reference_count']}", sep="\n")
    payload = dict(
        case_id=case_id, theta_deg=np.rad2deg(theta_rad), z_m=meta["z_m"],
        cylinder_radius_m=meta["cylinder_radius_m"], disk_radius_m=meta["disk_radius_m"],
        **{f"cfd_{key}": value for key, value in cfd_fields.items()},
        **{f"prediction_{key}": value for key, value in ai_fields.items()})
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        comparison_path = destination / f"comparison_{case_id}.npz"
        np.savez_compressed(
            comparison_path, case_id=case_id,
            theta_deg=np.rad2deg(theta_rad), z_m=meta["z_m"],
            cylinder_radius_m=meta["cylinder_radius_m"], disk_radius_m=meta["disk_radius_m"],
            **{f"cfd_{key}": value for key, value in cfd_fields.items()},
            **{f"prediction_{key}": value for key, value in ai_fields.items()},
        )
        html_path = destination / f"comparison_{case_id}.html"
        export_html(comparison_path, html_path)
        print(f"Comparison HTML saved: {html_path}")

    if show_plot or output_dir is not None:
        fig = build_static_figure(**comparison, radius_over_rd=meta["radius_over_rd"])
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(output_dir) / f"error_{case_id}.png", dpi=150)
        if show_plot:
            plt.show()
        plt.close(fig)

    row = metrics_row(case_id, stats, bounds)
    return (row, payload) if return_payload else row


def evaluate_regions(case_id, device, data_root, model_dir, output_dir,
                     sa_data_root=None, sa_model_dir=None, show_plot=False):
    row, payload = evaluate_case(case_id, device, show_plot=show_plot,
        data_root=data_root, model_dir=model_dir, output_dir=output_dir, return_payload=True)
    rows = [row]
    if sa_data_root is not None:
        sa_row, sa = evaluate_case(case_id, device, show_plot=False,
            data_root=sa_data_root, model_dir=sa_model_dir, return_payload=True)
        sa_row["case_id"] = "SA/" + case_id  # result label only; extraction identity is unchanged
        rows.append(sa_row)
        payload.update({f"sa_{key}": value for key, value in sa.items()})
        path = Path(output_dir) / f"comparison_{case_id}.npz"
        np.savez_compressed(path, **payload)
        from visualize_validation import export_html
        export_html(path, path.with_suffix(".html"))
    return rows


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


def regenerate_results(output_dir):
    """Replot saved arrays and update display-range statistics without inference."""
    import csv
    import matplotlib.pyplot as plt
    from visualize_validation import load_comparison, export_html, build_static_figure
    from display_analysis import error_statistics, metrics_row

    paths = sorted(Path(output_dir).glob("comparison_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No saved comparison_*.npz in {output_dir}")
    rows = []
    for path in paths:
        data = load_comparison(path)
        stats, _, bounds = error_statistics(**{k: v for k, v in data.items() if k != "case_id"})
        rows.append(metrics_row(data["case_id"], stats, bounds))
        with np.load(path, allow_pickle=False) as archive:
            bundled_sa = "sa_cylinder_radius_m" in archive
        if bundled_sa:
            sa = load_comparison(path, "SA")
            sa_stats, _, sa_bounds = error_statistics(
                **{k: v for k, v in sa.items() if k != "case_id"}, radius_over_rd=SA_RADIUS_OVER_RD)
            rows.append(metrics_row("SA/" + sa["case_id"], sa_stats, sa_bounds))
        export_html(path, path.with_suffix(".html"))
        fig = build_static_figure(**data)
        fig.savefig(path.with_name("error_" + path.stem.removeprefix("comparison_") + ".png"), dpi=150)
        plt.close(fig)
    output = Path(output_dir) / "validation_errors_cyl2rd.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated HTML, PNG and RD-limited error CSV: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate validation cases")
    parser.add_argument("--csv", type=Path, default=Path("cases.csv"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--model-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--no-show", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--replot", action="store_true", help="Regenerate HTML, PNG and error CSV from saved comparisons, without inference")
    mode.add_argument("--html-only", action="store_true",
                        help="Regenerate only HTML from comparison_*.npz in --output-dir; no inference or other outputs")
    parser.add_argument("--sa-data-root", type=Path)
    parser.add_argument("--sa-model-dir", type=Path)
    args = parser.parse_args()
    if args.replot:
        regenerate_results(args.output_dir)
        raise SystemExit(0)
    if args.html_only:
        regenerate_html(args.output_dir)
        raise SystemExit(0)
    import torch
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    case_ids = validation_case_ids(args.csv, args.data_root)
    if not case_ids:
        raise RuntimeError("cases.csv에서 type='V' 검증 케이스를 찾지 못했습니다.")
    sa_root = args.sa_data_root
    if sa_root is None and args.data_root.with_name("dataset_zrd_sa").is_dir():
        sa_root = args.data_root.with_name("dataset_zrd_sa")
    sa_models = args.sa_model_dir or args.model_dir.resolve().with_name("models_zrd_sa")
    if sa_root is not None:
        if sa_root.resolve() == args.data_root.resolve() or sa_models.resolve() == args.model_dir.resolve():
            raise ValueError("FATO and SA require separate dataset/model directories")
        if set(validation_case_ids(args.csv, sa_root)) != set(case_ids):
            raise ValueError("FATO and SA validation case identifiers differ")
    results = []
    for case_id in case_ids:
        results.extend(evaluate_regions(case_id, device, args.data_root, args.model_dir,
            args.output_dir, sa_root, sa_models, show_plot=not args.no_show))
    output = args.output_dir / "validation_errors_cyl2rd.csv"
    pd.DataFrame(results).to_csv(output, index=False)
    print(f"검증 결과 저장: {output}")
