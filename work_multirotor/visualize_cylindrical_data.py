"""추출된 r=2R_D 원통 표면을 theta-z 평면으로 펼쳐 확인한다."""

import argparse
import os
from pathlib import Path
import random

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import numpy as np

from data_utils_cylindrical import load_case_input, plot_extent


DATA_ROOT = Path("./dataset_zrd")


def visualize_extracted_data(case_id=None, data_root=DATA_ROOT, output_dir=Path("results/visualize")):
    data_root = Path(data_root)
    inputs = sorted((data_root / "inputs").glob("input_*.npz"))
    if not inputs:
        raise RuntimeError("데이터가 없습니다. extract_nd.py를 먼저 실행하세요.")

    selected = random.choice(inputs) if case_id is None else data_root / "inputs" / f"input_{case_id}.npz"
    case_id = selected.name[len("input_"):-len(".npz")]
    meta = load_case_input(selected)
    fields = {
        c: np.load(data_root / f"targets_{c}" / f"{c}_{case_id}.npy")
        for c in ("u", "v", "w")
    }
    fields["magnitude"] = np.sqrt(fields["u"] ** 2 + fields["v"] ** 2 + fields["w"] ** 2)

    theta_deg = np.linspace(0, 360, meta["shape_ztheta"][1], endpoint=False)
    extent = plot_extent(theta_deg, meta["z_m"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, key in zip(axes.flat, ("u", "v", "w", "magnitude")):
        image = ax.imshow(fields[key], origin="lower", extent=extent, aspect="auto", cmap="turbo")
        ax.set_title(key.upper())
        ax.set_xlabel("azimuth theta [deg]")
        ax.set_ylabel("z [m]")
        ax.set_ylim(float(meta["z_m"][0]), float(meta["z_m"][-1]))
        fig.colorbar(image, ax=ax, label="V/Vi")

    plt.suptitle(
        f"{case_id} | cylinder r=2R_D | L/D={meta['l_over_d']:.3f} | "
        f"R_D={meta['disk_radius_m']:.3f} m"
    )
    plt.tight_layout()
    output = Path(output_dir) / f"surface_{case_id}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save cylindrical data plots without a GUI")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("results/visualize"))
    parser.add_argument("--case-id", help="Default: choose one random case")
    args = parser.parse_args()
    visualize_extracted_data(args.case_id, args.data_root, args.output_dir)
