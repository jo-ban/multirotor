"""추출된 r=2R_D 원통 표면을 theta-z 평면으로 펼쳐 확인한다."""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np

from data_utils_cylindrical import load_case_input


DATA_ROOT = Path("./dataset_nd")


def visualize_extracted_data(case_id=None):
    inputs = sorted((DATA_ROOT / "inputs").glob("input_*.npz"))
    if not inputs:
        raise RuntimeError("데이터가 없습니다. extract_nd.py를 먼저 실행하세요.")

    selected = random.choice(inputs) if case_id is None else DATA_ROOT / "inputs" / f"input_{case_id}.npz"
    case_id = selected.name[len("input_"):-len(".npz")]
    meta = load_case_input(selected)
    fields = {
        c: np.load(DATA_ROOT / f"targets_{c}" / f"{c}_{case_id}.npy")
        for c in ("u", "v", "w")
    }
    fields["magnitude"] = np.sqrt(fields["u"] ** 2 + fields["v"] ** 2 + fields["w"] ** 2)

    extent = [0.0, 360.0, 0.0, meta["height_over_rd"]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, key in zip(axes.flat, ("u", "v", "w", "magnitude")):
        image = ax.imshow(fields[key], origin="lower", extent=extent, aspect="auto", cmap="turbo")
        ax.set_title(key.upper())
        ax.set_xlabel("azimuth theta [deg]")
        ax.set_ylabel("s/R_D (rotor to ground)")
        fig.colorbar(image, ax=ax, label="V/Vi")

    plt.suptitle(
        f"{case_id} | cylinder r=2R_D | L/D={meta['l_over_d']:.3f} | "
        f"R_D={meta['disk_radius_m']:.3f} m"
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_extracted_data()
