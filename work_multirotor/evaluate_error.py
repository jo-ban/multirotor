from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_utils_cylindrical import (
    TARGET_SCALE,
    disk_radius_from_row,
    height_over_rd_from_row,
    l_over_d_from_row,
    load_case_input,
    make_case_id,
    make_input_surface,
)
from model_cylindrical import CylindricalUNet2D
from normalization import dimensional_velocity


DATA_ROOT = Path("./dataset_nd")
MODEL_PATHS = {c: Path(f"rotor_unet_cyl2d_{c}.pth") for c in ("u", "v", "w")}


def validation_case_ids(csv_path="./cases.csv"):
    dataframe = pd.read_csv(csv_path)
    if "type" not in dataframe.columns:
        return []
    rows = dataframe[dataframe["type"].astype(str).str.upper() == "V"]
    return [
        make_case_id(
            l_over_d_from_row(row),
            disk_radius_from_row(row),
            height_over_rd_from_row(row),
        )
        for _, row in rows.iterrows()
    ]


def load_model(component, device):
    checkpoint = torch.load(MODEL_PATHS[component], map_location=device)
    model = CylindricalUNet2D(
        in_channels=int(checkpoint.get("in_channels", 5)),
        out_channels=1,
        base_channels=int(checkpoint.get("base_channels", 16)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, float(checkpoint.get("target_scale", TARGET_SCALE))


def evaluate_case(case_id, device, show_plot=True):
    meta = load_case_input(DATA_ROOT / "inputs" / f"input_{case_id}.npz")
    input_array = make_input_surface(
        meta["l_over_d"], meta["height_over_rd"], meta["shape_ztheta"]
    )
    inputs = torch.from_numpy(input_array[None, ...]).to(device)
    cfd_nd = {
        c: np.load(DATA_ROOT / f"targets_{c}" / f"{c}_{case_id}.npy")
        for c in ("u", "v", "w")
    }

    ai_nd = {}
    with torch.no_grad():
        for component in ("u", "v", "w"):
            model, scale = load_model(component, device)
            ai_nd[component] = model(inputs)[0, 0].cpu().numpy() / scale
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    dl = meta["disk_loading"]
    cfd = {c: dimensional_velocity(cfd_nd[c], dl) for c in ("u", "v", "w")}
    ai = {c: dimensional_velocity(ai_nd[c], dl) for c in ("u", "v", "w")}
    maes = {c: float(np.mean(np.abs(cfd[c] - ai[c]))) for c in ("u", "v", "w")}
    magnitude_cfd = np.sqrt(cfd["u"] ** 2 + cfd["v"] ** 2 + cfd["w"] ** 2)
    magnitude_ai = np.sqrt(ai["u"] ** 2 + ai["v"] ** 2 + ai["w"] ** 2)
    magnitude_error = np.abs(magnitude_cfd - magnitude_ai)
    magnitude_mae = float(np.mean(magnitude_error))

    print(
        f"{case_id} | U MAE={maes['u']:.5f} | V MAE={maes['v']:.5f} | "
        f"W MAE={maes['w']:.5f} | |V| MAE={magnitude_mae:.5f} m/s"
    )

    if show_plot:
        extent = (0.0, 360.0, 0.0, meta["height_over_rd"])
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, data, title in zip(
            axes,
            (magnitude_cfd, magnitude_ai, magnitude_error),
            ("CFD magnitude", "AI magnitude", "Absolute error"),
        ):
            image = ax.imshow(data, origin="lower", extent=extent, aspect="auto", cmap="turbo")
            ax.set_title(title)
            ax.set_xlabel("azimuth theta [deg]")
            ax.set_ylabel("s/R_D (rotor to ground)")
            fig.colorbar(image, ax=ax, label="m/s")
        plt.suptitle(f"{case_id} | cylinder r=2R_D | magnitude MAE={magnitude_mae:.5f} m/s")
        plt.tight_layout()
        plt.show()

    return {
        "case_id": case_id,
        **{f"mae_{c}": maes[c] for c in maes},
        "mae_magnitude": magnitude_mae,
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    case_ids = validation_case_ids("./cases.csv")
    if not case_ids:
        raise RuntimeError("cases.csv에서 type='V' 검증 케이스를 찾지 못했습니다.")
    results = [evaluate_case(case_id, device, show_plot=True) for case_id in case_ids]
    pd.DataFrame(results).to_csv("validation_errors_cyl2rd.csv", index=False)
    print("검증 결과 저장: validation_errors_cyl2rd.csv")
