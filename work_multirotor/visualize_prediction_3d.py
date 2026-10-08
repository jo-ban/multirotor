"""Export an existing prediction NPZ to a self-contained interactive HTML."""
import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from display_analysis import display_window, region_component_menus
from data_utils_cylindrical import CYLINDER_RADIUS_OVER_RD, SA_RADIUS_OVER_RD


FIELDS = {
    "u_r_mps": "U_r (radial)",
    "u_theta_mps": "U_theta (azimuthal)",
    "u_z_mps": "U_z (axial)",
    "magnitude_mps": "|U| (speed)",
}


def _component_figure(path, component=0, radius_over_rd=CYLINDER_RADIUS_OVER_RD):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    theta = np.asarray(data["theta_deg"], dtype=float)
    heights = np.asarray(data["z_m"], dtype=float)
    radius = float(data["cylinder_radius_m"])
    if (theta.ndim != 1 or heights.ndim != 1 or min(theta.size, heights.size) < 2
            or not np.isfinite(theta).all() or not np.isfinite(heights).all()
            or not np.all(np.diff(theta) > 0) or not np.all(np.diff(heights) > 0)
            or theta[-1] - theta[0] >= 360 or not np.isfinite(radius) or radius <= 0):
        raise ValueError("Invalid cylinder coordinates; expected increasing theta/z and positive radius")
    for key in FIELDS:
        if data[key].shape != (heights.size, theta.size) or not np.isfinite(data[key]).all():
            raise ValueError(f"{key}: expected finite [z, theta] array")

    rd = float(data.get("disk_radius_m", radius / radius_over_rd))
    if not np.isclose(radius, radius_over_rd * rd, rtol=1e-6):
        raise ValueError("Cylinder radius does not match the selected region")
    mask, bounds = display_window(heights, rd)
    heights = heights[mask]
    for key in FIELDS:
        data[key] = data[key][mask]

    # Repeat the first column at 360 degrees to close the display seam.
    angles, z = np.meshgrid(np.deg2rad(np.r_[theta, theta[0] + 360]), heights)
    x, y = radius * np.cos(angles), radius * np.sin(angles)
    fig = go.Figure()
    for index, (key, label) in enumerate(FIELDS.items()):
        if index != component:
            continue
        values = np.concatenate((data[key], data[key][:, :1]), axis=1)
        limit = max(float(np.max(np.abs(values))), 1e-9)
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, surfacecolor=values, visible=True,
            colorscale="Turbo" if key == "magnitude_mps" else "RdBu_r",
            cmin=0 if key == "magnitude_mps" else -limit, cmax=limit,
            colorbar=dict(title=f"{label}<br>m/s"), name=label,
            customdata=np.stack((np.rad2deg(angles), values), axis=-1),
            hovertemplate=("theta=%{customdata[0]:.1f}°<br>z=%{z:.3f} m"
                           f"<br>{label}=%{{customdata[1]:.3f}} m/s<extra></extra>"),
        ))

    # Schematic square rotor layout, not an aircraft CAD model.
    geometry = ("rotor_spacing_m", "rotor_diameter_m", "rotor_z_m")
    if all(key in data for key in geometry):
        spacing, diameter, rotor_z = (float(data[key]) for key in geometry)
        if not np.isfinite([spacing, diameter, rotor_z]).all() or min(spacing, diameter) <= 0:
            raise ValueError("Invalid rotor geometry")
        angle = np.linspace(0, 2 * np.pi, 65)
        for cx, cy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            cx, cy = cx * spacing / 2, cy * spacing / 2
            fig.add_trace(go.Scatter3d(
                x=cx + diameter / 2 * np.cos(angle),
                y=cy + diameter / 2 * np.sin(angle), z=np.full(angle.size, rotor_z),
                mode="lines", line=dict(color="#64748b", width=5),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter3d(
                x=[0, cx], y=[0, cy], z=[rotor_z, rotor_z], mode="lines",
                line=dict(color="#64748b", width=6), showlegend=False, hoverinfo="skip",
            ))
    fig.update_layout(
        title=f"Cylinder velocity · r = {radius_over_rd:g}R_D", template="plotly_white",
        margin=dict(l=20, r=30, t=190, b=45), height=760,
        scene=dict(zaxis_range=bounds, xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]",
                   aspectmode="data", dragmode="orbit", uirevision="cylinder",
                   camera=dict(eye=dict(x=1.6, y=1.6, z=0.9))),
        annotations=[dict(text="Drag: rotate · Scroll: zoom · Rotor layout: schematic, center (0, 0)",
                          x=0.5, y=-0.05, xref="paper", yref="paper", showarrow=False)],
    )
    return fig


def build_figure(path, sa_path=None):
    """Independent region files; only the selected velocity surface is live."""
    frames, first = [], None
    for region, source, ratio in (("FATO", path, CYLINDER_RADIUS_OVER_RD),
                                  ("SA", sa_path, SA_RADIUS_OVER_RD)):
        if source is None:
            continue
        for component in range(len(FIELDS)):
            fig = _component_figure(source, component, ratio)
            fig.update_layout(updatemenus=region_component_menus(
                region, component, FIELDS.values(), sa_path is not None))
            if first is None:
                first = fig
            # Rotor overlays are region-specific too; keep trace slots stable.
            if len(fig.data) != len(first.data):
                raise ValueError("FATO and SA prediction files must contain matching rotor geometry metadata")
            frames.append(go.Frame(name=f"{region}/{component}", data=fig.data,
                                   traces=list(range(len(fig.data))), layout=fig.layout))
    first.frames = frames
    return first


def export_html(input_path, output_path, sa_input_path=None):
    figure = build_figure(input_path, sa_path=sa_input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output), include_plotlyjs=True, full_html=True, auto_play=False,
                      config={"responsive": True, "scrollZoom": True, "displaylogo": False})
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Prediction NPZ")
    parser.add_argument("--output", type=Path, default=Path("prediction_3d.html"))
    args = parser.parse_args()
    export_html(args.input, args.output)
    print(f"Interactive HTML saved: {args.output}")
