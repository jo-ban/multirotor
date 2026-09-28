"""Compare physical CFD and prediction on the sampled r=2RD cylinder."""
import argparse
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FIELDS = ("u_r", "u_theta", "u_z", "magnitude")
LABELS = ("U_r (radial)", "U_theta (azimuthal)", "U_z (axial)", "|U| (speed)")


def comparison_fields(cfd, prediction):
    """Require matching finite physical fields; do not permit broadcasting."""
    result = []
    shape = None
    for key in FIELDS:
        actual = np.asarray(cfd[key], dtype=float)
        predicted = np.asarray(prediction[key], dtype=float)
        if (actual.ndim != 2 or actual.shape != predicted.shape
                or (shape is not None and actual.shape != shape)
                or not np.isfinite(actual).all() or not np.isfinite(predicted).all()):
            raise ValueError(f"{key}: expected matching finite [z, theta] fields")
        if key == "magnitude" and (np.any(actual < 0) or np.any(predicted < 0)):
            raise ValueError("Speed must be nonnegative")
        shape = actual.shape
        result.append((actual, predicted, np.abs(predicted - actual)))
    return result


def build_figure(theta_deg, z_m, radius_m, cfd, prediction, case_id="", view="3d"):
    """Four rows, CFD/prediction/absolute-error columns; metrics use full grids.

    The 3D display samples at most 64 heights and 96 angles for responsiveness.
    No angular averaging or axisymmetric reconstruction is performed.
    """
    if view not in ("3d", "2d"):
        raise ValueError("view must be 3d or 2d")
    fields = comparison_fields(cfd, prediction)
    theta = np.asarray(theta_deg, dtype=float)
    heights = np.asarray(z_m, dtype=float)
    radius = float(radius_m)
    if (theta.ndim != 1 or heights.ndim != 1 or min(theta.size, heights.size) < 2
            or not np.isfinite(theta).all() or not np.isfinite(heights).all()
            or not np.all(np.diff(theta) > 0) or not np.all(np.diff(heights) > 0)
            or theta[-1] - theta[0] >= 360 or not np.isfinite(radius) or radius <= 0
            or fields[0][0].shape != (heights.size, theta.size)):
        raise ValueError("Invalid cylinder coordinates or field shape")
    titles = []
    for label, (_, _, error) in zip(LABELS, fields):
        titles.extend((f"CFD · {label}", f"Prediction · {label}",
                       f"Abs error · MAE={error.mean():.4f} m/s"))
    fig = make_subplots(rows=4, cols=3, specs=[[{"type": "scene" if view == "3d" else "xy"}
                                              for _ in range(3)] for _ in range(4)],
                        subplot_titles=titles, horizontal_spacing=0.08, vertical_spacing=0.07)
    zi = np.unique(np.linspace(0, heights.size - 1, min(64, heights.size)).astype(int))
    ti = np.unique(np.linspace(0, theta.size - 1, min(96, theta.size)).astype(int))
    angles, z = np.meshgrid(np.deg2rad(np.r_[theta[ti], theta[ti][0] + 360]), heights[zi])
    x, y = radius * np.cos(angles), radius * np.sin(angles)
    for row, (label, values) in enumerate(zip(LABELS, fields), 1):
        limit = max(float(np.max(np.abs(values[0]))), float(np.max(np.abs(values[1]))), 1e-9)
        for col, value in enumerate(values, 1):
            error = col == 3
            low = 0 if error or row == 4 else -limit
            high = max(float(value.max()), 1e-9) if error else limit
            scale = "Reds" if error else ("Turbo" if row == 4 else "RdBu_r")
            index = (row - 1) * 3 + col
            axis = ("scene" if index == 1 else f"scene{index}")
            domain = fig.layout[axis].domain if view == "3d" else None
            colorbar = dict(title="m/s", thickness=9, len=0.17,
                            x=(domain.x[1] if domain else (0.28, 0.64, 1.0)[col - 1]),
                            y=(sum(domain.y) / 2 if domain else 0.9 - (row - 1) * 0.265))
            if view == "3d":
                sampled = value[np.ix_(zi, ti)]
                closed = np.column_stack((sampled, sampled[:, 0]))
                trace = go.Surface(x=x, y=y, z=z, surfacecolor=closed,
                    cmin=low, cmax=high, colorscale=scale, colorbar=colorbar,
                    customdata=np.stack((np.rad2deg(angles), closed), axis=-1),
                    hovertemplate="theta=%{customdata[0]:.2f}°<br>z=%{z:.3f} m"
                                  "<br>value=%{customdata[1]:.5f} m/s<extra></extra>")
            else:
                trace = go.Heatmap(x=theta, y=heights, z=value, zmin=low, zmax=high,
                    colorscale=scale, colorbar=colorbar,
                    hovertemplate="theta=%{x:.2f}°<br>z=%{y:.3f} m"
                                  "<br>value=%{z:.5f} m/s<extra></extra>")
            fig.add_trace(trace, row=row, col=col)
    if view == "3d":
        fig.update_scenes(xaxis_title="x relative to center [m]", yaxis_title="y relative to center [m]",
                          zaxis_title="z [m]", aspectmode="data", dragmode="orbit",
                          camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)))
    else:
        fig.update_xaxes(title_text="theta [deg]")
        fig.update_yaxes(title_text="z [m]")
    fig.update_layout(title=f"{escape(str(case_id))} · CFD / prediction / absolute error · r={radius:.4f} m",
                      template="plotly_white", height=1500, margin=dict(l=40, r=85, t=90, b=45),
                      showlegend=False, uirevision="validation")
    return fig


def load_comparison(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(theta_deg=archive["theta_deg"], z_m=archive["z_m"],
                    radius_m=float(archive["cylinder_radius_m"]),
                    case_id=str(archive["case_id"]),
                    cfd={key: archive[f"cfd_{key}"] for key in FIELDS},
                    prediction={key: archive[f"prediction_{key}"] for key in FIELDS})


def export_html(input_path, output_path):
    data = load_comparison(input_path)
    figure = build_figure(**data)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output), include_plotlyjs=True, full_html=True,
                      config={"responsive": True, "scrollZoom": True, "displaylogo": False})
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_html(args.input, args.output)
