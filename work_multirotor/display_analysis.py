"""RD-limited display and full-resolution error statistics (physical SI units)."""
import numpy as np
from data_utils_cylindrical import CYLINDER_RADIUS_OVER_RD, CYLINDER_RADII_OVER_RD

FIELDS = ("u_r", "u_theta", "u_z", "magnitude")


def display_window(z_m, rd_m):
    """Keep absolute z <= RD, retaining the existing lower z bound."""
    z = np.asarray(z_m, dtype=float)
    rd = float(rd_m)
    if (z.ndim != 1 or z.size < 2 or not np.isfinite(z).all()
            or not np.all(np.diff(z) > 0) or not np.isfinite(rd) or rd <= 0):
        raise ValueError("Expected increasing finite z [m] and positive RD [m]")
    mask = z <= rd
    if not mask.any() or min(rd, z[-1]) <= z[0]:
        raise ValueError("No nonempty display interval at z <= RD")
    return mask, (float(z[0]), float(min(rd, z[-1])))


def error_statistics(theta_deg, z_m, radius_m, cfd, prediction, rd_m=None, radius_over_rd=CYLINDER_RADIUS_OVER_RD):
    """Extrema use original samples, never the decimated/seam/interpolated mesh.

    NMAE retains the existing aggregate formula. A spatial percentage maximum
    uses |prediction-CFD|/|CFD|*100; exact zero references alone are excluded.
    Ties select the first point in increasing z then theta order.
    """
    radius = float(radius_m)
    ratio = float(radius_over_rd)
    if ratio not in CYLINDER_RADII_OVER_RD:
        raise ValueError("Expected FATO or SA radius/RD")
    rd = radius / ratio if rd_m is None else float(rd_m)
    if not np.isfinite(radius) or radius <= 0 or not np.isclose(radius, ratio * rd, rtol=1e-6):
        raise ValueError(f"Expected cylinder radius = {ratio:g} RD")
    theta, z = np.asarray(theta_deg, dtype=float), np.asarray(z_m, dtype=float)
    mask, bounds = display_window(z, rd)
    if (theta.ndim != 1 or theta.size < 2 or not np.isfinite(theta).all()
            or not np.all(np.diff(theta) > 0) or theta[-1] - theta[0] >= 360):
        raise ValueError("Expected increasing, nonduplicated theta coordinates")
    heights = z[mask]
    result = {}
    for key in FIELDS:
        actual = np.asarray(cfd[key], dtype=float)
        predicted = np.asarray(prediction[key], dtype=float)
        if (actual.shape != (z.size, theta.size) or predicted.shape != actual.shape
                or not np.isfinite(actual).all() or not np.isfinite(predicted).all()):
            raise ValueError(f"{key}: expected finite matching [z, theta] arrays")
        actual, predicted = actual[mask], predicted[mask]
        error = np.abs(predicted - actual)
        valid = actual != 0
        percent = np.full(actual.shape, np.nan)
        np.divide(error, np.abs(actual), out=percent, where=valid)
        percent *= 100

        def point(values):
            i, j = np.unravel_index(np.nanargmax(values), values.shape)
            angle = np.deg2rad(theta[j])
            return dict(value=float(values[i, j]), theta_deg=float(theta[j]),
                        z_m=float(heights[i]), x_m=float(radius * np.cos(angle)),
                        y_m=float(radius * np.sin(angle)), error_mps=float(error[i, j]),
                        reference_mps=float(actual[i, j]), prediction_mps=float(predicted[i, j]))

        reference = float(np.mean(np.abs(actual)))
        result[key] = dict(mae_mps=float(error.mean()),
                           nmae_pct=float(100 * error.mean() / reference) if reference else None,
                           max_error=point(error), max_percent=point(percent) if valid.any() else None,
                           zero_reference_count=int((~valid).sum()), sample_count=int(error.size))
    return result, mask, bounds


def point_label(point, kind, html=False):
    sep = "<br>" if html else "\n"
    label, unit = ("A: max |error|", "m/s") if kind == "error" else ("B: max error rate", "%")
    if point is None:
        return f"{label}: N/A (all CFD = 0)"
    return sep.join((f"{label} = {point['value']:.6g} {unit}",
                     f"theta={point['theta_deg']:.4f} deg, z={point['z_m']:.6g} m",
                     f"x={point['x_m']:.6g} m, y={point['y_m']:.6g} m"))


def metrics_row(case_id, stats, bounds):
    row = dict(case_id=case_id, display_z_min_m=bounds[0], display_z_max_m=bounds[1])
    for field, values in stats.items():
        row[f"mae_{field}"] = values["mae_mps"]
        row[f"nmae_{field}_pct"] = values["nmae_pct"]
        row[f"zero_reference_count_{field}"] = values["zero_reference_count"]
        row[f"sample_count_{field}"] = values["sample_count"]
        for kind, unit in (("error", "mps"), ("percent", "pct")):
            p = values[f"max_{kind}"]
            row[f"max_{kind}_{field}_{unit}"] = None if p is None else p["value"]
            for coordinate in ("theta_deg", "x_m", "y_m", "z_m"):
                row[f"max_{kind}_{field}_{coordinate}"] = None if p is None else p[coordinate]
    return row


def region_component_menus(region, component, labels, has_sa):
    """Shared offline Plotly controls; unavailable SA never selects FATO data."""
    def button(label, target):
        return dict(label=label, method="animate", args=[[target], dict(
            mode="immediate", frame=dict(duration=0, redraw=True),
            transition=dict(duration=0))])

    regions = [button("FATO", "FATO/0")]
    regions.append(button("SA", "SA/0") if has_sa else
                   dict(label="SA 데이터 없음", method="skip", args=[]))
    return [dict(type="buttons", direction="right", buttons=regions,
                 active=0 if region == "FATO" else 1, x=0, y=1.24,
                 xanchor="left", yanchor="top", showactive=has_sa),
            dict(type="buttons", direction="right",
                 buttons=[button(label, f"{region}/{i}") for i, label in enumerate(labels)],
                 active=component, x=0, y=1.12, xanchor="left", yanchor="top")]
