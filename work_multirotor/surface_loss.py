"""L1 error of cylindrical velocity and its covariant surface gradient."""
import math

import torch


def surface_gradient(error, z_over_rd, radius_over_rd):
    """Return (B, 3, 2, Nz, Ntheta), ordered (z*, theta/r*).

    Velocities retain the same common TARGET_SCALE as value loss. Coordinates
    are z*=z/RD and r*=r/RD; there is no radial derivative or interpolation.
    Theta samples cover [0, 2*pi) uniformly, without a duplicate endpoint.
    """
    if error.ndim != 4 or error.shape[1] != 3:
        raise ValueError("Expected velocity channels (ur, utheta, uz) in (B,3,Nz,Ntheta)")
    batch, _, nz, ntheta = error.shape
    z = torch.as_tensor(z_over_rd, dtype=error.dtype, device=error.device)
    if z.ndim == 1:
        z = z.expand(batch, -1)
    r = torch.as_tensor(radius_over_rd, dtype=error.dtype, device=error.device).reshape(-1)
    if r.numel() == 1:
        r = r.expand(batch)
    if (nz < 3 or ntheta < 3 or z.shape != (batch, nz) or r.shape != (batch,)
            or not torch.isfinite(z).all() or not (z.diff(dim=-1) > 0).all()
            or not torch.isfinite(r).all() or not (r > 0).all()):
        raise ValueError("Require increasing actual z/RD coordinates and positive r/RD per sample")
    # Nonuniform central differences; second-order one-sided z endpoints.
    dz = torch.stack([torch.gradient(error[i], spacing=(z[i],), dim=(1,),
                                     edge_order=2)[0] for i in range(batch)])
    dtheta = (error.roll(-1, dims=-1) - error.roll(1, dims=-1)) / (4 * math.pi / ntheta)
    er, et, ez = error.unbind(dim=1)
    circumferential = torch.stack((dtheta[:, 0] - et,
                                   dtheta[:, 1] + er, dtheta[:, 2]), dim=1)
    circumferential = circumferential / r[:, None, None, None]
    return torch.stack((dz, circumferential), dim=2)


def velocity_losses(prediction, target, z_over_rd, radius_over_rd, lambda_gradient):
    if not math.isfinite(lambda_gradient) or lambda_gradient < 0:
        raise ValueError("lambda_gradient must be finite and nonnegative")
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    error = prediction - target
    value = error.abs().mean()
    gradient = surface_gradient(error, z_over_rd, radius_over_rd).abs().mean()
    weighted = lambda_gradient * gradient
    return dict(value_loss=value, gradient_loss=gradient,
                weighted_gradient_loss=weighted, total_loss=value + weighted)
