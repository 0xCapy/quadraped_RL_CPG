from __future__ import annotations

from typing import Sequence

import torch


def first_attr(obj, names: Sequence[str], default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(-1)
    return x


def ensure_float(x: torch.Tensor, like: torch.Tensor | None = None) -> torch.Tensor:
    x = x.float()
    if like is not None:
        x = x.to(device=like.device, dtype=like.dtype)
    return x


def safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return torch.sqrt(torch.clamp(torch.sum(x * x, dim=dim, keepdim=keepdim), min=1.0e-12))


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def quat_to_euler_xyz(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quaternion to XYZ Euler.

    Expects shape [..., 4] in wxyz order.
    """
    q = quat_wxyz
    w, x, y, z = q.unbind(dim=-1)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device, dtype=yaw.dtype)
    q[:, 0] = torch.cos(half)
    q[:, 3] = torch.sin(half)
    return q


def quat_multiply_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    out = torch.empty_like(q1)
    out[..., 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    out[..., 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    out[..., 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    out[..., 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return out


def flatten_foot_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.reshape(x.shape[0], -1)
    return x


def contact_binary_and_force_mag(contact_force_xyz: torch.Tensor, threshold_n: float) -> tuple[torch.Tensor, torch.Tensor]:
    if contact_force_xyz.ndim == 2:
        force_mag = contact_force_xyz.abs()
    else:
        force_mag = safe_norm(contact_force_xyz, dim=-1)
    binary = (force_mag > threshold_n).float()
    return binary, force_mag


def phase_features_from_groups(group_a_phase: torch.Tensor, group_b_phase: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.sin(2.0 * torch.pi * group_a_phase),
            torch.cos(2.0 * torch.pi * group_a_phase),
            torch.sin(2.0 * torch.pi * group_b_phase),
            torch.cos(2.0 * torch.pi * group_b_phase),
        ],
        dim=-1,
    )


def forward_speed_from_body_velocity(root_lin_vel_b: torch.Tensor, axis: str = "x") -> torch.Tensor:
    axis = axis.lower()
    idx = 0 if axis == "x" else 1
    return root_lin_vel_b[:, idx]


def build_policy_obs(
    *,
    base_ang_vel_b: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    q_baseline: torch.Tensor,
    prev_action: torch.Tensor,
    v_cmd: torch.Tensor,
    heading_error: torch.Tensor,
    gait_param_state: torch.Tensor,
    contact_binary: torch.Tensor,
    contact_force_mag: torch.Tensor,
    toe_pos_b: torch.Tensor,
    toe_vel_b: torch.Tensor,
    phase_features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build a compact residual-policy observation.

    Main changes relative to the older version:
    - include root linear velocity so the policy can directly react to bounce / fore-aft slip;
    - include heading error relative to the reset heading so straight-line recovery is observable;
    - include the current gait-parameter state so the policy is not blind to the slow-varying
      touchdown-buffer / amplitude adaptations it has already applied.
    """
    pieces = [
        ensure_float(base_ang_vel_b),
        ensure_float(root_lin_vel_b),
        ensure_float(projected_gravity_b),
        ensure_float(joint_pos - q_baseline),
        ensure_float(joint_vel),
        ensure_float(contact_binary),
        ensure_float(flatten_foot_tensor(toe_pos_b)),
        ensure_float(phase_features),
        ensure_float(prev_action),
        ensure_float(ensure_2d(v_cmd)),
        ensure_float(ensure_2d(heading_error)),
        ensure_float(gait_param_state),
    ]
    policy = torch.cat(pieces, dim=-1)
    return {"policy": policy}
