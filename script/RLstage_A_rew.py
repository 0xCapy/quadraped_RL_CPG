from __future__ import annotations

from typing import Dict

import torch

from RLstage_A_cfg import RLstage_A_RewardCfg


LEG_ORDER = ("LF", "RF", "LB", "RB")


def split_diagonal_contacts(contact_binary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pair_a = 0.5 * (contact_binary[:, 0] + contact_binary[:, 3])
    pair_b = 0.5 * (contact_binary[:, 1] + contact_binary[:, 2])
    return pair_a, pair_b


def diag_support_asymmetry(contact_binary: torch.Tensor) -> torch.Tensor:
    pair_a, pair_b = split_diagonal_contacts(contact_binary)
    return torch.abs(pair_a - pair_b)


def support_continuity(contact_binary: torch.Tensor, min_contacts: float = 2.0) -> torch.Tensor:
    contact_count = torch.sum(contact_binary, dim=-1)
    return torch.clamp((contact_count - (min_contacts - 1.0)) / 2.0, 0.0, 1.0)


def swing_mask_from_leg_phase(leg_phase: torch.Tensor, swing_ratio: float) -> torch.Tensor:
    return (leg_phase < swing_ratio).float()


def scuff_penalty(contact_binary: torch.Tensor, leg_phase: torch.Tensor, swing_ratio: float) -> torch.Tensor:
    swing_mask = swing_mask_from_leg_phase(leg_phase, swing_ratio)
    return torch.mean(contact_binary * swing_mask, dim=-1)


def reward_forward_track(v_fwd_body: torch.Tensor, v_cmd: torch.Tensor, sigma_mps: float) -> torch.Tensor:
    err = v_fwd_body - v_cmd.squeeze(-1)
    return torch.exp(-(err * err) / max(1.0e-6, sigma_mps * sigma_mps))


def reward_alive(v_fwd_body: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(v_fwd_body)


def compute_stage_a_reward(
    *,
    v_fwd_body: torch.Tensor,
    v_cmd: torch.Tensor,
    heading_error: torch.Tensor,
    yaw_rate: torch.Tensor,
    roll: torch.Tensor,
    pitch: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    action_residual: torch.Tensor,
    prev_action_residual: torch.Tensor,
    contact_binary: torch.Tensor,
    leg_phase: torch.Tensor,
    reward_cfg: RLstage_A_RewardCfg,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Residual reward with heading/stability first and contact quality second."""
    r_track = reward_forward_track(v_fwd_body, v_cmd, reward_cfg.vel_track_sigma_mps)
    r_alive = reward_alive(v_fwd_body)
    r_support = support_continuity(contact_binary, reward_cfg.support_continuity_min_contacts)

    # Important design change:
    # - p_yaw now means absolute heading error relative to the reset heading, not world yaw.
    # - p_rp is now a linear tilt magnitude proxy instead of roll^2 + pitch^2.
    # - p_bounce is now absolute vertical body velocity, which is easier for PPO to feel than z_vel^2.
    p_yaw = torch.abs(heading_error)
    p_yaw_rate = torch.abs(yaw_rate)
    p_rp = torch.sqrt(torch.clamp(roll.square() + pitch.square(), min=1.0e-12))
    p_bounce = torch.abs(root_lin_vel_b[:, 2])
    p_action_mag = torch.mean(action_residual.square(), dim=-1)
    p_action_rate = torch.mean((action_residual - prev_action_residual).square(), dim=-1)
    p_scuff = scuff_penalty(contact_binary, leg_phase, swing_ratio=reward_cfg.scuff_swing_ratio)
    p_diag = diag_support_asymmetry(contact_binary)

    reward = (
        reward_cfg.w_forward_track * r_track
        + reward_cfg.w_alive * r_alive
        + reward_cfg.w_support_continuity * r_support
        - reward_cfg.w_yaw_abs_pen * p_yaw
        - reward_cfg.w_yaw_rate_pen * p_yaw_rate
        - reward_cfg.w_roll_pitch_pen * p_rp
        - reward_cfg.w_vertical_bounce_pen * p_bounce
        - reward_cfg.w_action_mag_pen * p_action_mag
        - reward_cfg.w_action_rate_pen * p_action_rate
        - reward_cfg.w_scuff_pen * p_scuff
        - reward_cfg.w_diag_asym_pen * p_diag
    )

    terms = {
        'r_track': r_track,
        'r_alive': r_alive,
        'r_support': r_support,
        'p_yaw': p_yaw,
        'p_yaw_rate': p_yaw_rate,
        'p_rp': p_rp,
        'p_bounce': p_bounce,
        'p_action_mag': p_action_mag,
        'p_action_rate': p_action_rate,
        'p_scuff': p_scuff,
        'p_diag': p_diag,
    }
    return reward, terms
