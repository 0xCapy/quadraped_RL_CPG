from __future__ import annotations

"""Soft-ground Stage A environment.

This keeps the original Stage A residual-learning logic intact, but swaps the
flat rigid plane for a compliant-contact plane so the ground behaves like a
light spring-damper surface.

Design choice:
- We deliberately do NOT move to FEM/deformable terrain here.
- This file only changes the ground-contact model, so baseline and residual RL
  remain easy to compare against the rigid-ground version.
"""

from RLstage_soft_cfg import RLstage_Soft_Config
from RLstage_A_env import (
    RLstage_AEnv,
    RLstage_A_IsaacCfg,
    _STAGEA_RIGID_DISTAL_PATHS,
)

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor, FrameTransformer, Imu
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "RLstage_soft_env.py requires Isaac Lab to be available in the runtime environment."
    ) from exc


def _spawn_soft_ground_plane(stage_cfg: RLstage_Soft_Config) -> None:
    """Spawn a compliant-contact ground plane.

    The softness comes from PhysX compliant-contact material parameters on a
    rigid plane, not from a full deformable mesh.
    """
    try:
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=stage_cfg.soft_ground.static_friction,
            dynamic_friction=stage_cfg.soft_ground.dynamic_friction,
            restitution=stage_cfg.soft_ground.restitution,
            compliant_contact_stiffness=(
                stage_cfg.soft_ground.compliant_contact_stiffness
                if stage_cfg.soft_ground.enabled
                else 0.0
            ),
            compliant_contact_damping=(
                stage_cfg.soft_ground.compliant_contact_damping
                if stage_cfg.soft_ground.enabled
                else 0.0
            ),
        )
        gp = sim_utils.GroundPlaneCfg(
            size=(stage_cfg.soft_ground.size_m, stage_cfg.soft_ground.size_m),
            color=stage_cfg.soft_ground.color,
            physics_material=mat,
        )
        if hasattr(gp, "func"):
            gp.func("/World/GroundPlane", gp)
            return
    except Exception:
        pass

    # Conservative fallback: still spawn a visible plane so the user can run,
    # but this fallback may lose the compliant-contact behavior depending on the
    # local Isaac Sim version.
    try:  # pragma: no cover
        from omni.physx.scripts import physicsUtils

        physicsUtils.add_ground_plane(
            stage=None,
            planePath="/World/GroundPlane",
            axis="Z",
            size=stage_cfg.soft_ground.size_m,
            position=(0.0, 0.0, 0.0),
            color=stage_cfg.soft_ground.color,
        )
        return
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to spawn a soft ground plane."
        ) from exc


class RLstage_SoftEnv(RLstage_AEnv):
    """Stage A env with compliant soft ground."""

    cfg: RLstage_A_IsaacCfg

    def __init__(self, cfg: RLstage_A_IsaacCfg, render_mode: str | None = None, **kwargs):
        self.stage_cfg: RLstage_Soft_Config = cfg.stage_cfg
        super().__init__(cfg, render_mode=render_mode, **kwargs)

    def _setup_scene(self) -> None:
        _spawn_soft_ground_plane(self.stage_cfg)

        # Keep the original rigid-body distal frame workaround.
        try:
            rigid_paths = [
                _STAGEA_RIGID_DISTAL_PATHS["LF"],
                _STAGEA_RIGID_DISTAL_PATHS["RF"],
                _STAGEA_RIGID_DISTAL_PATHS["LB"],
                _STAGEA_RIGID_DISTAL_PATHS["RB"],
            ]
            for frame_cfg, rigid_path in zip(self.cfg.scene.frame_transformer.target_frames, rigid_paths):
                frame_cfg.prim_path = rigid_path
        except Exception:
            pass

        self.robot = Articulation(self.cfg.scene.robot)
        self.imu = Imu(self.cfg.scene.imu)
        self.contact_sensor = ContactSensor(self.cfg.scene.contact_sensor)
        self.frame_transformer = FrameTransformer(self.cfg.scene.frame_transformer)

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["imu"] = self.imu
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.scene.sensors["frame_transformer"] = self.frame_transformer

        self.scene.clone_environments(copy_from_source=False)
        if hasattr(self.scene, "filter_collisions"):
            self.scene.filter_collisions(global_prim_paths=["/World/GroundPlane"])


def make_soft_env_cfg(stage_cfg: RLstage_Soft_Config, num_envs: int | None = None) -> RLstage_A_IsaacCfg:
    """Build Isaac-side env cfg for the soft-ground variant."""
    cfg = RLstage_A_IsaacCfg()
    cfg.scene.num_envs = int(num_envs if num_envs is not None else stage_cfg.train.num_envs)
    cfg.scene.env_spacing = stage_cfg.train.env_spacing_m
    cfg.decimation = stage_cfg.action.decimation
    cfg.sim.dt = stage_cfg.train.sim_dt_s
    cfg.sim.render_interval = stage_cfg.action.decimation
    cfg.episode_length_s = stage_cfg.termination.episode_length_s
    cfg.observation_space = stage_cfg.obs.obs_dim_policy
    cfg.action_space = stage_cfg.action.dim
    cfg.state_space = 0
    cfg.stage_cfg = stage_cfg
    return cfg
