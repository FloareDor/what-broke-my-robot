"""Thin wrapper around the Go1 mujoco model, just exposes qpos/qvel/contacts/forces for the rest of the code."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import mujoco
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GO1_SCENE_PATH = REPO_ROOT / "third_party" / "mujoco_menagerie" / "unitree_go1" / "scene.xml"

# Actuator order as declared in go1.xml <actuator>, grouped per leg.
ACTUATOR_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]

FOOT_GEOM_NAMES = ["FR", "FL", "RR", "RL"]
TORSO_BODY_NAME = "trunk"


@dataclass
class StepRecord:
    """One snapshot of one timestep. Stack a bunch of these and you get a trajectory."""

    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    actuator_force: np.ndarray
    torso_pos: np.ndarray
    torso_quat: np.ndarray
    torso_linvel: np.ndarray
    foot_contacts: np.ndarray  # bool per foot geom, any contact this step


@dataclass
class Trajectory:
    records: list[StepRecord] = field(default_factory=list)

    def append(self, record: StepRecord) -> None:
        self.records.append(record)

    def to_dict(self) -> dict[str, np.ndarray]:
        if not self.records:
            return {}
        return {
            "time": np.array([r.time for r in self.records]),
            "qpos": np.stack([r.qpos for r in self.records]),
            "qvel": np.stack([r.qvel for r in self.records]),
            "ctrl": np.stack([r.ctrl for r in self.records]),
            "actuator_force": np.stack([r.actuator_force for r in self.records]),
            "torso_pos": np.stack([r.torso_pos for r in self.records]),
            "torso_quat": np.stack([r.torso_quat for r in self.records]),
            "torso_linvel": np.stack([r.torso_linvel for r in self.records]),
            "foot_contacts": np.stack([r.foot_contacts for r in self.records]),
        }


class Go1Env:
    """Simple deterministic env around the Go1 model, just reset() and step(), no reward or done."""

    def __init__(
        self,
        model_path: str | pathlib.Path = GO1_SCENE_PATH,
        control_dt: float = 0.02,
        model: mujoco.MjModel | None = None,
    ):
        self.model_path = str(model_path)
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.control_dt = control_dt
        self.control_substeps = max(1, round(control_dt / self.model.opt.timestep))

        self._home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self._actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATOR_NAMES]
        )
        self._foot_geom_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in FOOT_GEOM_NAMES]
        )
        self._torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY_NAME)

        self.home_qpos = self.model.key_qpos[self._home_key_id].copy()
        self.home_ctrl = self.model.key_ctrl[self._home_key_id].copy()

    def clone_model(self) -> mujoco.MjModel:
        """Fresh copy of the model, loaded from the xml again, so the nominal one never gets touched by accident."""
        return mujoco.MjModel.from_xml_path(self.model_path)

    def reset(self, seed: int | None = None, qpos_noise_std: float = 0.0) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.home_qpos
        self.data.ctrl[:] = self.home_ctrl
        if qpos_noise_std > 0.0:
            rng = np.random.default_rng(seed)
            self.data.qpos[7:] += rng.normal(0.0, qpos_noise_std, size=self.model.nq - 7)
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl: np.ndarray) -> StepRecord:
        """Send in a 12-dim ctrl command, advance one control step, and get a record back."""
        self.data.ctrl[self._actuator_ids] = ctrl
        for _ in range(self.control_substeps):
            mujoco.mj_step(self.model, self.data)
        return self._record()

    def _record(self) -> StepRecord:
        contacts = np.zeros(len(self._foot_geom_ids), dtype=bool)
        for i, geom_id in enumerate(self._foot_geom_ids):
            for c in range(self.data.ncon):
                contact = self.data.contact[c]
                if contact.geom1 == geom_id or contact.geom2 == geom_id:
                    contacts[i] = True
                    break
        return StepRecord(
            time=float(self.data.time),
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            ctrl=self.data.ctrl[self._actuator_ids].copy(),
            actuator_force=self.data.actuator_force[self._actuator_ids].copy(),
            torso_pos=self.data.xpos[self._torso_body_id].copy(),
            torso_quat=self.data.xquat[self._torso_body_id].copy(),
            torso_linvel=self.data.cvel[self._torso_body_id][3:6].copy(),
            foot_contacts=contacts,
        )

    @property
    def torso_height(self) -> float:
        return float(self.data.xpos[self._torso_body_id][2])
