#!/usr/bin/env python

"""Record phone-teleoperated ARX X5 demonstrations with the vendor SDK.

This entry point intentionally uses the packaged ``ARX5_beta`` SDK instead of
the community ``lerobot_robot_arx5`` plugin.  The latter expects the separate
``arx5_interface`` extension, while this machine already has a working
``bimanual.SingleArm`` implementation for the X5-2025 arm.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


LEROBOT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = LEROBOT_ROOT.parent
ARX5_SDK_ROOT = WORKSPACE_ROOT / "ARX5_beta"
if str(ARX5_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(ARX5_SDK_ROOT))

from bimanual import SingleArm  # noqa: E402
from lerobot.cameras import CameraConfig  # noqa: E402
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.cameras.utils import make_cameras_from_configs  # noqa: E402
from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature  # noqa: E402
from lerobot.configs.dataset import DatasetRecordConfig  # noqa: E402
from lerobot.lerobot_types import RobotAction, RobotObservation, TransitionKey  # noqa: E402
from lerobot.processor import (  # noqa: E402
    RobotActionProcessorStep,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.config import RobotConfig  # noqa: E402
from lerobot.robots.robot import Robot  # noqa: E402
from lerobot.scripts import lerobot_record as lerobot_record_module  # noqa: E402
from lerobot.scripts.lerobot_record import RecordConfig, record  # noqa: E402
from lerobot.teleoperators.phone import PhoneConfig  # noqa: E402
from lerobot.teleoperators.phone.config_phone import PhoneOS  # noqa: E402
from lerobot.utils.keyboard_input import apply_recording_control, create_key_listener  # noqa: E402
from lerobot.utils.rotation import Rotation  # noqa: E402


LOGGER = logging.getLogger(__name__)

JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
EEF_KEYS = ("eef.x", "eef.y", "eef.z", "eef.roll", "eef.pitch", "eef.yaw", "eef.gripper")

_HOME_REQUEST = threading.Event()
_RECORDING_PHASE = threading.Event()
_HOME_EPOCH = 0
_STANDARD_RECORD_LOOP = lerobot_record_module.record_loop
_STANDARD_KEYBOARD_LISTENER = lerobot_record_module.init_keyboard_listener


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    horizontal = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    if horizontal > 1e-8:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        pitch = np.arctan2(-matrix[2, 0], horizontal)
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = np.arctan2(-matrix[1, 2], matrix[1, 1])
        pitch = np.arctan2(-matrix[2, 0], horizontal)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle, dtype=float) + np.pi) % (2 * np.pi) - np.pi


@RobotConfig.register_subclass("vendor_x5_follower")
@dataclass
class VendorX5FollowerConfig(RobotConfig):
    can_interface: str = "can1"
    arm_type: int = 2  # X5-2025 in the packaged vendor SDK.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    gripper_min: float = -3.4
    gripper_max: float = 0.1
    max_xyz_step_m: float = 0.01
    max_rpy_step_rad: float = 0.05
    max_gripper_step: float = 0.08
    shutdown_home_duration_s: float = 0.5
    shutdown_damping_seconds: float = 2.0


class VendorX5Follower(Robot):
    config_class = VendorX5FollowerConfig
    name = "vendor_x5_follower"

    def __init__(self, config: VendorX5FollowerConfig):
        super().__init__(config)
        self.config = config
        self.arm: SingleArm | None = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features: dict[str, type | tuple[int, int, int]] = {}
        for name in (*JOINT_NAMES, "gripper"):
            features[f"{name}.pos"] = float
            features[f"{name}.velocity"] = float
            features[f"{name}.effort"] = float
        features.update({key: float for key in EEF_KEYS})
        features.update(
            {
                name: (camera.height, camera.width, 3)
                for name, camera in self.config.cameras.items()
            }
        )
        return features

    @property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in EEF_KEYS}

    @property
    def is_connected(self) -> bool:
        # The arm is the safety-critical resource. Reporting it as connected even
        # if a camera failed/disconnected ensures LeRobot's finally block still
        # invokes our shutdown sequence.
        return self.arm is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self.arm is not None:
            raise RuntimeError("X5 is already connected")

        try:
            self.arm = SingleArm({"can_port": self.config.can_interface, "type": self.config.arm_type})
            for camera in self.cameras.values():
                camera.connect()
        except BaseException:
            # Includes Ctrl+C/SystemExit during camera initialization, after the
            # CAN arm has already been created.
            self._close_resources()
            raise

    def _require_arm(self) -> SingleArm:
        if self.arm is None:
            raise RuntimeError("X5 is not connected")
        return self.arm

    def _handle_home_request(self, arm: SingleArm) -> None:
        global _HOME_EPOCH
        if not _HOME_REQUEST.is_set():
            return

        _HOME_REQUEST.clear()
        print("Returning X5 to its home pose...")
        arm.go_home(duration=2.0, wait=True)
        _HOME_EPOCH += 1
        print("X5 home pose reached.")

    def get_observation(self) -> RobotObservation:
        arm = self._require_arm()
        self._handle_home_request(arm)
        joint_pos = np.asarray(arm.get_joint_positions(), dtype=float)
        joint_vel = np.asarray(arm.get_joint_velocities(), dtype=float)
        joint_effort = np.asarray(arm.get_joint_currents(), dtype=float)
        eef = np.asarray(arm.get_ee_pose_xyzrpy(), dtype=float)
        gripper = float(arm.get_gripper_pos())

        if joint_pos.shape[0] < 6 or joint_vel.shape[0] < 6 or joint_effort.shape[0] < 6:
            raise RuntimeError("X5 SDK returned fewer than six arm joints")

        observation: RobotObservation = {}
        for index, name in enumerate(JOINT_NAMES):
            observation[f"{name}.pos"] = float(joint_pos[index])
            observation[f"{name}.velocity"] = float(joint_vel[index])
            observation[f"{name}.effort"] = float(joint_effort[index])

        observation["gripper.pos"] = gripper
        observation["gripper.velocity"] = float(arm.get_gripper_vel())
        observation["gripper.effort"] = float(arm.get_gripper_current())
        for index, key in enumerate(EEF_KEYS[:6]):
            observation[key] = float(eef[index])
        observation["eef.gripper"] = gripper

        for name, camera in self.cameras.items():
            observation[name] = camera.async_read()
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        arm = self._require_arm()
        current_pose = np.asarray(arm.get_ee_pose_xyzrpy(), dtype=float)
        desired_pose = np.array([float(action[key]) for key in EEF_KEYS[:6]], dtype=float)

        safe_pose = current_pose.copy()
        safe_pose[:3] += np.clip(
            desired_pose[:3] - current_pose[:3],
            -self.config.max_xyz_step_m,
            self.config.max_xyz_step_m,
        )
        safe_pose[3:] = _wrap_angle(
            current_pose[3:]
            + np.clip(
                _wrap_angle(desired_pose[3:] - current_pose[3:]),
                -self.config.max_rpy_step_rad,
                self.config.max_rpy_step_rad,
            )
        )

        current_gripper = float(arm.get_gripper_pos())
        safe_gripper = float(
            np.clip(
                current_gripper
                + np.clip(
                    float(action["eef.gripper"]) - current_gripper,
                    -self.config.max_gripper_step,
                    self.config.max_gripper_step,
                ),
                self.config.gripper_min,
                self.config.gripper_max,
            )
        )

        arm.set_ee_pose_xyzrpy(safe_pose, duration=0)
        arm.set_gripper_pos(safe_gripper)
        if arm.fault is not None:
            raise RuntimeError(f"X5 safety fault: {arm.fault}")

        sent = {key: float(safe_pose[index]) for index, key in enumerate(EEF_KEYS[:6])}
        sent["eef.gripper"] = safe_gripper
        return sent

    def _close_resources(self) -> None:
        arm = self.arm
        # Clear first so cleanup is idempotent and a destructor cannot issue the
        # shutdown sequence twice while an exception is propagating.
        self.arm = None

        if arm is not None:
            if self.config.shutdown_home_duration_s > 0:
                print(
                    "X5 shutdown: returning home "
                    f"({self.config.shutdown_home_duration_s:.1f} seconds)..."
                )
                try:
                    accepted = arm.go_home(
                        duration=self.config.shutdown_home_duration_s,
                        wait=True,
                    )
                    if accepted is False:
                        LOGGER.error("X5 SDK rejected the shutdown home command")
                except BaseException:
                    # Never let a failed/interrupted home command prevent damping.
                    LOGGER.exception("Failed to return X5 home during shutdown")

            print(
                "X5 shutdown: entering damping protection "
                f"({self.config.shutdown_damping_seconds:.1f} seconds)..."
            )
            try:
                accepted = arm.damp(self.config.shutdown_damping_seconds)
                if accepted is False:
                    raise RuntimeError("X5 SDK rejected the damping command")
            except BaseException:
                LOGGER.exception("Failed to enable X5 damping; trying protection mode")
                try:
                    arm.protect_mode()
                except BaseException:
                    LOGGER.exception(
                        "X5 damping and protection mode both failed; cut arm power immediately"
                    )
            try:
                arm.close()
            except BaseException:
                LOGGER.exception("Failed to close X5 SDK")

        for camera in self.cameras.values():
            if camera.is_connected:
                try:
                    camera.disconnect()
                except BaseException:
                    LOGGER.exception("Failed to disconnect camera")

    def disconnect(self) -> None:
        self._close_resources()


@dataclass
class PhoneToX5Cartesian(RobotActionProcessorStep):
    phone_os: PhoneOS
    fps: int = 30
    position_scale: float = 0.5
    rotation_scale: float = 1.0
    max_translation_from_reference_m: float = 0.25
    max_rotation_from_reference_rad: float = 1.2
    gripper_speed_per_s: float = 1.5
    gripper_min: float = -3.4
    gripper_max: float = 0.1
    max_xyz_step_m: float = 0.01
    max_rpy_step_rad: float = 0.05
    max_gripper_step: float = 0.08
    _enabled_previous: bool = field(default=False, init=False, repr=False)
    _reference_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _held_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _gripper_command: float | None = field(default=None, init=False, repr=False)
    _last_home_epoch: int = field(default=0, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        if self._last_home_epoch != _HOME_EPOCH:
            self.reset()
            self._last_home_epoch = _HOME_EPOCH

        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Phone-to-X5 mapping requires the current robot observation")

        current_pose = np.array([float(observation[key]) for key in EEF_KEYS[:6]], dtype=float)
        current_gripper = float(observation["eef.gripper"])
        enabled = bool(action["phone.enabled"])

        if self._held_pose is None:
            self._held_pose = current_pose.copy()
        if self._gripper_command is None:
            self._gripper_command = current_gripper

        if enabled:
            if not self._enabled_previous or self._reference_pose is None:
                self._reference_pose = current_pose.copy()

            phone_position = np.asarray(action["phone.pos"], dtype=float)
            phone_rotation = action["phone.rot"]
            mapped_position = np.array(
                [-phone_position[1], phone_position[0], phone_position[2]], dtype=float
            )
            mapped_position *= self.position_scale
            mapped_position = np.clip(
                mapped_position,
                -self.max_translation_from_reference_m,
                self.max_translation_from_reference_m,
            )

            phone_rotvec = np.asarray(phone_rotation.as_rotvec(), dtype=float)
            mapped_rotvec = np.array(
                [phone_rotvec[1], phone_rotvec[0], -phone_rotvec[2]], dtype=float
            )
            mapped_rotvec *= self.rotation_scale
            rotation_norm = float(np.linalg.norm(mapped_rotvec))
            if rotation_norm > self.max_rotation_from_reference_rad:
                mapped_rotvec *= self.max_rotation_from_reference_rad / rotation_norm

            desired_pose = self._reference_pose.copy()
            desired_pose[:3] += mapped_position
            reference_rotation = _rpy_to_matrix(self._reference_pose[3:])
            delta_rotation = Rotation.from_rotvec(mapped_rotvec).as_matrix()
            desired_pose[3:] = _matrix_to_rpy(reference_rotation @ delta_rotation)
            self._held_pose = desired_pose

        raw_inputs: dict[str, Any] = action["phone.raw_inputs"]
        if self.phone_os == PhoneOS.IOS:
            gripper_velocity = float(raw_inputs.get("a3", 0.0))
        else:
            gripper_velocity = float(raw_inputs.get("reservedButtonA", 0.0)) - float(
                raw_inputs.get("reservedButtonB", 0.0)
            )
        self._gripper_command = float(
            np.clip(
                self._gripper_command + gripper_velocity * self.gripper_speed_per_s / self.fps,
                self.gripper_min,
                self.gripper_max,
            )
        )

        desired_pose = self._held_pose.copy()
        safe_pose = current_pose.copy()
        safe_pose[:3] += np.clip(
            desired_pose[:3] - current_pose[:3], -self.max_xyz_step_m, self.max_xyz_step_m
        )
        safe_pose[3:] = _wrap_angle(
            current_pose[3:]
            + np.clip(
                _wrap_angle(desired_pose[3:] - current_pose[3:]),
                -self.max_rpy_step_rad,
                self.max_rpy_step_rad,
            )
        )
        safe_gripper = float(
            current_gripper
            + np.clip(
                self._gripper_command - current_gripper,
                -self.max_gripper_step,
                self.max_gripper_step,
            )
        )

        self._enabled_previous = enabled
        mapped_action = {key: float(safe_pose[index]) for index, key in enumerate(EEF_KEYS[:6])}
        mapped_action["eef.gripper"] = safe_gripper
        return mapped_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION] = {
            key: PolicyFeature(type=FeatureType.ACTION, shape=(1,)) for key in EEF_KEYS
        }
        return features

    def reset(self) -> None:
        self._enabled_previous = False
        self._reference_pose = None
        self._held_pose = None
        self._gripper_command = None


def _x5_record_loop(*args, **kwargs):
    if kwargs.get("dataset") is None:
        _RECORDING_PHASE.clear()
    else:
        _RECORDING_PHASE.set()
    try:
        return _STANDARD_RECORD_LOOP(*args, **kwargs)
    finally:
        _RECORDING_PHASE.clear()


def _init_x5_keyboard_listener():
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    def on_key(name: str) -> None:
        key = name.lower()
        if key == "h":
            _HOME_REQUEST.set()
            if _RECORDING_PHASE.is_set():
                events["rerecord_episode"] = True
                events["exit_early"] = True
                print("H pressed: discarding this episode, then returning X5 home during reset.")
            else:
                print("H pressed: returning X5 home.")
        elif key in ("right", "n"):
            apply_recording_control("right", events)
        elif key in ("left", "r"):
            apply_recording_control("left", events)
        elif key in ("esc", "q"):
            apply_recording_control("esc", events)

    listener = create_key_listener(
        on_key,
        controls_help="H=home, Right/n=next, Left/r=re-record, Esc/q=quit",
    )
    return listener, events


def _install_shutdown_signal_handlers() -> dict[int, Any]:
    """Turn catchable termination signals into orderly Python unwinding.

    The first signal raises ``SystemExit``, causing LeRobot's ``finally`` block
    to disconnect the X5. Further termination signals are ignored while the
    home/damping sequence is in progress so a second Ctrl+C cannot interrupt it.
    """

    previous: dict[int, Any] = {}
    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    triggered = False

    def request_orderly_shutdown(signum, _frame) -> None:
        nonlocal triggered
        if triggered:
            return
        triggered = True
        for handled_signal in handled_signals:
            signal.signal(handled_signal, signal.SIG_IGN)
        print(f"\nReceived signal {signum}; starting X5 safe shutdown...")
        raise SystemExit(128 + signum)

    for handled_signal in handled_signals:
        previous[handled_signal] = signal.getsignal(handled_signal)
        signal.signal(handled_signal, request_orderly_shutdown)
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for handled_signal, handler in previous.items():
        signal.signal(handled_signal, handler)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record ARX X5-2025 demonstrations using an Android or iOS phone."
    )
    parser.add_argument("--phone-os", choices=("android", "ios"), default="android")
    parser.add_argument("--can-interface", default="can1")
    parser.add_argument("--front-camera", default="/dev/video10")
    parser.add_argument("--wrist-camera", default="/dev/video4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--episode-time-s", type=float, default=30.0)
    parser.add_argument("--reset-time-s", type=float, default=10.0)
    parser.add_argument("--task", default="Teleoperate the X5 arm with a phone")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--display-data", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.fps <= 0 or args.num_episodes <= 0:
        raise ValueError("fps and num-episodes must be positive")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_id = args.repo_id or f"local/x5_phone_{stamp}"
    if args.push_to_hub and args.repo_id is None:
        raise ValueError("--repo-id=<hf_username>/<dataset_name> is required with --push-to-hub")
    dataset_root = args.root or WORKSPACE_ROOT / "outputs" / repo_id.split("/", 1)[-1]

    cameras: dict[str, CameraConfig] = {}
    if args.front_camera:
        cameras["front"] = OpenCVCameraConfig(
            index_or_path=Path(args.front_camera), width=args.width, height=args.height, fps=args.fps
        )
    if args.wrist_camera:
        cameras["wrist"] = OpenCVCameraConfig(
            index_or_path=Path(args.wrist_camera), width=args.width, height=args.height, fps=args.fps
        )

    phone_os = PhoneOS(args.phone_os)
    robot_config = VendorX5FollowerConfig(
        id="x5_2025_follower",
        can_interface=args.can_interface,
        arm_type=2,
        cameras=cameras,
    )
    teleop_config = PhoneConfig(id=f"{args.phone_os}_phone", phone_os=phone_os)
    dataset_config = DatasetRecordConfig(
        repo_id=repo_id,
        root=dataset_root,
        fps=args.fps,
        num_episodes=args.num_episodes,
        episode_time_s=args.episode_time_s,
        reset_time_s=args.reset_time_s,
        single_task=args.task,
        video=True,
        push_to_hub=args.push_to_hub,
        no_stamp=True,
    )

    phone_to_x5 = PhoneToX5Cartesian(phone_os=phone_os, fps=args.fps)
    phone_action_pipeline = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[phone_to_x5],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    print(f"Dataset directory: {dataset_root}")
    print("Keep the X5 workspace clear and be ready to cut power before enabling phone motion.")
    print("Recording controls: H = home, Right Arrow = next, Left Arrow = redo, Esc = stop.")

    lerobot_record_module.record_loop = _x5_record_loop
    lerobot_record_module.init_keyboard_listener = _init_x5_keyboard_listener
    previous_signal_handlers = _install_shutdown_signal_handlers()
    try:
        record(
            RecordConfig(
                robot=robot_config,
                teleop=teleop_config,
                dataset=dataset_config,
                display_data=args.display_data,
                play_sounds=False,
            ),
            teleop_action_processor=phone_action_pipeline,
        )
    finally:
        lerobot_record_module.record_loop = _STANDARD_RECORD_LOOP
        lerobot_record_module.init_keyboard_listener = _STANDARD_KEYBOARD_LISTENER
        _restore_signal_handlers(previous_signal_handlers)


if __name__ == "__main__":
    main()
