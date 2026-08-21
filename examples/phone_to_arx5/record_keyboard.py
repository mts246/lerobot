#!/usr/bin/env python

"""Record ARX X5 demonstrations using Cartesian keyboard jog control.

The robot adapter, camera observations, episode lifecycle, and LeRobot dataset
writing are reused from ``record.py``.  Only the phone teleoperator is replaced:
held keys command Cartesian linear/angular velocity, matching the jog mode in
``Arx_phone_control/arx_phone_teleop``.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent
LEROBOT_ROOT = EXAMPLE_DIR.parents[1]
WORKSPACE_ROOT = LEROBOT_ROOT.parent
ARX_PHONE_TELEOP_ROOT = WORKSPACE_ROOT / "Arx_phone_control" / "arx_phone_teleop"

# ``record.py`` is an example-local module rather than an installed package.
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
if str(ARX_PHONE_TELEOP_ROOT) not in sys.path:
    sys.path.insert(0, str(ARX_PHONE_TELEOP_ROOT))

import record as phone_record  # noqa: E402
from teleop_core import TeleopLimits, TeleopMapper  # noqa: E402

from lerobot.cameras import CameraConfig  # noqa: E402
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature  # noqa: E402
from lerobot.configs.dataset import DatasetRecordConfig  # noqa: E402
from lerobot.lerobot_types import RobotAction, RobotObservation, TransitionKey  # noqa: E402
from lerobot.processor import (  # noqa: E402
    RobotActionProcessorStep,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.scripts.lerobot_record import RecordConfig, record  # noqa: E402
from lerobot.teleoperators.config import TeleoperatorConfig  # noqa: E402
from lerobot.teleoperators.teleoperator import Teleoperator  # noqa: E402
from lerobot.utils.keyboard_input import (  # noqa: E402
    TerminalKeyListener,
    apply_recording_control,
    keyboard as pynput_keyboard,
    pynput_can_capture,
    pynput_listener_is_trusted,
)


LOGGER = logging.getLogger(__name__)

LINEAR_KEYS = {
    "w": (0, 1.0),
    "s": (0, -1.0),
    "a": (1, 1.0),
    "d": (1, -1.0),
    "r": (2, 1.0),
    "f": (2, -1.0),
}
ANGULAR_KEYS = {
    "j": (0, -1.0),
    "l": (0, 1.0),
    "i": (1, -1.0),
    "k": (1, 1.0),
    "u": (2, 1.0),
    "o": (2, -1.0),
}
# This X5's observed hardware direction is X=open and Z=close.
GRIPPER_KEYS = {"x": 1.0, "z": -1.0}
MOVEMENT_KEYS = frozenset((*LINEAR_KEYS, *ANGULAR_KEYS, *GRIPPER_KEYS))

JOG_ACTION_KEYS = (
    "jog.linear.x",
    "jog.linear.y",
    "jog.linear.z",
    "jog.angular.roll",
    "jog.angular.pitch",
    "jog.angular.yaw",
    "jog.gripper",
)


@TeleoperatorConfig.register_subclass("x5_keyboard_cartesian")
@dataclass(kw_only=True)
class X5KeyboardCartesianTeleoperatorConfig(TeleoperatorConfig):
    """Configuration for the local Cartesian keyboard teleoperator."""

    terminal_pulse_s: float = 0.20


class X5KeyboardCartesianTeleoperator(Teleoperator):
    """Thread-safe held-key state exposed through LeRobot's Teleoperator API.

    On desktop X11/Windows/macOS, pynput supplies real press/release events.  A
    controlling terminal is supported as a fallback; because terminals have no
    release event, each movement key produces a short pulse and OS key-repeat
    refreshes that pulse while the key is held.
    """

    config_class = X5KeyboardCartesianTeleoperatorConfig
    name = "x5_keyboard_cartesian"

    def __init__(self, config: X5KeyboardCartesianTeleoperatorConfig):
        super().__init__(config)
        if config.terminal_pulse_s <= 0:
            raise ValueError("terminal_pulse_s must be positive")
        self.config = config
        self._lock = threading.RLock()
        self._pressed: set[str] = set()
        self._pulse_deadlines: dict[str, float] = {}
        self._command_times: dict[str, float] = {}
        self._recording_events: dict[str, bool] | None = None
        self._listener: Any = None
        self._connected = False
        self._terminal_mode = False

    @property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in JOG_ACTION_KEYS}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        global _ACTIVE_TELEOP

        if self._connected:
            raise RuntimeError("Keyboard teleoperator is already connected")

        listener = None
        if pynput_keyboard is not None and pynput_can_capture():
            listener = pynput_keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            listener.start()
            if not pynput_listener_is_trusted(listener):
                listener.stop()
                listener = None

        if listener is None and sys.stdin.isatty():
            self._terminal_mode = True
            listener = TerminalKeyListener(self._on_terminal_key)
            listener.start()

        if listener is None:
            raise RuntimeError(
                "No keyboard input backend is available. Run from an interactive terminal, "
                "or use an X11/Windows/macOS desktop with pynput installed."
            )

        self._listener = listener
        self._connected = True
        _ACTIVE_TELEOP = self
        if self._terminal_mode:
            LOGGER.warning(
                "Terminal keyboard fallback active: movement uses short pulses/key repeat; "
                "for true press/release control, run under an X11 desktop."
            )

    def attach_recording_events(self, events: dict[str, bool]) -> None:
        with self._lock:
            self._recording_events = events
            # Do not carry keys pressed while cameras/robot were still connecting
            # into the first control tick.
            self._pressed.clear()
            self._pulse_deadlines.clear()

    @staticmethod
    def _key_name(key: Any) -> str | None:
        char = getattr(key, "char", None)
        if char:
            return str(char).lower()
        name = getattr(key, "name", None)
        return str(name).lower() if name else None

    def _on_press(self, key: Any) -> None:
        name = self._key_name(key)
        if name is None:
            return
        with self._lock:
            first_press = name not in self._pressed
            self._pressed.add(name)
        if first_press:
            self._dispatch_command(name)

    def _on_release(self, key: Any) -> None:
        name = self._key_name(key)
        if name is None:
            return
        with self._lock:
            self._pressed.discard(name)

    def _on_terminal_key(self, name: str) -> None:
        name = name.lower()
        if name in MOVEMENT_KEYS:
            with self._lock:
                self._pulse_deadlines[name] = time.monotonic() + self.config.terminal_pulse_s
        else:
            self._dispatch_command(name)

    def _dispatch_command(self, name: str) -> None:
        events = self._recording_events
        if events is None:
            return

        # Suppress auto-repeat for commands while retaining it for terminal jog pulses.
        now = time.monotonic()
        last = self._command_times.get(name, float("-inf"))
        if now - last < 0.30:
            return
        self._command_times[name] = now

        if name == "h":
            phone_record._HOME_REQUEST.set()
            if phone_record._RECORDING_PHASE.is_set():
                events["rerecord_episode"] = True
                events["exit_early"] = True
                print("H pressed: discarding this episode, then returning X5 home during reset.")
            else:
                print("H pressed: returning X5 home.")
        elif name in ("right", "n"):
            apply_recording_control("right", events)
        elif name in ("left", "m"):
            apply_recording_control("left", events)
        elif name in ("esc", "q"):
            apply_recording_control("esc", events)

    @staticmethod
    def _axis_value(active: set[str], mapping: dict[str, tuple[int, float]], axis: int) -> float:
        value = sum(
            direction
            for key, (index, direction) in mapping.items()
            if index == axis and key in active
        )
        return float(np.clip(value, -1.0, 1.0))

    def get_action(self) -> RobotAction:
        if not self._connected:
            raise RuntimeError("Keyboard teleoperator is not connected")

        with self._lock:
            if self._terminal_mode:
                now = time.monotonic()
                expired = [key for key, deadline in self._pulse_deadlines.items() if deadline <= now]
                for key in expired:
                    del self._pulse_deadlines[key]
                active = set(self._pulse_deadlines)
            else:
                active = set(self._pressed)

        linear = [self._axis_value(active, LINEAR_KEYS, axis) for axis in range(3)]
        angular = [self._axis_value(active, ANGULAR_KEYS, axis) for axis in range(3)]
        gripper = float(
            np.clip(sum(value for key, value in GRIPPER_KEYS.items() if key in active), -1.0, 1.0)
        )
        return {
            "jog.linear.x": linear[0],
            "jog.linear.y": linear[1],
            "jog.linear.z": linear[2],
            "jog.angular.roll": angular[0],
            "jog.angular.pitch": angular[1],
            "jog.angular.yaw": angular[2],
            "jog.gripper": gripper,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    def disconnect(self) -> None:
        global _ACTIVE_TELEOP

        listener = self._listener
        self._listener = None
        self._connected = False
        with self._lock:
            self._pressed.clear()
            self._pulse_deadlines.clear()
            self._recording_events = None
        if listener is not None:
            with contextlib.suppress(Exception):
                listener.stop()
        if _ACTIVE_TELEOP is self:
            _ACTIVE_TELEOP = None


_ACTIVE_TELEOP: X5KeyboardCartesianTeleoperator | None = None


@dataclass
class KeyboardJogToX5Cartesian(RobotActionProcessorStep):
    """Integrate normalized keyboard axes into safe X5 Cartesian targets."""

    fps: int = 30
    max_linear_speed: float = 0.18
    max_angular_speed: float = math.radians(45.0)
    session_translation_limit: tuple[float, float, float] = (0.30, 0.30, 0.25)
    gripper_speed_per_s: float = 1.5
    gripper_min: float = -3.4
    gripper_max: float = 0.1
    max_xyz_step_m: float = 0.01
    max_rpy_step_rad: float = 0.05
    max_gripper_step: float = 0.08
    _mapper: TeleopMapper = field(init=False, repr=False)
    _gripper_target: float | None = field(default=None, init=False, repr=False)
    _gripper_active_previous: bool = field(default=False, init=False, repr=False)
    _active_previous: bool = field(default=False, init=False, repr=False)
    _last_update_time: float | None = field(default=None, init=False, repr=False)
    _last_home_epoch: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        positive_values = (
            self.fps,
            self.gripper_speed_per_s,
            self.max_xyz_step_m,
            self.max_rpy_step_rad,
            self.max_gripper_step,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive_values):
            raise ValueError("fps, speeds, and per-step safety limits must be positive")
        self._new_mapper()

    def _new_mapper(self) -> None:
        self._mapper = TeleopMapper(
            TeleopLimits(
                session_translation_limit=self.session_translation_limit,
                max_linear_speed=self.max_linear_speed,
                max_angular_speed=self.max_angular_speed,
            )
        )
        self._active_previous = False
        self._gripper_active_previous = False
        self._last_update_time = None

    def _dt(self, active: bool) -> float:
        now = time.monotonic()
        if not active or not self._active_previous or self._last_update_time is None:
            dt = 1.0 / self.fps
        else:
            dt = now - self._last_update_time
        self._last_update_time = now if active else None
        return dt

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Keyboard jog mapping requires the current robot observation")

        current_pose = np.array(
            [float(observation[key]) for key in phone_record.EEF_KEYS[:6]], dtype=float
        )
        current_gripper = float(observation["eef.gripper"])

        if self._last_home_epoch != phone_record._HOME_EPOCH:
            self._new_mapper()
            self._mapper.set_session_origin(current_pose)
            self._gripper_target = current_gripper
            self._last_home_epoch = phone_record._HOME_EPOCH
        elif self._mapper.session_origin is None:
            self._mapper.set_session_origin(current_pose)

        linear = tuple(float(action[key]) for key in JOG_ACTION_KEYS[:3])
        angular = tuple(float(action[key]) for key in JOG_ACTION_KEYS[3:6])
        gripper_axis = float(action["jog.gripper"])
        pose_active = any(abs(value) > 1e-9 for value in (*linear, *angular))
        dt = self._dt(pose_active or abs(gripper_axis) > 1e-9)

        if pose_active:
            if not self._active_previous:
                self._mapper.end(current_pose)
            desired_pose = self._mapper.update_jog(linear, angular, dt)
        else:
            self._mapper.end(current_pose)
            desired_pose = tuple(float(value) for value in current_pose)

        # Match VendorX5Follower.send_action's final safety clipping here so the
        # action saved in the dataset is also the action actually sent.
        desired_pose_array = np.asarray(desired_pose, dtype=float)
        safe_pose = current_pose.copy()
        safe_pose[:3] += np.clip(
            desired_pose_array[:3] - current_pose[:3],
            -self.max_xyz_step_m,
            self.max_xyz_step_m,
        )
        safe_pose[3:] = phone_record._wrap_angle(
            current_pose[3:]
            + np.clip(
                phone_record._wrap_angle(desired_pose_array[3:] - current_pose[3:]),
                -self.max_rpy_step_rad,
                self.max_rpy_step_rad,
            )
        )

        if self._gripper_target is None:
            self._gripper_target = current_gripper
        if abs(gripper_axis) > 1e-9:
            self._gripper_target = float(
                np.clip(
                    self._gripper_target + gripper_axis * self.gripper_speed_per_s * dt,
                    self.gripper_min,
                    self.gripper_max,
                )
            )
        elif self._gripper_active_previous:
            # Gripper keys command velocity, so releasing them means stop now.
            self._gripper_target = current_gripper
        safe_gripper = float(
            current_gripper
            + np.clip(
                self._gripper_target - current_gripper,
                -self.max_gripper_step,
                self.max_gripper_step,
            )
        )

        self._active_previous = pose_active
        self._gripper_active_previous = abs(gripper_axis) > 1e-9
        mapped_action = {
            key: float(safe_pose[index])
            for index, key in enumerate(phone_record.EEF_KEYS[:6])
        }
        mapped_action["eef.gripper"] = safe_gripper
        return mapped_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION] = {
            key: PolicyFeature(type=FeatureType.ACTION, shape=(1,))
            for key in phone_record.EEF_KEYS
        }
        return features

    def reset(self) -> None:
        self._new_mapper()
        self._gripper_target = None
        self._last_home_epoch = -1


def _init_shared_keyboard_listener():
    """Give LeRobot recording controls to the already-running teleop listener."""
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }
    if _ACTIVE_TELEOP is None or not _ACTIVE_TELEOP.is_connected:
        raise RuntimeError("Cartesian keyboard teleoperator is not connected")
    _ACTIVE_TELEOP.attach_recording_events(events)
    return None, events


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record ARX X5-2025 demonstrations using Cartesian keyboard jog control."
    )
    parser.add_argument("--can-interface", default="can1")
    parser.add_argument(
        "--left-camera",
        "--front-camera",
        dest="left_camera",
        default="/dev/video4",
        help="left external camera device (--front-camera is kept as a compatibility alias)",
    )
    parser.add_argument("--right-camera", default="/dev/video16")
    parser.add_argument("--wrist-camera", default="/dev/video10")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--episode-time-s", type=float, default=60.0)
    parser.add_argument("--reset-time-s", type=float, default=300)
    parser.add_argument("--task", default="Teleoperate the X5 Cartesian pose with a keyboard")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append new episodes to an existing dataset (requires --repo-id and --root)",
    )
    parser.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--display-data", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--linear-speed", type=float, default=0.18, help="maximum XYZ speed in m/s")
    parser.add_argument(
        "--angular-speed-deg", type=float, default=45.0, help="maximum RPY speed in deg/s"
    )
    parser.add_argument("--gripper-speed", type=float, default=1.5, help="gripper units per second")
    parser.add_argument(
        "--session-limit",
        type=float,
        nargs=3,
        default=(0.30, 0.30, 0.25),
        metavar=("X", "Y", "Z"),
        help="maximum XYZ displacement from the initial/home pose in metres",
    )
    parser.add_argument(
        "--terminal-pulse-s",
        type=float,
        default=0.20,
        help="movement pulse duration for terminal fallback (desktop press/release is preferred)",
    )
    return parser.parse_args()


def _make_cameras(args: argparse.Namespace) -> dict[str, CameraConfig]:
    cameras: dict[str, CameraConfig] = {}
    camera_paths = {
        "left": args.left_camera,
        "right": args.right_camera,
        "wrist": args.wrist_camera,
    }
    for name, device_path in camera_paths.items():
        if device_path:
            cameras[name] = OpenCVCameraConfig(
                index_or_path=Path(device_path),
                width=args.width,
                height=args.height,
                fps=args.fps,
            )
    return cameras


def main() -> None:
    args = _parse_args()
    positive_values = (
        args.fps,
        args.num_episodes,
        args.episode_time_s,
        args.linear_speed,
        args.angular_speed_deg,
        args.gripper_speed,
        args.terminal_pulse_s,
        *args.session_limit,
    )
    if not all(math.isfinite(value) and value > 0 for value in positive_values):
        raise ValueError("fps, counts, durations, speeds, and session limits must be positive")
    if args.reset_time_s < 0:
        raise ValueError("reset-time-s must be non-negative")
    if args.resume:
        if args.repo_id is None:
            raise ValueError("--repo-id is required with --resume")
        if args.root is None:
            raise ValueError("--root is required with --resume")
        if not (args.root / "meta" / "info.json").is_file():
            raise ValueError(f"--root does not contain an existing LeRobot dataset: {args.root}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_id = args.repo_id or f"local/x5_keyboard_{stamp}"
    if args.push_to_hub and args.repo_id is None:
        raise ValueError("--repo-id=<hf_username>/<dataset_name> is required with --push-to-hub")
    dataset_root = args.root or WORKSPACE_ROOT / "outputs" / repo_id.split("/", 1)[-1]

    robot_config = phone_record.VendorX5FollowerConfig(
        id="x5_2025_follower",
        can_interface=args.can_interface,
        arm_type=2,
        cameras=_make_cameras(args),
    )
    teleop_config = X5KeyboardCartesianTeleoperatorConfig(
        id="x5_cartesian_keyboard",
        terminal_pulse_s=args.terminal_pulse_s,
    )
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

    keyboard_to_x5 = KeyboardJogToX5Cartesian(
        fps=args.fps,
        max_linear_speed=args.linear_speed,
        max_angular_speed=math.radians(args.angular_speed_deg),
        session_translation_limit=tuple(args.session_limit),
        gripper_speed_per_s=args.gripper_speed,
    )
    action_pipeline = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[keyboard_to_x5],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    print(f"Dataset directory: {dataset_root}")
    print("Keep the X5 workspace clear and keep the emergency power cut-off within reach.")
    print("XYZ: W/S=+/-X, A/D=+/-Y, R/F=+/-Z")
    print("RPY: L/J=+/-roll, K/I=+/-pitch, U/O=+/-yaw; gripper: X=open, Z=close")
    print("Recording: H=home, Right/N=next, Left/M=re-record, Esc/Q=stop")

    phone_record.lerobot_record_module.record_loop = phone_record._x5_record_loop
    phone_record.lerobot_record_module.init_keyboard_listener = _init_shared_keyboard_listener
    previous_signal_handlers = phone_record._install_shutdown_signal_handlers()
    try:
        record(
            RecordConfig(
                robot=robot_config,
                teleop=teleop_config,
                dataset=dataset_config,
                display_data=args.display_data,
                play_sounds=False,
                resume=args.resume,
            ),
            teleop_action_processor=action_pipeline,
        )
    finally:
        phone_record.lerobot_record_module.record_loop = phone_record._STANDARD_RECORD_LOOP
        phone_record.lerobot_record_module.init_keyboard_listener = (
            phone_record._STANDARD_KEYBOARD_LISTENER
        )
        phone_record._restore_signal_handlers(previous_signal_handlers)


if __name__ == "__main__":
    main()
