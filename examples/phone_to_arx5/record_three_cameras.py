#!/usr/bin/env python

"""Record phone-teleoperated X5 demonstrations with three cameras.

This entry point reuses the X5 robot adapter, phone-to-X5 mapping, recording
controls, and dataset logic from ``record.py``.  Its only functional difference
is the camera layout: two external cameras named ``left`` and ``right``, plus
one wrist camera named ``wrist``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import record as base_record
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.scripts.lerobot_record import RecordConfig, record
from lerobot.teleoperators.phone import PhoneConfig
from lerobot.teleoperators.phone.config_phone import PhoneOS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record ARX X5-2025 demonstrations with an Android or iOS phone, "
            "two external cameras, and one wrist camera."
        )
    )
    parser.add_argument("--phone-os", choices=("android", "ios"), default="android")
    parser.add_argument("--can-interface", default="can1")
    parser.add_argument("--left-camera", default="")
    parser.add_argument("--right-camera", default="")
    parser.add_argument("--wrist-camera", default="")
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
    if args.fps <= 0 or args.num_episodes <= 0:
        raise ValueError("fps and num-episodes must be positive")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_id = args.repo_id or f"local/x5_phone_{stamp}"
    if args.push_to_hub and args.repo_id is None:
        raise ValueError("--repo-id=<hf_username>/<dataset_name> is required with --push-to-hub")
    dataset_root = args.root or base_record.WORKSPACE_ROOT / "outputs" / repo_id.split("/", 1)[-1]

    phone_os = PhoneOS(args.phone_os)
    robot_config = base_record.VendorX5FollowerConfig(
        id="x5_2025_follower",
        can_interface=args.can_interface,
        arm_type=2,
        cameras=_make_cameras(args),
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

    phone_to_x5 = base_record.PhoneToX5Cartesian(phone_os=phone_os, fps=args.fps)
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

    base_record.lerobot_record_module.record_loop = base_record._x5_record_loop
    base_record.lerobot_record_module.init_keyboard_listener = (
        base_record._init_x5_keyboard_listener
    )
    previous_signal_handlers = base_record._install_shutdown_signal_handlers()
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
        base_record.lerobot_record_module.record_loop = base_record._STANDARD_RECORD_LOOP
        base_record.lerobot_record_module.init_keyboard_listener = (
            base_record._STANDARD_KEYBOARD_LISTENER
        )
        base_record._restore_signal_handlers(previous_signal_handlers)


if __name__ == "__main__":
    main()
