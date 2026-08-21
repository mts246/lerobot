#!/usr/bin/env python3
"""Convert an ARX5/X5 LeRobot v3 dataset to DreamZero's LeRobot v2/GEAR layout.

The source dataset is treated as read-only. The converter creates one parquet and
one video per episode in a separate output directory, then writes the GEAR metadata
required by DreamZero.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd


DEFAULT_SOURCE = Path("/home/user/x5/outputs/x5_keyboard_20260817_201634")
DEFAULT_OUTPUT = Path("/home/user/x5/arx5_train_data")
DATA_PATTERN = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATTERN = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)

STATE_MAPPING = {
    "eef_position": (21, 24),
    "eef_rotation": (24, 27),
    "gripper_position": (27, 28),
}
ACTION_MAPPING = {
    "eef_position": (0, 3),
    "eef_rotation": (3, 6),
    "gripper_position": (6, 7),
}
RELATIVE_ACTION_KEYS = tuple(ACTION_MAPPING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embodiment-tag", default="x5")
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing non-empty output directory. Never changes --source.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=4, ensure_ascii=False)
        file.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_output(source: Path, output: Path, force: bool) -> None:
    source = source.resolve()
    output = output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("--output must be separate from --source and may not contain it")
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(f"Output is not empty: {output}. Use --force to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "meta").mkdir()


def load_episode_table(source: Path) -> pd.DataFrame:
    paths = sorted((source / "meta" / "episodes").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError("No LeRobot v3 episode metadata found under meta/episodes/")
    table = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    required = {
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Episode metadata is missing columns: {sorted(missing)}")
    return table.sort_values("episode_index").reset_index(drop=True)


def source_data_path(source: Path, episode: pd.Series) -> Path:
    return source / (
        f"data/chunk-{int(episode['data/chunk_index']):03d}/"
        f"file-{int(episode['data/file_index']):03d}.parquet"
    )


def source_video_path(source: Path, episode: pd.Series, video_key: str) -> Path:
    chunk_col = f"videos/{video_key}/chunk_index"
    file_col = f"videos/{video_key}/file_index"
    if chunk_col not in episode or file_col not in episode:
        raise ValueError(f"Episode metadata has no file mapping for {video_key}")
    return source / (
        f"videos/{video_key}/chunk-{int(episode[chunk_col]):03d}/"
        f"file-{int(episode[file_col]):03d}.mp4"
    )


def output_data_path(output: Path, episode_index: int, chunk_size: int) -> Path:
    return output / DATA_PATTERN.format(
        episode_chunk=episode_index // chunk_size, episode_index=episode_index
    )


def output_video_path(
    output: Path, episode_index: int, chunk_size: int, video_key: str
) -> Path:
    return output / VIDEO_PATTERN.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
        video_key=video_key,
    )


def split_parquet_files(
    source: Path, output: Path, episodes: pd.DataFrame, chunk_size: int
) -> list[Path]:
    cache: dict[Path, pd.DataFrame] = {}
    output_paths = []
    for _, episode in episodes.iterrows():
        episode_index = int(episode["episode_index"])
        src = source_data_path(source, episode)
        if not src.exists():
            raise FileNotFoundError(src)
        if src not in cache:
            cache[src] = pd.read_parquet(src)
        source_df = cache[src]
        episode_df = source_df.loc[source_df["episode_index"] == episode_index].copy()
        expected_length = int(episode["length"])
        if len(episode_df) != expected_length:
            raise ValueError(
                f"Episode {episode_index}: metadata length={expected_length}, parquet rows={len(episode_df)}"
            )
        target = output_data_path(output, episode_index, chunk_size)
        target.parent.mkdir(parents=True, exist_ok=True)
        source_episode_ids = source_df["episode_index"].unique()
        if len(source_episode_ids) == 1 and int(source_episode_ids[0]) == episode_index:
            shutil.copy2(src, target)
        else:
            episode_df.to_parquet(target, index=False)
        output_paths.append(target)
    return output_paths


def video_frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return int(stream.frames)
        return sum(1 for _ in container.decode(video=0))


def extract_video_segment(
    source_path: Path,
    target_path: Path,
    start_seconds: float,
    frame_count: int,
    fps: float,
) -> None:
    """Decode a timestamped segment and re-encode exactly frame_count frames."""
    with av.open(str(source_path)) as input_container, av.open(str(target_path), "w") as output_container:
        input_stream = input_container.streams.video[0]
        output_stream = output_container.add_stream("libx264", rate=Fraction(str(fps)))
        output_stream.width = input_stream.codec_context.width
        output_stream.height = input_stream.codec_context.height
        output_stream.pix_fmt = "yuv420p"

        written = 0
        epsilon = 0.5 / fps
        for frame in input_container.decode(video=0):
            timestamp = float(frame.time or 0.0)
            if timestamp + epsilon < start_seconds:
                continue
            frame.pts = written
            frame.time_base = Fraction(1, round(fps))
            for packet in output_stream.encode(frame):
                output_container.mux(packet)
            written += 1
            if written == frame_count:
                break
        for packet in output_stream.encode():
            output_container.mux(packet)
    if written != frame_count:
        raise ValueError(
            f"Could only extract {written}/{frame_count} frames from {source_path} at {start_seconds}s"
        )


def split_video_files(
    source: Path,
    output: Path,
    episodes: pd.DataFrame,
    video_keys: list[str],
    chunk_size: int,
    fps: float,
) -> None:
    use_count: dict[tuple[str, Path], int] = defaultdict(int)
    for _, episode in episodes.iterrows():
        for key in video_keys:
            use_count[(key, source_video_path(source, episode, key))] += 1

    for _, episode in episodes.iterrows():
        episode_index = int(episode["episode_index"])
        expected_frames = int(episode["length"])
        for key in video_keys:
            src = source_video_path(source, episode, key)
            if not src.exists():
                raise FileNotFoundError(src)
            target = output_video_path(output, episode_index, chunk_size, key)
            target.parent.mkdir(parents=True, exist_ok=True)
            start_col = f"videos/{key}/from_timestamp"
            start = float(episode[start_col]) if start_col in episode else 0.0
            if use_count[(key, src)] == 1 and video_frame_count(src) == expected_frames:
                shutil.copy2(src, target)
            else:
                extract_video_segment(src, target, start, expected_frames, fps)


def normalize_array_column(series: pd.Series) -> np.ndarray:
    array = np.stack(series.to_numpy()).astype(np.float64)
    return array.reshape(-1, 1) if array.ndim == 1 else array


def calculate_stats(parquet_paths: list[Path]) -> dict[str, dict[str, list[float]]]:
    frames = [pd.read_parquet(path, columns=["observation.state", "action", "timestamp"]) for path in parquet_paths]
    stats: dict[str, dict[str, list[float]]] = {}
    for column in ("observation.state", "action", "timestamp"):
        array = np.concatenate([normalize_array_column(frame[column]) for frame in frames], axis=0)
        stats[column] = {
            "mean": np.mean(array, axis=0).tolist(),
            "std": np.std(array, axis=0).tolist(),
            "min": np.min(array, axis=0).tolist(),
            "max": np.max(array, axis=0).tolist(),
            "q01": np.quantile(array, 0.01, axis=0).tolist(),
            "q99": np.quantile(array, 0.99, axis=0).tolist(),
        }
    return stats


def calculate_relative_stats(
    parquet_paths: list[Path], action_horizon: int
) -> dict[str, dict[str, list[float]]]:
    samples: dict[str, list[np.ndarray]] = {key: [] for key in RELATIVE_ACTION_KEYS}
    for path in parquet_paths:
        frame = pd.read_parquet(path, columns=["observation.state", "action"])
        state = normalize_array_column(frame["observation.state"])
        action = normalize_array_column(frame["action"])
        usable = len(frame) - action_horizon
        for index in range(max(usable, 0)):
            for key in RELATIVE_ACTION_KEYS:
                state_start, state_end = STATE_MAPPING[key]
                action_start, action_end = ACTION_MAPPING[key]
                samples[key].append(
                    action[index : index + action_horizon, action_start:action_end]
                    - state[index, state_start:state_end]
                )

    result: dict[str, dict[str, list[float]]] = {}
    for key, chunks in samples.items():
        if not chunks:
            raise ValueError(
                f"No relative samples for {key}; episodes must be longer than action horizon {action_horizon}"
            )
        array = np.concatenate(chunks, axis=0)
        result[key] = {
            "mean": np.mean(array, axis=0).tolist(),
            "std": np.std(array, axis=0).tolist(),
            "min": np.min(array, axis=0).tolist(),
            "max": np.max(array, axis=0).tolist(),
            "q01": np.quantile(array, 0.01, axis=0).tolist(),
            "q99": np.quantile(array, 0.99, axis=0).tolist(),
        }
    return result


def task_mapping(episodes: pd.DataFrame, parquet_paths: list[Path]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    path_by_episode = {
        int(pd.read_parquet(path, columns=["episode_index"])["episode_index"].iloc[0]): path
        for path in parquet_paths
    }
    for _, episode in episodes.iterrows():
        episode_index = int(episode["episode_index"])
        tasks_value = episode["tasks"]
        tasks = [str(value) for value in np.atleast_1d(tasks_value).tolist() if str(value)]
        frame = pd.read_parquet(path_by_episode[episode_index], columns=["task_index"])
        indices = [int(value) for value in frame["task_index"].drop_duplicates().tolist()]
        if len(tasks) == len(indices):
            pairs = zip(indices, tasks, strict=True)
        elif len(tasks) == 1:
            pairs = ((index, tasks[0]) for index in indices)
        else:
            raise ValueError(
                f"Episode {episode_index}: cannot match task indices {indices} to task texts {tasks}"
            )
        for index, text in pairs:
            if index in mapping and mapping[index] != text:
                raise ValueError(f"Task index {index} maps to conflicting texts")
            mapping[index] = text
    return mapping


def modality_metadata(video_keys: list[str]) -> dict[str, Any]:
    def field(original_key: str, start: int, end: int) -> dict[str, Any]:
        return {
            "original_key": original_key,
            "start": start,
            "end": end,
            "rotation_type": None,
            "absolute": True,
            "dtype": "float32",
            "range": None,
        }

    return {
        "state": {
            key: field("observation.state", start, end)
            for key, (start, end) in STATE_MAPPING.items()
        },
        "action": {
            key: field("action", start, end)
            for key, (start, end) in ACTION_MAPPING.items()
        },
        "video": {
            key.removeprefix("observation.images."): {"original_key": key}
            for key in video_keys
        },
        "annotation": {"task": {"original_key": "task_index"}},
    }


def validate_output(output: Path, info: dict[str, Any], action_horizon: int) -> None:
    episodes = [json.loads(line) for line in (output / "meta" / "episodes.jsonl").read_text().splitlines()]
    modality = read_json(output / "meta" / "modality.json")
    relative = read_json(output / "meta" / "relative_stats_dreamzero.json")
    tasks = [json.loads(line) for line in (output / "meta" / "tasks.jsonl").read_text().splitlines()]
    if not episodes or not tasks:
        raise ValueError("Converted episodes/tasks metadata is empty")
    if set(relative) != set(RELATIVE_ACTION_KEYS):
        raise ValueError("Relative statistics keys do not match action mapping")
    if len(modality["video"]) != len(
        [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]
    ):
        raise ValueError("Video modality count mismatch")

    for episode in episodes:
        index = int(episode["episode_index"])
        parquet = output / DATA_PATTERN.format(
            episode_chunk=index // int(info["chunks_size"]), episode_index=index
        )
        frame = pd.read_parquet(parquet)
        if len(frame) != int(episode["length"]):
            raise ValueError(f"Episode {index} parquet length mismatch")
        if len(frame) <= action_horizon:
            raise ValueError(f"Episode {index} is too short for action horizon {action_horizon}")
        for short_name, video_meta in modality["video"].items():
            video = output / VIDEO_PATTERN.format(
                episode_chunk=index // int(info["chunks_size"]),
                episode_index=index,
                video_key=video_meta["original_key"],
            )
            count = video_frame_count(video)
            if count != len(frame):
                raise ValueError(
                    f"Episode {index} video {short_name} has {count} frames, expected {len(frame)}"
                )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    info_path = source / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    source_info = read_json(info_path)
    if source_info.get("features", {}).get("observation.state", {}).get("shape") != [28]:
        raise ValueError("Expected X5 observation.state shape [28]")
    if source_info.get("features", {}).get("action", {}).get("shape") != [7]:
        raise ValueError("Expected X5 action shape [7]")

    prepare_output(source, output, args.force)
    try:
        episodes = load_episode_table(source)
        chunk_size = int(source_info.get("chunks_size", 1000))
        fps = float(source_info["fps"])
        video_keys = [
            key
            for key, feature in source_info["features"].items()
            if feature.get("dtype") == "video"
        ]
        parquet_paths = split_parquet_files(source, output, episodes, chunk_size)
        split_video_files(source, output, episodes, video_keys, chunk_size, fps)

        tasks_by_index = task_mapping(episodes, parquet_paths)
        task_rows = [
            {"task_index": index, "task": text}
            for index, text in sorted(tasks_by_index.items())
        ]
        episode_rows = [
            {
                "episode_index": int(row["episode_index"]),
                "tasks": [str(value) for value in np.atleast_1d(row["tasks"]).tolist()],
                "length": int(row["length"]),
            }
            for _, row in episodes.iterrows()
        ]

        info = dict(source_info)
        info["codebase_version"] = "v2.0-dreamzero-gear"
        info["source_codebase_version"] = source_info.get("codebase_version")
        info["robot_type"] = args.embodiment_tag
        info["data_path"] = DATA_PATTERN
        info["video_path"] = VIDEO_PATTERN

        write_json(output / "meta" / "info.json", info)
        write_json(output / "meta" / "modality.json", modality_metadata(video_keys))
        write_json(
            output / "meta" / "embodiment.json",
            {"robot_type": args.embodiment_tag, "embodiment_tag": args.embodiment_tag},
        )
        write_json(output / "meta" / "stats.json", calculate_stats(parquet_paths))
        write_json(
            output / "meta" / "relative_stats_dreamzero.json",
            calculate_relative_stats(parquet_paths, args.action_horizon),
        )
        write_jsonl(output / "meta" / "tasks.jsonl", task_rows)
        write_jsonl(output / "meta" / "episodes.jsonl", episode_rows)
        write_json(
            output / "meta" / "conversion.json",
            {
                "source": str(source),
                "source_format": source_info.get("codebase_version", "unknown"),
                "target_format": "DreamZero GEAR / LeRobot v2 layout",
                "action_horizon": args.action_horizon,
                "state_mapping": STATE_MAPPING,
                "action_mapping": ACTION_MAPPING,
            },
        )
        validate_output(output, info, args.action_horizon)
    except Exception:
        # A failed conversion should not leave a directory that looks train-ready.
        shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        raise

    print(f"Conversion complete: {output}")
    print(f"Episodes: {len(episodes)}, frames: {sum(row['length'] for row in episode_rows)}")
    print(f"Videos: {', '.join(key.removeprefix('observation.images.') for key in video_keys)}")
    print(f"Relative action keys: {', '.join(RELATIVE_ACTION_KEYS)}")


if __name__ == "__main__":
    main()
