# ARX5/X5 → DreamZero/GEAR conversion

This directory contains a non-destructive converter for the ARX5/X5 data produced by
the local LeRobot v3 collector. It converts the packed v3 files into the per-episode
LeRobot v2 layout consumed by DreamZero and generates all GEAR metadata.

The source directory is never edited. By default:

- source: `/home/user/x5/outputs/x5_keyboard_20260817_201634`
- output: `/home/user/x5/arx5_train_data`
- embodiment tag: `x5`
- action horizon: `24`

## Convert

Use the existing `lerobot` conda environment, which contains pandas, PyArrow, NumPy,
and PyAV:

```bash
cd /home/user/x5/lerobot
conda run -n lerobot python examples/arx5_data_convert/convert_arx5_to_dreamzero.py
```

For a later recording:

```bash
conda run -n lerobot python examples/arx5_data_convert/convert_arx5_to_dreamzero.py \
  --source /path/to/new_recording \
  --output /path/to/new_dreamzero_dataset
```

The converter refuses to replace a non-empty output directory unless `--force` is
given. `--force` applies only to the exact output directory and still never changes
the source.

## X5 modality mapping

| DreamZero key | Source column | Slice | Size |
|---|---|---:|---:|
| `state.eef_position` | `observation.state` | `[21:24]` | 3 |
| `state.eef_rotation` | `observation.state` | `[24:27]` | 3 |
| `state.gripper_position` | `observation.state` | `[27:28]` | 1 |
| `action.eef_position` | `action` | `[0:3]` | 3 |
| `action.eef_rotation` | `action` | `[3:6]` | 3 |
| `action.gripper_position` | `action` | `[6:7]` | 1 |
| `video.front` | `observation.images.front` | — | — |
| `video.wrist` | `observation.images.wrist` | — | — |
| `annotation.task` | `task_index` → `tasks.jsonl` | — | — |

The converter calculates relative statistics for all three action keys using a
24-step future action chunk and the current state as reference.

## Integrate the X5 config into DreamZero

The converter does not edit the DreamZero checkout. Before training, make these
small integration changes in `/home/user/x5/dreamzero`:

1. Add `X5 = "x5"` to `EmbodimentTag` in
   `groot/vla/data/schema/embodiment_tags.py`.
2. Insert the two YAML blocks from
   `dreamzero_config/x5_base_fragment.yaml` into
   `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`, before
   the global maps. Add the four `x5` entries noted at the bottom of the fragment
   to the existing `modality_configs`, `transforms`, `metadata_versions`, and `fps`
   maps.
3. Copy the data config and training script:

```bash
cd /home/user/x5/dreamzero
cp /home/user/x5/lerobot/examples/arx5_data_convert/dreamzero_config/x5_relative.yaml \
  groot/vla/configs/data/dreamzero/x5_relative.yaml
cp /home/user/x5/lerobot/examples/arx5_data_convert/dreamzero_config/x5_training.sh \
  scripts/train/x5_training.sh
chmod +x scripts/train/x5_training.sh
```

## Prepare weights and environment

Follow DreamZero's installation instructions first, then download the three required
weight sets from the DreamZero repository root:

```bash
hf download Wan-AI/Wan2.1-I2V-14B-480P \
  --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
hf download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl
hf download GEAR-Dreams/DreamZero-AgiBot --repo-type model \
  --local-dir ./checkpoints/DreamZero-AgiBot
```

Run a 10-step data/config smoke test first:

```bash
cd /home/user/x5/dreamzero
X5_DATA_ROOT=/home/user/x5/arx5_train_data NUM_GPUS=1 MAX_STEPS=10 \
  bash scripts/train/x5_training.sh
```

Only after the smoke test loads a batch and starts optimizing should `MAX_STEPS`,
GPU count, batch size, and checkpoint cadence be raised.

## Important data limitations of the current recording

The current converted dataset contains only one episode, 667 frames at 30 FPS
(22.23 seconds), and one generic instruction: “Teleoperate the X5 Cartesian pose
with a keyboard.” The gripper action is constant throughout the episode. This is
enough to test conversion and the training input pipeline, but not enough to learn a
usable manipulation policy.

Before real fine-tuning, collect substantially more successful demonstrations with:

- task-specific language such as “pick up the red block” rather than a teleoperation
  description;
- varied object positions, camera backgrounds, approach directions, and failure-free
  trajectories;
- active gripper open/close examples;
- consistent camera order and calibration (front first, wrist second);
- the exact same state/action units and Euler-angle convention that inference will use.

DreamZero's own guidance targets roughly 30 minutes of play data for few-shot robot
adaptation. Two views are supported by the supplied config (`num_views=2`), although
the AgiBot checkpoint was trained with three views, so a third stable external view
may improve transfer if available.
