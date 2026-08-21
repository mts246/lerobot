#!/bin/bash
set -euo pipefail
export HYDRA_FULL_ERROR=1

X5_DATA_ROOT=${X5_DATA_ROOT:-/home/user/x5/arx5_train_data}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/dreamzero_x5_lora}
NUM_GPUS=${NUM_GPUS:-1}
MAX_STEPS=${MAX_STEPS:-10}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-./checkpoints/Wan2.1-I2V-14B-480P}
TOKENIZER_DIR=${TOKENIZER_DIR:-./checkpoints/umt5-xxl}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-./checkpoints/DreamZero-AgiBot}

for required in \
    "$X5_DATA_ROOT/meta/modality.json" \
    "$X5_DATA_ROOT/meta/relative_stats_dreamzero.json" \
    "$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    "$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    "$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
    "$TOKENIZER_DIR" \
    "$PRETRAINED_MODEL_PATH"; do
    if [ ! -e "$required" ]; then
        echo "ERROR: required path is missing: $required" >&2
        exit 1
    fi
done

torchrun --nproc_per_node "$NUM_GPUS" --standalone \
    groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/x5_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=2 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json \
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size=1 \
    max_steps="$MAX_STEPS" \
    weight_decay=1e-5 \
    save_total_limit=3 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    x5_data_root="$X5_DATA_ROOT" \
    dit_version="$WAN_CKPT_DIR" \
    text_encoder_pretrained_path="$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
    tokenizer_path="$TOKENIZER_DIR" \
    pretrained_model_path="$PRETRAINED_MODEL_PATH" \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
