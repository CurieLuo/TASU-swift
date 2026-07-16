#!/bin/bash
# ==============================================================================
# TASU (PS-SLM) 完整训练脚本 — 使用 swift_sft_wrapper.py
# 容器: c7ea55ec98f9
# 数据: /workspace/data/asr/train/multitask.jsonl (281,240 条)
# 模型: Qwen2.5-1.5B-Instruct + SenseVoiceSmall
# ==============================================================================
set -e

cd /workspace/tasu_swift

# 0. 环境变量
export TASU_CKPT_PATH=/workspace/checkpoint/baseline/pytorch_model.bin
export ASCEND_RT_VISIBLE_DEVICES=0

# 1. 输出目录（带时间戳避免覆盖）
OUTPUT_DIR=/workspace/output/Qwen2.5-1.5B-Instruct/v$(date +%Y%m%d-%H%M%S)
mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "TASU Training Start"
echo "Data:   /workspace/data/asr/train/multitask.jsonl"
echo "Model:  /workspace/model/Qwen2.5-1.5B-Instruct"
echo "Base:   ${TASU_CKPT_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================"

# 2. 启动训练
python /workspace/tasu_swift/scripts/swift_sft_wrapper.py \
  --external_plugins /workspace/tasu_swift/scripts/tasu_plugin.py \
  --model_type tasu \
  --model /workspace/model/Qwen2.5-1.5B-Instruct \
  --template tasu \
  --dataset /workspace/data/asr/train/multitask.jsonl \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.03 \
  --lora_rank 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --target_modules q_proj k_proj v_proj o_proj up_proj gate_proj down_proj \
  --attn_impl sdpa \
  --dataloader_num_workers 0 \
  --split_dataset_ratio 0.0 \
  --save_steps 5000 \
  --logging_steps 10 \
  --bf16 true \
  --seed 42 \
  2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "========================================"
echo "Training finished. Output: ${OUTPUT_DIR}"
echo "========================================"
