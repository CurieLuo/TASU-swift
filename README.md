# TASU (PS-SLM) SWIFT 4.2.1 适配文档

> 本文档描述 TASU（PS-SLM）模型如何适配到 SWIFT (ms-swift) v4.2.1 框架，以及完整的训练 / 推理运行步骤。

---

## 目录

1. [架构概览](#1-架构概览)
2. [项目结构](#2-项目结构)
3. [关键适配问题](#3-关键适配问题)
4. [环境要求](#4-环境要求)
5. [训练](#5-训练)
6. [推理](#6-推理)
7. [WER 验证](#7-wer-验证)
8. [已知问题与解决方案](#8-已知问题与解决方案)
9. [配置参考](#9-配置参考)

---

## 1. 架构概览

```
音频输入 (wav / Kaldi ark)
    ↓
SenseVoiceSmall Encoder (frozen)
    ↓  encoder_out [B, T, 512]
EncoderProjectorLinearSiLU (trainable)
    ↓  projected [B, T', 1536]
Qwen2.5-1.5B-Instruct + LoRA (frozen backbone, LoRA adapter)
    ↓
文本输出
```

| 组件 | 模型 | 维度 | 训练状态 |
|------|------|------|----------|
| 语音编码器 | SenseVoiceSmall (funasr) | 输入 → `[T, 512]` | **Frozen** |
| 投影层 | `EncoderProjectorLinearSiLU` | `512 → 2048 → 1536` | **Trainable** (LoRA `modules_to_save`) |
| 大语言模型 | Qwen2.5-1.5B-Instruct | 1536-dim | **Frozen** + LoRA |

**关键设计**：
- 投影层使用 `LayerNorm → Linear → SiLU → Linear` 结构，将 SenseVoice 的 512-dim encoder 输出对齐到 Qwen 的 1536-dim embedding 空间。
- Baseline projector 的 `encoder_dim=512` 是 **encoder_out 维度**，不是 CTC posterior 的 25055 vocab size。因此 `ctc_posterior=False`。
- `<speech>` token 被添加为 tokenizer 的特殊 token（ID 151665），在 `_merge_input_ids_with_audio_features` 中被替换为实际的音频 embedding。

---

## 2. 项目结构

```
tasu_swift/
├── model/
│   ├── register.py           # 注册 tasu model_type 和 TASUTemplate
│   ├── swift_model.py        # 核心 TASUModel (forward / generate)
│   ├── tasu_pretrained.py    # PreTrainedModel wrapper (满足 SWIFT / peft 期望)
│   ├── tasu_template.py      # TASUTemplate: fbank 提取 + data collator
│   ├── projector.py          # EncoderProjectorLinearSiLU
│   ├── SenseVoice.py         # SenseVoice 相关工具
│   └── tokenizer.py          # SenseVoice tokenizer
├── scripts/
│   ├── tasu_plugin.py        # 核心插件: peft dummy + 模型/模板/数据集注册
│   ├── swift_sft_wrapper.py  # 训练包装器 (先注入 peft dummy 再 import SWIFT)
│   ├── swift_infer_wrapper.py# 推理包装器
│   ├── run_inference_manual.py # 手动推理脚本 (推荐，绕过 TransformersEngine bug)
│   ├── run_inference.py      # 标准推理入口
│   ├── inference_tasu.py     # 推理工具
│   └── fix_peft_compat.py    # peft 兼容性补丁
├── training/
│   ├── adapter.py            # 数据集注册 (asr/gr/s2tt/ser/slu)
│   └── data_prep.py          # 数据预处理
└── README_SWIFT.md           # 本文档
```

---

## 3. 关键适配问题

### 3.1 peft 0.6.0 兼容性

**问题**：容器内 peft 版本为 0.6.0，但 SWIFT 4.2.1 的 `swift/tuners/peft.py` 在模块导入时会引用 `BOFTConfig`、`VeraConfig`、`LoftQConfig` 等类，这些在 peft 0.6.0 中不存在，导致 `ImportError`。

**解决方案**：在 **任何 SWIFT 子模块导入之前**，通过 `tasu_plugin.py` 和 `swift_sft_wrapper.py` 向 `peft` 包注入空类（dummy）：

```python
# 必须在 import swift 之前执行
import peft
for _name in ["BOFTConfig", "BOFTModel", "LoftQConfig", "LoHaConfig",
              "LoKrConfig", "OFTConfig", "VeraConfig", "VeraModel"]:
    if not hasattr(peft, _name):
        _cls = type(_name, (), {})
        if _name in ("VeraModel", "BOFTModel"):
            _cls._create_and_replace = lambda *args, **kwargs: None
        setattr(peft, _name, _cls)
```

同时，`tasu_plugin.py` 中还会对 SWIFT 的 `LoraConfig.__init__` 打补丁，丢弃旧 peft 不支持的参数（`use_rslora`、`use_dora` 等）。

### 3.2 原始 PS-SLM JSONL 格式适配

**问题**：原始 PS-SLM 数据格式为 `{"key", "task", "target", "path"}`，而 SWIFT 期望 `{"messages", "audios"}`。SWIFT 的 `AutoPreprocessor` 在处理包含 `path` 字段的数据时会调用 `ResponsePreprocessor`，导致 `path` 被错误地 drop 掉。

**解决方案**：`tasu_plugin.py` 中 monkey-patch `AutoPreprocessor._get_preprocessor`，检测原始格式（存在 `task` + `path` + `target`）并返回 `PSLMPreprocessor`：

```python
class PSLMPreprocessor(RowPreprocessor):
    def preprocess(self, row):
        task = row.pop("task", None)
        target = row.pop("target", None)
        path = row.pop("path", None)
        row.pop("key", None)
        instruction = TASK_INSTRUCTIONS.get(task, task) if task else ""
        row["messages"] = [
            {"role": "user", "content": instruction + "<speech>"},
            {"role": "assistant", "content": target or ""},
        ]
        row["audios"] = [path] if path else []
        return row
```

这样用户**无需手动转换数据格式**，直接使用原始 JSONL 即可训练/推理。

### 3.3 SWIFT TransformersEngine 推理 Bug

**问题**：SWIFT 的 `TransformersEngine` 在调用 `model.generate()` 时，**不会传递 `input_features` 和 `input_feature_length`** 参数。这导致多模态模型在推理时接收不到音频特征，输出为空或错误内容。

**解决方案**：使用 `scripts/run_inference_manual.py` 手动推理脚本，绕过 `TransformersEngine`，直接调用 `model.generate()` 并传入音频特征：

```python
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    input_features=input_features,         # ← 关键参数
    input_feature_length=input_feature_length,  # ← 关键参数
    max_new_tokens=128,
    ...
)
```

### 3.4 Kaldi ark 文件读取

**问题**：`funasr` 的 `load_audio_text_image_video` 无法读取 Kaldi ark 格式的音频文件。

**解决方案**：`TASUTemplate._extract_audio()` 中自定义 ark 读取逻辑：

```python
if ".ark:" in audio_path or audio_path.endswith(".ark"):
    import kaldiio
    _, data = kaldiio.load_mat(audio_path)
    data = data.astype(np.float32) / 32768.0  # int16 → float32 [-1, 1]
    audio_list = [data]
```

### 3.5 `encoder_projector` 不被 LoRA 保存

**问题**：SWIFT 的 `get_modules_to_save` 默认不会将 `encoder_projector` 加入 `modules_to_save`，导致训练时投影层权重不会被保存到 checkpoint 中。

**解决方案**：`tasu_plugin.py` 中 monkey-patch `get_modules_to_save`：

```python
def _patched_get_modules_to_save(args, model, task_type=None):
    modules_to_save = _orig_get_modules_to_save(args, model, task_type)
    if hasattr(model, "tasu_model") and hasattr(model.tasu_model, "encoder_projector"):
        if "encoder_projector" not in modules_to_save:
            modules_to_save.append("encoder_projector")
    return modules_to_save
```

### 3.6 `get_multimodal_target_regex` 找不到目标模块

**问题**：SWIFT 的 `get_multimodal_target_regex` 需要 `model_meta.model_arch` 来查找 target modules。TASU 模型的 `model_arch` 未设置，导致返回 None，后续 LoRA 初始化失败。

**解决方案**：`tasu_plugin.py` 中 monkey-patch，当 `model_arch` 为 None 时 fallback 到 `find_all_linears(model)`：

```python
def _patched_get_multimodal_target_regex(model, *args, **kwargs):
    model_arch = getattr(getattr(model, 'model_meta', None), 'model_arch', None)
    if model_arch is None:
        return _tf_utils.find_all_linears(model)
    return _orig_get_multimodal_target_regex(model, *args, **kwargs)
```

---

## 4. 环境要求

- **硬件**：Ascend NPU（已验证 Atlas 800T A2）
- **容器镜像**：`hub.szaic.com/hpc/ai_asr-jingpeng-ps-slm:v2.0`
- **Python**：3.10
- **关键依赖**：

| 包 | 版本 | 说明 |
|----|------|------|
| `ms-swift` | 4.2.1 | 框架版本 |
| `peft` | 0.6.0 | 容器预装，**必须**配合 dummy 补丁使用 |
| `torch` | 2.1.0 | 配合 CANN |
| `torch_npu` | 匹配 CANN | NPU 后端 |
| `funasr` | 1.x | SenseVoice 依赖 |
| `kaldiio` | latest | Kaldi ark 读取 |
| `transformers` | 4.47+ | Qwen2.5 支持 |

**环境变量**：

```bash
# 基线 projector 权重路径（训练前加载）
export TASU_CKPT_PATH=/workspace/checkpoint/baseline/pytorch_model.bin

# 模型缓存路径
export SENSEVOICE_PATH=/workspace/model/SenseVoiceSmall
```

---

## 5. 训练

### 5.1 数据准备

训练数据路径：
- 训练集：`/workspace/data/asr/train/multitask.jsonl`（281,240 条）
- 测试集 clean：`/workspace/data/asr/test-clean/multitask.jsonl`（2,619 条）
- 测试集 other：`/workspace/data/asr/test-other/multitask.jsonl`（2,939 条）

数据格式（原始 PS-SLM JSONL，**无需转换**）：

```jsonl
{"key": "1089-134691-0000", "task": "asr", "target": "THE BIBLE IS A WRITER", "path": "/workspace/data/asr/train/data/raw.ark:123456"}
```

### 5.2 启动训练

**方式一：使用 wrapper 脚本（推荐，确保 peft dummy 先注入）**

```bash
export TASU_CKPT_PATH=/workspace/checkpoint/baseline/pytorch_model.bin

python /workspace/tasu_swift/scripts/swift_sft_wrapper.py \
  --external_plugins /workspace/tasu_swift/scripts/tasu_plugin.py \
  --model_type tasu \
  --model /workspace/model/Qwen2.5-1.5B-Instruct \
  --dataset /workspace/data/asr/train/multitask.jsonl \
  --output_dir /workspace/output/Qwen2.5-1.5B-Instruct/v12 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --lora_rank 64 \
  --lora_alpha 16 \
  --target_modules q_proj k_proj v_proj o_proj up_proj gate_proj down_proj \
  --attn_impl sdpa \
  --dataloader_num_workers 0 \
  --split_dataset_ratio 0.0 \
  --logging_steps 10 \
  --save_steps 5000 \
  --warmup_ratio 0.03 \
  --bf16 \
  --deepspeed default-zero2
```

**方式二：直接使用 `swift sft`（需确保 peft dummy 已通过其他方式注入）**

```bash
swift sft \
  --external_plugins /workspace/tasu_swift/scripts/tasu_plugin.py \
  --model_type tasu \
  --model /workspace/model/Qwen2.5-1.5B-Instruct \
  ...（同上参数）
```

### 5.3 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--model_type tasu` | 必须 | 使用注册的 TASU loader |
| `--external_plugins` | 必须 | 加载 `tasu_plugin.py`，注册模型/模板/数据集 |
| `--attn_impl sdpa` | 必须 | NPU 不支持 flash_attn，使用 sdpa |
| `--dataloader_num_workers 0` | 必须 | NPU 多进程 DataLoader 会 segfault |
| `--target_modules` | 推荐全部 linear | 指定 LoRA 目标模块 |
| `--split_dataset_ratio 0.0` | 推荐 | 不拆分验证集（使用全部数据训练） |
| `TASU_CKPT_PATH` | 环境变量 | 指向 baseline projector checkpoint |

### 5.4 训练过程观察

已验证的 1 epoch 训练过程（281k 样本）：

- 初始 loss: ~1.81
- 最终 loss: ~0.09
- 耗时: ~5 小时（单卡 NPU）
- Checkpoint 大小: ~280 MB（LoRA weights + projector）

Checkpoint 结构：

```
checkpoint-17577/
├── adapter_model.safetensors   # LoRA weights + encoder_projector
├── adapter_config.json         # LoRA 配置
├── training_args.json          # 训练参数
└── ...
```

---

## 6. 推理

### 6.1 手动推理脚本（推荐）

由于 SWIFT `TransformersEngine` 不传递 `input_features`，**必须使用** `run_inference_manual.py`：

```bash
python /workspace/tasu_swift/scripts/run_inference_manual.py
```

脚本内配置：

```python
adapter_path = '/workspace/output/Qwen2.5-1.5B-Instruct/v12-20260524-192632/checkpoint-17577'
data_path = '/workspace/data/asr/test-clean/multitask.jsonl'
out_path = '/tmp/inference_results_test_clean.jsonl'
```

**关键参数**：
- `max_new_tokens=128`：**必须**！默认 64 会导致长语音严重截断，WER 大幅恶化。
- `num_beams=1`：贪婪解码即可，beam search 提升有限但速度更慢。

输出格式：

```jsonl
{"key": "1089-134691-0000", "gt": "THE BIBLE IS A WRITER", "pred": "THE BIBLE IS A WRITER"}
```

### 6.2 计算 WER

```bash
python -c "
import json
from jiwer import wer

with open('/tmp/inference_results_test_clean.jsonl') as f:
    data = [json.loads(line) for line in f]

refs = [d['gt'] for d in data]
hyps = [d['pred'] for d in data]

print(f'Total: {len(data)}')
print(f'WER: {wer(refs, hyps)*100:.2f}%')
"
```

---

## 7. WER 验证

在 Librispeech test-clean 和 test-other 上的验证结果：

| 数据集 | 样本数 | Baseline WER | TASU-SWIFT WER | 提升 |
|--------|--------|-------------|----------------|------|
| test-clean | 2,619 | 4.44% | **3.54%** | ↓ 0.90% |
| test-other | 2,939 | 9.18% | **8.75%** | ↓ 0.43% |

**说明**：
- Baseline 为原始 PS-SLM 代码在相同数据和模型上的结果。
- 所有结果均在 `max_new_tokens=128` 条件下测得（64 会导致 test-other WER 上升至 ~12%）。

---

## 8. 已知问题与解决方案

| # | 问题 | 现象 | 解决方案 |
|---|------|------|----------|
| 1 | peft 0.6.0 不兼容 | `ImportError: cannot import name 'BOFTConfig'` | 使用 `tasu_plugin.py` 或 `swift_sft_wrapper.py` 注入 dummy |
| 2 | SWIFT 不传递 audio features | 推理输出为空或乱码 | 使用 `run_inference_manual.py` 直接调用 `model.generate()` |
| 3 | `max_new_tokens` 太小 | 长语音被截断，WER 显著升高 | 设置为 **128** 或更大 |
| 4 | DataLoader worker segfault | 训练启动时崩溃 | `--dataloader_num_workers 0` |
| 5 | `encoder_dim` 不匹配 | 加载 checkpoint 时 shape mismatch | 确保 `encoder_dim=512`（非 25055），`ctc_posterior=False` |
| 6 | NPU device 错误 | `npu_rms_norm` CPU backend 错误 | 确保模型在调用 `generate()` 前已在 NPU 上 |
| 7 | `encoder_projector` 未保存 | checkpoint 中缺少 projector | `tasu_plugin.py` 中 patch `get_modules_to_save` |
| 8 | `model_arch` 为 None | LoRA 初始化找不到 target modules | `tasu_plugin.py` 中 patch `get_multimodal_target_regex` |

---

## 9. 配置参考

### 9.1 `get_tasu_model()` 关键参数

```python
get_tasu_model(
    model_id="/workspace/model/Qwen2.5-1.5B-Instruct",
    encoder_dim=512,          # ← 必须 512（匹配 baseline projector）
    llm_dim=1536,             # Qwen2.5-1.5B hidden_size
    ctc_posterior=False,      # ← 必须 False（baseline 使用 encoder_out）
    ckpt_path=os.getenv("TASU_CKPT_PATH", ""),
)
```

### 9.2 LoRA 配置

```python
lora_rank = 64
lora_alpha = 16
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
modules_to_save = ["encoder_projector"]  # 投影层全量训练
```

### 9.3 模板 Prompt 格式

```
<|im_start|>user
{task_instruction}<speech><|im_end|>
<|im_start|>assistant
{target_text}<|im_end|>
```

其中 `<speech>` 是 tokenizer 的特殊 token（ID 151665），在 forward 时被替换为音频 embedding。

---

## 附录：音频特征提取流程

1. **输入**：音频文件路径（wav / mp3 / Kaldi ark）
2. **SenseVoice WavFrontend**（CPU 上运行）：
   - 采样率 16kHz
   - 80-dim fbank + LFR (Low Frame Rate) = **560-dim**
   - 输出 `input_features: [T, 560]`，`input_feature_length: scalar`
3. **Encoder**：SenseVoiceSmall → `encoder_out: [T', 512]`
4. **Projector**：`EncoderProjectorLinearSiLU` → `[T', 1536]`
5. **Embedding merge**：替换 `<speech>` token 的 embedding 为音频 embedding
6. **LLM**：Qwen2.5-1.5B-Instruct + LoRA → 自回归生成文本
