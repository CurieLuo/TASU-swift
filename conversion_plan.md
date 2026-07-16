# TASU (PS-SLM) → SWIFT Conversion Plan

## 1. Repository Overview & Architecture Summary

**Project**: TASU (Text-only Alignment for Speech Understanding) / PS-SLM  
**Source**: `workspace/ps-slm/Multitask/`  
**Target Framework**: SWIFT (ms-swift) v4.2.1  
**Base LLM**: Qwen2.5-1.5B-Instruct (config default is 7B, but readme specifies 1.5B)  
**Audio Encoder**: SenseVoiceSmall (from FunAudioLLM, via `funasr`)  
**Projector Type**: `linear-silu` (`EncoderProjectorLinearSiLU`)  

### 1.1 Core Architecture

```
Audio (wav/ark/flac) → SenseVoice Encoder → CTC Head
                                              ↓
                                         projector (linear-silu)
                                              ↓
                                     Qwen2.5 LLM → Text Output
```

- **Encoder**: SenseVoiceSmall conformer-based encoder. Outputs 560-dim features (after LFR). CTC vocab size = 25055.
- **Projector**: `EncoderProjectorLinearSiLU` — LayerNorm → Linear(560, 2048) → SiLU → Linear(2048, llm_dim). No downsampling (`k=1`).
- **LLM**: Qwen2.5-Instruct loaded via `transformers.AutoModelForCausalLM`.
- **Special Token**: `<speech>` added to LLM tokenizer.

### 1.2 Key Features

| Feature | Description | Flag |
|---------|-------------|------|
| CTC Posterior | Use CTC posterior probability to weight LLM embedding matrix | `ctc_posterior=true` |
| PSD | Phoneme-Synchronous Decoding — prune encoder frames using CTC alignment | `do_psd=true` |
| GT Embedding | Use ground-truth text embedding as auxiliary input | `gt_emb=true` |
| GT Embedding Noise | Add Gaussian noise to GT embedding for regularization | `gt_emb_noise=true` |
| Voca Trans | Use CTC posterior * LLM embedding matrix as projector output (LegoSLM baseline) | `voca_trans=false` |
| Top-1 Embedding | Use argmax CTC prediction to index embedding matrix | `top1_emb=false` |

---

## 2. Model Conversion Strategy

**Strategy B: Custom Model with Standard Backbone**

TASU is NOT in SWIFT's built-in supported model list. It uses:
- A **custom audio encoder** (SenseVoiceSmall from funasr)
- A **custom projector** (linear-silu with LayerNorm + SiLU)
- Custom forward logic (CTC posterior, PSD, GT embedding noise)
- A **standard LLM backbone** (Qwen2.5) that SWIFT supports natively

Therefore, the conversion approach is:
1. Use `@register_model(ModelMeta(...))` to register TASU as a custom model in SWIFT.
2. The custom `get_model` loader constructs the full model: encoder + projector + LLM.
3. Load the Qwen2.5 backbone via SWIFT's standard loader (`get_model_processor`).
4. Implement the custom forward pass including CTC posterior, PSD, and GT embedding logic.
5. Use SWIFT's `SftArguments` and `sft_main` for training.

---

## 3. Component Breakdown

### 3.1 Audio Encoder — SenseVoiceSmall

- **Source**: `model/SenseVoice.py` (adapted from FunAudioLLM/SenseVoice)
- **Implementation**: funasr-based conformer encoder with SANM attention.
- **Input**: 80-dim fbank (with LFR → 560-dim effective input)
- **Output**: 560-dim encoder features + CTC logits (25055 vocab)
- **Frozen during training**: `freeze_encoder=true` (default)

**SWIFT Integration**:
- Since SenseVoice is NOT a standard SWIFT audio model, load it manually inside the custom model loader.
- Use `funasr`'s model loading utilities or directly instantiate the encoder class.

### 3.2 Projector — EncoderProjectorLinearSiLU

- **Source**: `model/projector.py`
- **Architecture**:
  ```python
  nn.LayerNorm(encoder_dim)
  nn.Sequential(
      nn.Linear(encoder_dim, 2048, bias=True),
      nn.SiLU(),
      nn.Linear(2048, llm_dim, bias=True),
  )
  ```
- **Input**: 560-dim encoder features
- **Output**: llm_dim (1536 for Qwen2.5-1.5B, 4096 for 7B)
- **Downsampling rate**: `k=1` (no temporal downsampling for linear-silu)
- **Frozen during training**: `freeze_projector=false` (default)

**SWIFT Integration**:
- Implement as a custom `nn.Module` inside the registered model.
- Initialize with Kaiming uniform for first linear, zeros for last bias.

### 3.3 LLM Backbone — Qwen2.5-Instruct

- **Model Type**: `qwen2_5` (SWIFT-supported)
- **Loading**: Via `swift.get_model_processor` or `transformers.AutoModelForCausalLM`
- **Frozen during training**: `freeze_llm=false` (full fine-tuning) or `use_peft=true` (LoRA)
- **LoRA Config** (when `use_peft=true`):
  - `r=64`, `lora_alpha=16`
  - `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]`
  - `lora_dropout=0.05`, `bias="none"`

**SWIFT Integration**:
- Load via SWIFT's standard model loading path.
- The custom model wrapper will call `self.llm.get_input_embeddings()` to access the embedding matrix for CTC posterior computation.

---

## 4. Data Format & Preprocessing

### 4.1 Original Data Format

JSONL with fields:
```json
{"key": "sample_id", "task": "ASR", "target": "transcription text", "path": "/path/to/audio.wav", "GT": "optional ground truth for CTC"}
```

Supported tasks: `ASR`, `EN2ZH`, `EN2DE`, `QA`, `SLU_scenario`

### 4.2 Prompt Template

**System/Format**: Qwen2.5-Instruct chat format with `<speech>` token.

```
<|im_start|>user
{task_prompt}<speech><|im_end|>
<|im_start|>assistant
{target}
```

- `{task_prompt}` is looked up from `conf/multiprompt.jsonl` based on `task` field.
- `<speech>` is a special token added to the LLM tokenizer. It is replaced by audio embeddings during the forward pass.

**multiprompt.jsonl format**:
```json
{"task": "ASR", "prompt": "Transcribe the speech to text."}
{"task": "EN2ZH", "prompt": "Translate the English speech to Chinese."}
```

### 4.3 Audio Preprocessing

| Parameter | Value | Source |
|-----------|-------|--------|
| sample_rate | 16000 Hz | `funasr.load_audio` |
| num_mel_bins | 80 | `FbankConfig` |
| frame_length | 25 ms | `FbankConfig` |
| frame_shift | 10 ms | `FbankConfig` |
| window_type | hamming | `FbankConfig` |
| dither | 0.001 | `FbankConfig` |
| low_freq | 0 | `FbankConfig` |
| high_freq | 8000 | `FbankConfig` |
| htk_compat | True | `FbankConfig` |
| use_energy | False | `FbankConfig` |
| LFR | Applied by SenseVoice frontend | `funasr.extract_fbank` |
| normalize | False (for raw wav) | `DataConfig.normalize` |

**Preprocessing Pipeline**:
1. Load audio via `funasr.utils.load_utils.load_audio_text_image_video`
2. Extract fbank via `funasr.utils.load_utils.extract_fbank`
3. Apply LFR (Low Frame Rate) inside SenseVoice frontend
4. Pad or trim to fixed length if `pad_or_trim=true`

### 4.4 SWIFT Dataset Format

For SWIFT training, convert the JSONL to SWIFT-compatible format:
- `query`: The prompt with `<speech>` placeholder (e.g., `<|im_start|>user\nTranscribe the speech to text.<speech><|im_end|>\n<|im_start|>assistant\n`)
- `response`: The target text
- `audio`: Absolute path to audio file (or ark offset)

SWIFT v4.2.1 supports `audio` field in dataset. Use `DatasetMeta` with `media_type="audio"`.

---

## 5. Training Configuration Mapping

### 5.1 Original Training Config → SWIFT SftArguments

| Original Config | Value | SWIFT Equivalent | Notes |
|-----------------|-------|------------------|-------|
| optimizer | AdamW | `SftArguments.optim="adamw_torch"` | Default in SWIFT |
| lr | 1e-4 | `SftArguments.learning_rate=1e-4` | |
| weight_decay | 0.0 | `SftArguments.weight_decay=0.0` | |
| warmup_steps | 1000 | `SftArguments.warmup_steps=1000` | |
| num_epochs | 3~5 | `SftArguments.num_train_epochs=5` | |
| gradient_accumulation_steps | 1 | `SftArguments.gradient_accumulation_steps=1` | |
| batch_size_training | dynamic | `SftArguments.per_device_train_batch_size=1` | Use dynamic batching in custom dataset |
| mixed_precision | True | `SftArguments.bf16=True` | For Ascend NPU |
| deepspeed | zero2 | `SftArguments.deepspeed="scripts/ds_config.json"` | DeepSpeed ZeRO-2 |
| seed | 42 | `SftArguments.seed=42` | |

### 5.2 Model Behavior Flags → Training Script

These flags are NOT standard SWIFT arguments. They must be passed via custom `model_kwargs` or environment variables and handled in the custom model:

| Flag | Default | Description |
|------|---------|-------------|
| `do_psd` | true | Enable Phoneme-Synchronous Decoding |
| `ctc_posterior` | true | Use CTC posterior for projector output |
| `gt_emb` | true | Use GT text embedding |
| `gt_emb_noise` | true | Add noise to GT embedding |
| `voca_trans` | false | LegoSLM baseline mode |
| `top1_emb` | false | Use top-1 CTC prediction for embedding |
| `freeze_encoder` | true | Freeze SenseVoice encoder |
| `freeze_projector` | false | Freeze projector |
| `freeze_llm` | false | Freeze LLM (should be true when using PEFT) |
| `use_peft` | false | Use LoRA fine-tuning |

### 5.3 DeepSpeed Configuration

Original uses DeepSpeed ZeRO-2. SWIFT `SftArguments` supports `deepspeed` parameter.

**ds_config.json** (ZeRO-2, for Ascend NPU):
```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "allgather_partitions": true,
    "allgather_bucket_size": 5e7,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e7,
    "contiguous_gradients": true
  },
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0
}
```

### 5.4 Attention Configuration (NPU)

⚠️ **CRITICAL**: For Huawei Ascend NPU, SWIFT v4.2.1 uses `attn_impl` parameter.

- **Valid values**: `"sdpa"` (recommended), `"eager"`
- **DO NOT use**: `use_flash_attn` — this parameter does NOT exist in NPU environments and causes `TypeError`.
- **In shell script**: Use `--attn_impl sdpa` (space-separated, NOT `--attn_impl=sdpa`).

---

## 6. Critical Implementation Details

### 6.1 Inference Behavior

From `inference_batch.py` and `inference_batch_deepspeed.py`:

| Parameter | Value |
|-----------|-------|
| max_new_tokens | 200 |
| num_beams | 4 |
| do_sample | False |
| top_p | 1.0 |
| temperature | 1.0 |
| repetition_penalty | 1.0 |
| length_penalty | 1.0 |

**Generation call pattern**:
```python
outputs = model.llm.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    max_new_tokens=200,
    num_beams=4,
    do_sample=False,
    top_p=1.0,
    temperature=1.0,
    repetition_penalty=1.0,
    length_penalty=1.0,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
```

### 6.2 Prompt Template (Exact Format)

**Before tokenization**:
```
<|im_start|>user
{task_prompt}<speech><|im_end|>
<|im_start|>assistant
{target_text}
```

- `task_prompt` is looked up from `multiprompt.jsonl` based on the `task` field in data.
- `<speech>` is replaced by audio embeddings during forward pass. It must be a single token in the tokenizer.
- The LLM tokenizer is Qwen2.5's tokenizer with `<speech>` added as a special token.

### 6.3 Model Forward Pass Logic (CRITICAL)

From `model/ps-slm.py`, the forward pass has multiple branches based on flags:

**Branch A: `voca_trans=true` (LegoSLM baseline)**
```
projector_outs = projector(encoder_outs)
ctc_outs = softmax(projector_outs)
projector_outs = einsum("btv,vh->bth", ctc_outs, embed_matrix[:projector_outs.size(-1)])
if top1_emb:
    projector_outs = embed_matrix[ctc_outs.argmax(dim=-1)]
```

**Branch B: `ctc_posterior=true` (default)**
```python
if ctc_posterior:
    projector_outs = projector(encoder_outs)
    ctc_outs = softmax(projector_outs)
    llm_embedding = llm.get_input_embeddings()
    embed_matrix = llm_embedding.weight
    projector_outs = torch.einsum("btv,vh->bth", ctc_outs, embed_matrix[:projector_outs.size(-1)])
    if top1_emb:
        projector_outs = embed_matrix[ctc_outs.argmax(dim=-1)]
```

**Branch C: Raw feature (fallback)**
```python
if do_psd:
    encoder_outs, encoder_feature_length = psd(encoder_out, encoder_out_lens, ctc_posterior, blank_id)
else:
    encoder_outs, encoder_feature_length = encoder_out, encoder_out_lens
projector_outs = projector(encoder_outs)
```

**GT Embedding Branch** (`gt_emb=true`):
- When `gt_emb=true`, the model also takes ground-truth text embedding as input.
- `gt_emb_noise=true` adds Gaussian noise to the GT embedding.

### 6.4 PSD (Phoneme-Synchronous Decoding)

- Implemented in `model/ps-slm.py` as a method `psd()`.
- Uses CTC forced alignment to prune encoder output frames.
- Only keeps frames corresponding to non-blank CTC predictions.
- Reduces sequence length for LLM, improving efficiency.

### 6.5 Freezing Strategy

| Component | freeze flag | Default |
|-----------|-------------|---------|
| SenseVoice Encoder | `freeze_encoder` | true |
| Projector | `freeze_projector` | false |
| LLM Backbone | `freeze_llm` | false (or true if use_peft) |

**Implementation**: Use `param.requires_grad = False` for frozen components before passing to optimizer.

### 6.6 Checkpoint Format

- Original saves `pytorch_model.bin` (full model state dict).
- When `use_peft=true`, only LoRA + projector weights are trainable.
- SWIFT should save using `save_safetensors` (default in v4.2.1).

---

## 7. File Structure for Converted Code

```
workspace/tasu_swift/
  conversion_plan.md              # This file
  optimization_plan.md            # Generated by Planner Agent
  model/
    swift_model.py                # TASU model definition (encoder + projector + LLM)
    register.py                   # SWIFT @register_model registration
  training/
    adapter.py                    # Dataset registration & preprocessing
    data_prep.py                  # JSONL → SWIFT dataset conversion utilities
  scripts/
    swift.sh                      # Training launcher (swift sft command)
    ds_config.json                # DeepSpeed ZeRO-2 configuration
    infer_swift.py                # Converted model inference
    infer_original.py             # Original model inference wrapper
  validation_report.md            # Validation results
  execution_log.json              # Pipeline execution log
```

---

## 8. SWIFT v4.2.1 API Notes (Verified in Container)

### 8.1 Imports

```python
from swift.arguments.sft_args import SftArguments
from swift.model import register_model, ModelMeta
from swift import sft_main, get_model_processor, get_template
from swift.trainers import Seq2SeqTrainer
```

### 8.2 Model Registration

```python
from swift.model import register_model, ModelMeta

register_model(
    ModelMeta(
        model_type='tasu',
        model_groups=[],
        template='tasu',
        get_function=get_tasu_model,
    )
)
```

### 8.3 Template Registration

```python
from swift import get_template, register_template
from swift.template import TemplateMeta

register_template(
    TemplateMeta(
        template_type='tasu',
        prefix=['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'],
        prompt=['<|im_start|>user\n{{query}}<|im_end|>\n<|im_start|>assistant\n'],
        chat_sep=[''],
        suffix=['<|im_end|>'],
        system_prefix=['<|im_start|>system\n{{system}}<|im_end|>\n'],
    )
)
```

---

## 9. Risk & Pitfall Notes

1. **NPU Flash Attention**: Do NOT use `use_flash_attn`. Use `attn_impl="sdpa"` instead.
2. **funasr Dependency**: SenseVoice encoder depends on `funasr` package. Must ensure it's installed in container.
3. **Data Format**: Original uses custom JSONL with `path` (wav/ark/flac) and `task`-based prompts. SWIFT dataset adapter must handle ark offsets and multiprompt lookup.
4. **CTC Posterior Logic**: The projector output has vocab-size dimension (25055), then multiplied with LLM embedding matrix. This is non-standard and must be exactly reproduced.
5. **PSD Complexity**: PSD uses CTC forced alignment. Must verify `funasr` or `torchaudio` provides compatible CTC alignment.
6. **Dynamic Batching**: Original uses `MultiTaskDynamicBatchDataset` with `max_frame_length`. SWIFT's built-in batching may not support this directly; may need custom collator.
