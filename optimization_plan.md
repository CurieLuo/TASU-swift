# TASU (PS-SLM) SWIFT Implementation Blueprint

## 1. Executive Summary

- **Model type**: TASU (Text-only Alignment for Speech Understanding) — Strategy B (Custom Model with Standard Backbone)
- **Architecture**: SenseVoiceSmall Encoder → CTC Head → LinearSiLU Projector → Qwen2.5-1.5B LLM
- **Total files to create**: 8
- **Key risk areas**:
  1. CTC posterior branch logic (multiple conditional forward paths)
  2. PSD (Phoneme-Synchronous Decoding) exact frame merging behavior
  3. Prompt template with `<speech>` token placement
  4. Projector input dim = 25055 (CTC vocab) NOT 560 (encoder features)
  5. No `resize_token_embeddings` called — `<speech>` token ID is OOV for LLM but replaced by audio embeddings

## 2. File Structure

```
model/
  swift_model.py       # TASUModel class with exact forward pass
  register.py          # @register_model with ModelMeta
  projector.py         # EncoderProjectorLinearSiLU (extract from source)
training/
  adapter.py           # Dataset registration, multiprompt lookup, collator
  data_prep.py         # JSONL preprocessing utilities
scripts/
  swift.sh             # Training launcher with env vars
  ds_config.json       # DeepSpeed ZeRO-2 config for Ascend NPU
  infer_swift.py       # Inference script with exact generate() params
```

## 3. Component Implementation Details

### 3.1 Model Architecture (model/swift_model.py)

**Original behavior**: The model consists of three main components loaded separately and wrapped in `slam_model_asr`.

**SWIFT implementation**: Register a custom model that internally constructs encoder + projector + LLM. The LLM backbone is loaded via SWIFT's standard path; encoder and projector are loaded manually.

**Critical parameters:**
- `encoder_dim`: **25055** (SenseVoice CTC vocab size — this is the projector INPUT dimension when `ctc_posterior=true`)
- `projector_hidden_dim`: **2048**
- `projector_output_dim`: **1536** (Qwen2.5-1.5B hidden dim)
- `activation`: **SiLU**
- `dropout`: **0.0** (no dropout in projector)
- `ds_rate`: **1** (no temporal downsampling)
- `blank_id`: **0**

**Projector exact architecture**:
```python
class EncoderProjectorLinearSiLU(nn.Module):
    def __init__(self, config, bottleneck=2048):
        super().__init__()
        in_dim = config.encoder_dim   # MUST be 25055 for ctc_posterior mode
        out_dim = config.llm_dim      # 1536
        self.norm = nn.LayerNorm(in_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, bottleneck, bias=True),
            nn.SiLU(),
            nn.Linear(bottleneck, out_dim, bias=True),
        )
        nn.init.kaiming_uniform_(self.ffn[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.ffn[2].bias)
        self.k = 1

    def forward(self, x):
        x = self.norm(x)
        return self.ffn(x)
```

**Pitfall**: Setting `encoder_dim=560` (raw encoder output) instead of 25055 (CTC vocab). The checkpoint `encoder_projector.ffn.0.weight` has shape `[2048, 25055]`, confirming input dim is 25055.
**Prevention**: Always use `encoder_dim=25055` when `ctc_posterior=true`.

### 3.2 Forward Pass (model/swift_model.py)

**Original computation flow**:

1. `input_features` [B, T, 560] — from funasr extract_fbank + LFR
2. Prepend 4 query tokens to speech:
   - `language_query` = embed([[0]]) — language ID token
   - `event_emo_query` = embed([[1, 2]]) — event/emotion query tokens
   - `textnorm_query` = embed([[2]]) — text normalization query token
   - `speech` = cat([language_query, event_emo_query, textnorm_query, speech], dim=1)
   - `speech_lengths` = input_feature_length + 4
3. `raw_encoder_out, raw_encoder_out_lens = self.encoder.encoder(speech, speech_lengths)`
4. `raw_logits = self.encoder.ctc.ctc_lo(raw_encoder_out)`
5. `raw_ctc_posterior = F.softmax(raw_logits, dim=-1)`
6. Remove 4 query tokens from outputs:
   - `ctc_posterior = raw_ctc_posterior[:, 4:, :]`  [B, T, 25055]
   - `encoder_out = raw_encoder_out[:, 4:, :]`      [B, T, 560]
   - `encoder_out_lens = torch.clamp(raw_encoder_out_lens - 4, min=0)`

**Then branch based on flags**:

**Branch A**: `ctc_posterior=true` and `gt_emb=true` (training with ground-truth)
- Use `ctc_pseudo_posterior(texts)` to create one-hot GT embeddings
- Add Gaussian noise if `gt_emb_noise=true`

**Branch B**: `ctc_posterior=true`, `gt_emb=false`, `do_psd=true`
- `encoder_outs, encoder_feature_length = self.psd(encoder_out, encoder_out_lens, ctc_posterior, blank_id=0)`
- PSD merges adjacent identical non-blank frames, then filters blank frames with threshold 0.9

**Branch C**: `ctc_posterior=true`, `gt_emb=false`, `do_psd=false`
- `encoder_outs = ctc_posterior`  [B, T, 25055]
- `encoder_feature_length = encoder_out_lens`

**Branch D**: `ctc_posterior=true`, `voca_trans=true` (LegoSLM baseline)
- `projector_outs = projector(encoder_outs)`  [B, T, 1536]
- `ctc_outs = softmax(projector_outs, dim=-1)`
- `llm_embedding = llm.get_input_embeddings()`
- `embed_matrix = llm_embedding.weight`
- `projector_outs = torch.einsum("btv,vh->bth", ctc_outs, embed_matrix[:projector_outs.size(-1)])`
- If `top1_emb=true`: `projector_outs = embed_matrix[ctc_outs.argmax(dim=-1)]`

**Branch E**: `ctc_posterior=false`
- `encoder_outs = encoder_out`  [B, T, 560]
- `encoder_feature_length = encoder_out_lens`

**Common after branching**:
- Apply projector: `projector_outs = projector(encoder_outs)`  [B, T', 1536]
- Merge with text embeddings by replacing `<speech>` token positions
- Pass merged embeddings to LLM: `outputs = self.llm(inputs_embeds=merged_embeds, attention_mask=attention_mask, labels=labels)`

**SWIFT equivalent**: Implement all branches exactly in custom model's `forward()` method.

**Pitfall**: Forgetting to remove the 4 query tokens ([:, 4:, :]) before passing to projector.
**Prevention**: The query tokens are prepended in step 2; they MUST be stripped in step 6.

### 3.3 Audio Preprocessing

**Original parameters:**
- sample_rate: **16000**
- n_mels: **80**
- frame_length: **25 ms**
- frame_shift: **10 ms**
- window_type: **hamming**
- dither: **0.001**
- low_freq: **0**
- high_freq: **8000**
- htk_compat: **True**
- use_energy: **False**
- LFR: **lfr_m=7, lfr_n=6** → output dim = 7 × 80 = **560**

**Preprocessing pipeline**:
1. `load_audio_text_image_video(audio_path, fs=16000)` — load audio
2. `extract_fbank(audio_list, data_type="sound", frontend=frontend)` — extract fbank
3. Output shape: `[B, T, 560]` after LFR

**SWIFT implementation**: Create a custom preprocessor that wraps funasr's `extract_fbank`. Register it in the dataset adapter.

**Pitfall**: Using standard Whisper mel-spectrogram instead of funasr fbank + LFR.
**Prevention**: Must use funasr's frontend or replicate exact LFR logic.

### 3.4 Prompt Template

**EXACT template string**:
```
<|im_start|>user
{task_prompt}<speech><|im_end|>
<|im_start|>assistant
{target_text}
```

- `{task_prompt}` is looked up from `multiprompt.jsonl` based on `task` field
- `<speech>` is replaced by audio embeddings during forward pass
- For inference, the prompt ends after `<|im_start|>assistant\n`; the model generates the rest

**Special token placement**: `<speech>` appears immediately after `{task_prompt}` and before `<|im_end|>`.

**System prompt**: None (Qwen2.5-Instruct default system prompt is used implicitly).

### 3.5 generate() Configuration

**Original parameters** (from `inference_batch.py` / `inference_batch_deepspeed.py`):
- num_beams: **4**
- temperature: **1.0**
- max_new_tokens: **200**
- do_sample: **False**
- top_k: **50** (default, not explicitly set)
- top_p: **1.0**
- repetition_penalty: **1.0**
- length_penalty: **1.0**
- bos_token_id: from tokenizer
- eos_token_id: from tokenizer
- pad_token_id: from tokenizer (same as eos_token_id)

**Generation call**:
```python
outputs = self.llm.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    max_new_tokens=200,
    num_beams=4,
    do_sample=False,
    top_p=1.0,
    temperature=1.0,
    repetition_penalty=1.0,
    length_penalty=1.0,
    pad_token_id=pad_token_id,
    eos_token_id=eos_token_id,
)
```

### 3.6 Model Behavior Flags

| Flag | Default | Where Set | Impact |
|------|---------|-----------|--------|
| ctc_posterior | **true** | `train_config.ctc_posterior` | If true, projector receives 25055-dim CTC posterior instead of 560-dim encoder features |
| do_psd | **true** | `train_config.do_psd` | If true, apply PSD to prune encoder frames using CTC alignment |
| voca_trans | **false** | `train_config.voca_trans` | If true, use LegoSLM baseline (projector output vocab-dim then einsum with embed matrix) |
| gt_emb | **true** (train) / **false** (infer) | `train_config.gt_emb` | If true, use GT text embedding as auxiliary input |
| gt_emb_noise | **true** (train) | `train_config.gt_emb_noise` | If true, add Gaussian noise to GT embedding |
| top1_emb | **false** | `train_config.top1_emb` | If true, use argmax CTC prediction to index embedding matrix |
| freeze_encoder | **true** | `train_config.freeze_encoder` | Freeze SenseVoice encoder parameters |
| freeze_projector | **false** | `train_config.freeze_projector` | Freeze projector parameters |
| freeze_llm | **true** | `train_config.freeze_llm` | Freeze LLM parameters (set to false if doing full fine-tuning) |
| use_peft | **false** | `train_config.use_peft` | Use LoRA for LLM fine-tuning |

**Pitfall**: Using `gt_emb=true` during inference. The checkpoint was trained with `gt_emb=true` but inference MUST use `gt_emb=false` because GT is not available.
**Prevention**: Explicitly set `gt_emb=false` in inference config.

### 3.7 Training Configuration

**Optimizer:**
- Type: **AdamW**
- lr: **1e-4** (from `TrainConfig`)
- betas: **(0.9, 0.999)**
- weight_decay: **0.0**
- eps: **1e-06**

**LR Schedule:**
- Type: **cosine with warmup** (DeepSpeed default)
- Warmup steps: **1000** (from `TrainConfig`) or **200** (from `ds_config.json`)
- Min lr: not explicitly set

**Batch Strategy:**
- Type: **dynamic**
- batch_size: determined by `MultiTaskDynamicBatchDataset`
- max_frame_length: **3000** (training) / **3000** (eval) — from shell script
- ds_rate: **8**

**Loss:**
- Primary: **LLM built-in CrossEntropyLoss** (via `model_outputs = self.llm(..., labels=labels)`)
- ignore_index: **-100** (prompt tokens are set to -100 so they don't contribute to loss)
- label_smoothing: **0.0**
- No auxiliary CTC loss during training

**Gradient clipping:** **1.0** (from DeepSpeed config)

**Mixed precision:** **bf16** (`use_fp16=false` but DeepSpeed config enables bf16)

### 3.8 Checkpoint Handling

**Save:**
- DeepSpeed saves with `exclude_frozen_parameters=True`
- Converts to fp32 state dict
- When saving PEFT: saves `encoder_projector` + unfrozen parts

**Load:**
- `ckpt_dict = torch.load(ckpt_path, map_location="cpu")`
- `model.load_state_dict(ckpt_dict, strict=False)`
- The provided checkpoint (`pytorch_model.bin`, ~218MB) contains ONLY projector weights:
  - `encoder_projector.norm.weight/bias: [25055]`
  - `encoder_projector.ffn.0.weight/bias: [2048, 25055]` / `[2048]`
  - `encoder_projector.ffn.2.weight/bias: [1536, 2048]` / `[1536]`

**Key mapping**: No renaming needed — checkpoint keys match model keys directly.

**Frozen layers**:
- Encoder: frozen (no checkpoint keys for encoder)
- LLM: frozen (no checkpoint keys for LLM)
- Projector: trainable (checkpoint contains projector weights)

### 3.9 Tokenizer & Embedding

- **Base tokenizer**: Qwen2.5-1.5B-Instruct tokenizer
- **vocab_size**: 151936 (Qwen2.5 default)
- **resize_embeddings**: **NO** — `resize_token_embeddings` is NOT called
- **padding_side**: "right" for training, "left" for inference
- **Special tokens**:
  - `<speech>`: added via `tokenizer.add_special_tokens({"additional_special_tokens": ["<speech>"]})`
  - `pad_token_id = eos_token_id`
  - `default_ignore_token = -100`
- **Note**: `<speech>` token ID may be OOV for the LLM embedding matrix, but it doesn't matter because the token is replaced by audio embeddings before being passed to the LLM.

**Pitfall**: Calling `model.resize_token_embeddings()` — this would change the embedding matrix shape and break checkpoint compatibility.
**Prevention**: Do NOT call `resize_token_embeddings()`. Only add the token to the tokenizer for ID conversion purposes.

### 3.10 Attention Mask

**Original logic**:
- In collator: `attention_mask = input_ids.ge(-1)` → all True (since input_ids >= -1 always)
- For padding positions: padded with `False`
- Padding style: "right" for training, "left" for inference

**SWIFT implementation**: The custom collator must compute attention_mask correctly, marking padding positions as False.

**Pitfall**: Using `attention_mask = input_ids != pad_token_id` which may give incorrect masks if pad_token_id equals eos_token_id.
**Prevention**: Use explicit padding tracking in collator rather than deriving from input_ids.

## 4. SWIFT-Specific Mapping

### 4.1 Model Registration

**Strategy**: B (Custom Model with Standard Backbone)

```python
from swift.model import register_model, ModelMeta

def get_tasu_model(model_id, **kwargs):
    # Load Qwen2.5 backbone via SWIFT
    # Load SenseVoice encoder manually
    # Load projector manually
    # Wrap in TASUModel
    return model, tokenizer

register_model(
    ModelMeta(
        model_type='tasu',
        model_groups=[],
        template='tasu',
        get_function=get_tasu_model,
    )
)
```

### 4.2 Training Entry

**SWIFT v4.2.1 API** (verified in container):
```python
from swift import SftArguments, sft_main

args = SftArguments(
    model_type='tasu',
    dataset=['tasu_dataset'],
    learning_rate=1e-4,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=1000,
    bf16=True,
    deepspeed='scripts/ds_config.json',
    attn_impl='sdpa',  # For Ascend NPU
    output_dir='output/tasu',
)
sft_main(args)
```

### 4.3 Dataset Registration

```python
from swift.dataset import register_dataset, DatasetMeta

register_dataset(
    DatasetMeta(
        dataset_name='tasu_dataset',
        media_type='audio',
        preprocess_func=preprocess_tasu,
    )
)
```

**Preprocessing function**:
1. Read JSONL line
2. Look up `task` in `multiprompt.jsonl` to get prompt
3. Format query: `<|im_start|>user\n{prompt}<speech><|im_end|>\n<|im_start|>assistant\n`
4. Response = `target` field
5. Audio = `path` field (wav/ark/flac)
6. Return dict with `query`, `response`, `audio`

## 5. Known SWIFT Constraints

1. **NPU Flash Attention**: `use_flash_attn` does NOT exist in Ascend NPU environments. Use `attn_impl="sdpa"` in `SftArguments`.
2. **Shell script format**: Use `--attn_impl sdpa` (space-separated), NOT `--attn_impl=sdpa`.
3. **SWIFT v4.2.1 API**: Use `SftArguments` (not `TrainArguments`), `sft_main` (not `SwiftSft`), `get_model_processor` (not `get_model_tokenizer`).
4. **DeepSpeed mandatory**: For multi-GPU/NPU training, DeepSpeed config is required.
5. **DatasetMeta**: Only supports `dataset_path` or `hf_dataset_id`, not direct `train_dataset` object.

## 6. Implementation Order

1. `model/projector.py` — Copy `EncoderProjectorLinearSiLU` from source (simplest, no dependencies)
2. `model/swift_model.py` — Implement `TASUModel` with exact forward pass logic
3. `model/register.py` — `@register_model` with `ModelMeta`
4. `training/adapter.py` — Dataset registration, multiprompt lookup, custom collator
5. `training/data_prep.py` — JSONL conversion utilities
6. `scripts/ds_config.json` — DeepSpeed ZeRO-2 configuration
7. `scripts/swift.sh` — Training launcher
8. `scripts/infer_swift.py` — Inference with exact generate() parameters

## 7. Verification Checklist

After Worker implements, verify:
- [ ] `encoder_projector.ffn.0.weight` shape is `[2048, 25055]` (not 560)
- [ ] Forward pass produces correct output shape [B, T', 1536] after projector
- [ ] PSD merges adjacent identical non-blank frames and filters blanks with threshold 0.9
- [ ] Training runs 10 steps without error on subset data
- [ ] Inference script loads checkpoint (`pytorch_model.bin`) with `strict=False`
- [ ] `generate()` uses `num_beams=4`, `max_new_tokens=200`, `do_sample=False`
- [ ] Prompt template matches exactly: `<|im_start|>user\n{prompt}<speech><|im_end|>\n<|im_start|>assistant\n`
- [ ] Audio preprocessing outputs [B, T, 560] via funasr fbank + LFR
- [ ] `<speech>` token is added to tokenizer but `resize_token_embeddings` is NOT called
- [ ] Labels for prompt tokens are set to -100 (ignored in loss)
- [ ] `freeze_encoder=true`, `freeze_llm=true`, `freeze_projector=false`
- [ ] `gt_emb=false` during inference (even if checkpoint was trained with `gt_emb=true`)
