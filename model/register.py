"""
SWIFT Model Registration for TASU (PS-SLM)

Registers:
  - ModelMeta  (model_type='tasu', is_multimodal=True)
  - TASUTemplate (handles SenseVoice audio frontend + Qwen2.5 text template)
"""
import os
from typing import Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from swift.model import register_model, ModelMeta, ModelGroup, Model
from swift.model.model_meta import BaseModelLoader
from swift.template import register_template, TemplateMeta

from model.swift_model import TASUModel
from model.projector import EncoderProjectorLinearSiLU
from model.tasu_template import TASUTemplate
from model.tasu_pretrained import TASUPreTrainedModel


# ---------------------------------------------------------------------------
# Template registration
# ---------------------------------------------------------------------------

register_template(
    TemplateMeta(
        template_type="tasu",
        prefix=[],
        prompt=["<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n"],
        chat_sep=["<|im_end|>\n"],
        suffix=["<|im_end|>\n"],
        template_cls=TASUTemplate,
        system_prefix=["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"],
        default_system="You are a helpful assistant.",
        auto_add_bos=False,
        stop_words=["<|endoftext|>"],
    )
)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class TASULoader(BaseModelLoader):
    """Custom loader for TASU model that builds encoder + projector + LLM."""

    def __init__(self, model_info, model_meta, *args, **kwargs):
        self.model_info = model_info
        self.model_meta = model_meta
        self.args = args
        self.kwargs = kwargs

    def load(self) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]:
        """Load TASU model and tokenizer."""
        model_kwargs = dict(self.kwargs.get('model_kwargs', {}) or {})
        device_map = model_kwargs.pop('device_map', None)
        tasu_model, tokenizer = get_tasu_model(
            self.model_info.model_dir,
            device_map=device_map,
            **model_kwargs
        )
        # Wrap in PreTrainedModel so that peft LoRA works
        config = AutoConfig.from_pretrained(self.model_info.model_dir, trust_remote_code=True)
        model = TASUPreTrainedModel(config, tasu_model)
        # SWIFT expects model_info / model_meta / model_dir attributes on both model and tokenizer
        model.model_info = self.model_info
        model.model_meta = self.model_meta
        model.model_dir = self.model_info.model_dir
        tokenizer.model_info = self.model_info
        tokenizer.model_meta = self.model_meta
        return model, tokenizer


# ---------------------------------------------------------------------------
# get_tasu_model (unchanged logic)
# ---------------------------------------------------------------------------

def get_tasu_model(model_id: str, **kwargs) -> Tuple[TASUModel, PreTrainedTokenizerBase]:
    """
    Load TASU model for SWIFT.

    Args:
        model_id: Path to model directory or HF model ID for LLM backbone
        **kwargs: Additional arguments including encoder_path, encoder_dim, etc.

    Returns:
        (model, tokenizer)
    """
    # Parse config from kwargs or use defaults
    encoder_path = kwargs.get("encoder_path", os.getenv("ENCODER_PATH", "/workspace/model/SenseVoiceSmall"))
    llm_path = kwargs.get("llm_path", model_id)
    encoder_dim = int(kwargs.get("encoder_dim", os.getenv("TASU_ENCODER_DIM", 512)))
    llm_dim = int(kwargs.get("llm_dim", os.getenv("TASU_LLM_DIM", 1536)))
    encoder_projector = kwargs.get("encoder_projector", "linear-silu")
    encoder_projector_ds_rate = int(kwargs.get("encoder_projector_ds_rate", 1))
    ctc_linear = kwargs.get("ctc_linear", "")
    ckpt_path = kwargs.get("ckpt_path", os.getenv("TASU_CKPT_PATH", ""))
    def _parse_bool(v):
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("true", "1", "yes")
    ctc_posterior = _parse_bool(kwargs.get("ctc_posterior", os.getenv("TASU_CTC_POSTERIOR", "false")))
    do_psd = _parse_bool(kwargs.get("do_psd", os.getenv("TASU_DO_PSD", "true")))
    device_map = kwargs.get("device_map", None)

    # Behavior flags
    freeze_encoder = bool(kwargs.get("freeze_encoder", True))
    freeze_projector = bool(kwargs.get("freeze_projector", False))
    freeze_llm = bool(kwargs.get("freeze_llm", True))
    use_peft = bool(kwargs.get("use_peft", False))

    voca_trans = bool(kwargs.get("voca_trans", False))
    gt_emb = bool(kwargs.get("gt_emb", False))
    gt_emb_noise = bool(kwargs.get("gt_emb_noise", False))
    top1_emb = bool(kwargs.get("top1_emb", False))

    # Simple config namespace
    class Config:
        pass

    model_config = Config()
    model_config.llm_path = llm_path
    model_config.llm_name = os.path.basename(llm_path)
    model_config.llm_dim = llm_dim
    model_config.encoder_name = "sensevoice"
    model_config.encoder_path = encoder_path
    model_config.encoder_dim = encoder_dim
    model_config.encoder_projector = encoder_projector
    model_config.encoder_projector_ds_rate = encoder_projector_ds_rate
    model_config.ctc_linear = ctc_linear

    train_config = Config()
    train_config.freeze_encoder = freeze_encoder
    train_config.freeze_projector = freeze_projector
    train_config.freeze_llm = freeze_llm
    train_config.use_peft = use_peft
    train_config.ctc_posterior = ctc_posterior
    train_config.do_psd = do_psd
    train_config.voca_trans = voca_trans
    train_config.gt_emb = gt_emb
    train_config.gt_emb_noise = gt_emb_noise
    train_config.top1_emb = top1_emb
    train_config.enable_deepspeed = bool(kwargs.get("enable_deepspeed", False))

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    DEFAULT_SPEECH_TOKEN = "<speech>"
    DEFAULT_IGNORE_TOKEN = -100
    special_tokens_dict = {"additional_special_tokens": [DEFAULT_SPEECH_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.default_ignore_token = DEFAULT_IGNORE_TOKEN
    tokenizer.default_speech_token = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)

    # Setup LLM
    use_cache = True
    dtype_str = str(kwargs.get("torch_dtype", os.getenv("TASU_TORCH_DTYPE", "bfloat16"))).lower()
    torch_dtype = {"float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
                   "float32": torch.float32, "fp32": torch.float32,
                   "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}.get(dtype_str, torch.bfloat16)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        use_cache=use_cache,
    )

    if freeze_llm:
        for name, param in model.named_parameters():
            param.requires_grad = False
        model.eval()

    # Setup SenseVoice encoder
    from model.SenseVoice import SenseVoiceSmall
    encoder, _ = SenseVoiceSmall.from_pretrained(encoder_path)
    if freeze_encoder:
        for name, param in encoder.named_parameters():
            param.requires_grad = False
        encoder.eval()

    # Setup projector
    if encoder_projector == "linear-silu":
        encoder_projector_module = EncoderProjectorLinearSiLU(model_config)
    else:
        raise ValueError(f"Unsupported projector type: {encoder_projector}")

    if freeze_projector:
        for name, param in encoder_projector_module.named_parameters():
            param.requires_grad = False
        encoder_projector_module.eval()

    # Wrap in TASUModel
    tasu_model = TASUModel(
        encoder=encoder,
        llm=model,
        encoder_projector=encoder_projector_module,
        tokenizer=tokenizer,
        train_config=train_config,
        model_config=model_config,
    )

    # Load checkpoint if provided
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from: {ckpt_path}")
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        tasu_model.load_state_dict(ckpt_dict, strict=False)
        print("Checkpoint loaded.")

    # Move to device if specified (inference path)
    if device_map is not None and device_map != 'auto':
        target_device = device_map
        if isinstance(target_device, str):
            tasu_model = tasu_model.to(target_device)
        elif isinstance(target_device, dict):
            # For dict device_map, rely on accelerate dispatch_model on the wrapped model
            pass
    elif device_map == 'auto':
        try:
            from accelerate import dispatch_model
            tasu_model = dispatch_model(tasu_model)
        except Exception:
            pass

    return tasu_model, tokenizer


# ---------------------------------------------------------------------------
# Register with SWIFT 4.2.1
# ---------------------------------------------------------------------------

_model_group = ModelGroup(
    models=[Model(hf_model_id="Qwen/Qwen2.5-1.5B-Instruct")]
)

register_model(
    ModelMeta(
        model_type="tasu",
        model_groups=[_model_group],
        template="tasu",
        loader=TASULoader,
        is_multimodal=True,
    )
)
