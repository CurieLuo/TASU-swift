"""
TASU SWIFT Plugin

This file is loaded via ``--external_plugins`` when running ``swift sft``
or ``swift infer``.  It performs three essential setup steps:

1. Inject peft compatibility dummies (BOFTConfig, LoftQConfig, etc.) so that
   SWIFT 4.2.1 can import on the container's older peft (0.6.0).
2. Import ``model.register`` → registers the ``tasu`` model type and its
   ``TASUTemplate`` with SWIFT.
3. Import ``training.adapter`` → registers TASU datasets (asr_task, gr_task,
   s2tt_task, ser_task, slu_task) with SWIFT.

Usage::

    swift sft --external_plugins /workspace/tasu_swift/scripts/tasu_plugin.py \
              --model_type tasu --model /path/to/Qwen2.5-1.5B-Instruct ...
"""
import os
import sys

# ---------------------------------------------------------------------------
# 1. Ensure the workspace is on PYTHONPATH so all imports work
# ---------------------------------------------------------------------------
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_DIR = os.path.dirname(_PLUGIN_DIR)
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

# ---------------------------------------------------------------------------
# 2. Peft compatibility shim (must run BEFORE any swift import)
# ---------------------------------------------------------------------------
import peft

_MISSING = []
for _name in [
    "BOFTConfig",
    "BOFTModel",
    "LoftQConfig",
    "LoHaConfig",
    "LoKrConfig",
    "OFTConfig",
    "VeraConfig",
    "VeraModel",
]:
    if not hasattr(peft, _name):
        _cls = type(_name, (), {})
        # SWIFT's hot_patch_peft_module() accesses _create_and_replace on these dummies
        if _name in ("VeraModel", "BOFTModel"):
            _cls._create_and_replace = lambda *args, **kwargs: None
        setattr(peft, _name, _cls)
        _MISSING.append(_name)

for _mod_name, _class_name in [
    ("peft.tuners.adalora", "AdaLoraModel"),
    ("peft.tuners.lora", "Embedding"),
]:
    try:
        __import__(_mod_name)
        _mod = sys.modules[_mod_name]
        if not hasattr(_mod, _class_name):
            setattr(_mod, _class_name, type(_class_name, (), {}))
    except Exception:
        pass

if _MISSING:
    print(f"[tasu_plugin] Injected peft dummies: {', '.join(_MISSING)}")

# ---------------------------------------------------------------------------
# 3. Register TASU model + template
# ---------------------------------------------------------------------------
import model.register  # noqa: F401,E402

# ---------------------------------------------------------------------------
# 4. Monkey-patch SWIFT's LoraConfig to accept kwargs old peft rejects
# ---------------------------------------------------------------------------
# SWIFT's LoraConfig is a dataclass that does NOT call peft.LoraConfig.__init__,
# so patching peft.LoraConfig is ineffective.  We must patch SWIFT's wrapper.
from swift.tuners.peft import LoraConfig as _SwiftLoraConfig  # noqa: E402

_SWIFT_LORA_EXTRA = {"use_rslora", "use_dora"}
_orig_swift_lora_init = _SwiftLoraConfig.__init__


def _patched_swift_lora_init(self, *args, **kwargs):
    for key in list(kwargs.keys()):
        if key in _SWIFT_LORA_EXTRA:
            kwargs.pop(key)
    return _orig_swift_lora_init(self, *args, **kwargs)


_SwiftLoraConfig.__init__ = _patched_swift_lora_init

# ---------------------------------------------------------------------------
# 5. Monkey-patch get_modules_to_save so projector is always saved with LoRA
# ---------------------------------------------------------------------------
from swift.pipelines.train import tuner as _tuner_module  # noqa: E402

_orig_get_modules_to_save = _tuner_module.get_modules_to_save


def _patched_get_modules_to_save(args, model, task_type=None):
    modules_to_save = _orig_get_modules_to_save(args, model, task_type)
    # Detect TASU model by checking for encoder_projector
    if hasattr(model, "tasu_model") and hasattr(model.tasu_model, "encoder_projector"):
        # PEFT modules_to_save expects the leaf module name on the base model.
        # The projector sits at base_model.model.tasu_model.encoder_projector,
        # so the string to match is "encoder_projector".
        if "encoder_projector" not in modules_to_save:
            modules_to_save.append("encoder_projector")
    return modules_to_save


_tuner_module.get_modules_to_save = _patched_get_modules_to_save

# ---------------------------------------------------------------------------
# 6. Monkey-patch get_multimodal_target_regex for TASU (no model_arch)
# ---------------------------------------------------------------------------
from swift.utils import transformers_utils as _tf_utils  # noqa: E402

_orig_get_multimodal_target_regex = _tf_utils.get_multimodal_target_regex


def _patched_get_multimodal_target_regex(model, *args, **kwargs):
    model_arch = getattr(getattr(model, 'model_meta', None), 'model_arch', None)
    if model_arch is None:
        # TASU has no model_arch; fall back to finding all linear modules
        return _tf_utils.find_all_linears(model)
    return _orig_get_multimodal_target_regex(model, *args, **kwargs)


_tf_utils.get_multimodal_target_regex = _patched_get_multimodal_target_regex

# ---------------------------------------------------------------------------
# 7. Monkey-patch AutoPreprocessor to handle raw PS-SLM JSONL
# ---------------------------------------------------------------------------
from swift.dataset.preprocessor.core import AutoPreprocessor, RowPreprocessor  # noqa: E402

TASK_INSTRUCTIONS = {
    "asr": "Transcribe the speech into text.",
    "gr": "Transcribe the speech into text.",
    "s2tt": "Translate the speech into the target language text.",
    "ser": "Recognize the emotion from the speech.",
    "slu": "Understand the speech and extract the slot values.",
}


class PSLMPreprocessor(RowPreprocessor):
    """Convert raw PS-SLM dicts (key/task/target/path) into SWIFT messages+audios."""

    def preprocess(self, row):
        task = row.pop("task", None)
        target = row.pop("target", None)
        path = row.pop("path", None)
        # Remove any leftover non-standard keys to avoid column errors
        row.pop("key", None)
        instruction = TASK_INSTRUCTIONS.get(task, task) if task else ""
        # Append <audio> placeholder; TASUTemplate.replace_tag will map it to <speech>
        messages = [
            {"role": "user", "content": instruction + "<audio>"},
            {"role": "assistant", "content": target or ""},
        ]
        row["messages"] = messages
        row["audios"] = [path] if path else []
        return row


_orig_auto_get_preprocessor = AutoPreprocessor._get_preprocessor


def _patched_auto_get_preprocessor(self, dataset):
    features = dataset.features
    # Detect raw PS-SLM format (has both 'task' and 'path')
    if 'task' in features and 'path' in features and 'target' in features:
        return PSLMPreprocessor()
    return _orig_auto_get_preprocessor(self, dataset)


AutoPreprocessor._get_preprocessor = _patched_auto_get_preprocessor

# ---------------------------------------------------------------------------
# 8. Register TASU datasets
# ---------------------------------------------------------------------------
import training.adapter  # noqa: F401,E402

print("[tasu_plugin] TASU model, template, and datasets registered.")
