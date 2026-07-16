#!/usr/bin/env python3
"""
TASU SWIFT SFT Wrapper

必须在 SWIFT 任何子模块导入之前先注入 peft dummy，否则
swift.tuners.peft 在模块级就会因为 BOFTConfig 不存在而崩溃。

用法（替代 swift sft）：
    python /workspace/tasu_swift/scripts/swift_sft_wrapper.py \
        --external_plugins /workspace/tasu_swift/scripts/tasu_plugin.py \
        --model_type tasu ...
"""
import sys
import os

# ---------------------------------------------------------------------------
# 0. 先把 workspace 加入路径，确保后续 import 能找到我们自己的模块
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

# ---------------------------------------------------------------------------
# 1. 注入 peft dummy（必须在 import swift 之前）
# ---------------------------------------------------------------------------
import peft

_MISSING = []
for _name in [
    "BOFTConfig", "BOFTModel", "LoftQConfig", "LoHaConfig",
    "LoKrConfig", "OFTConfig", "VeraConfig", "VeraModel",
]:
    if not hasattr(peft, _name):
        _cls = type(_name, (), {})
        if _name in ("VeraModel", "BOFTModel"):
            _cls._create_and_replace = lambda *a, **k: None
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
    print(f"[swift_sft_wrapper] Injected peft dummies: {', '.join(_MISSING)}", file=sys.stderr)

# ---------------------------------------------------------------------------
# 2. 对 SWIFT 的 LoraConfig 打补丁（接受旧 peft 不认识的 kwargs）
# ---------------------------------------------------------------------------
# 注意：这里不能提前 import swift，所以先不 patch。
# tasu_plugin.py 在 --external_plugins 被加载时会负责这一步。

# ---------------------------------------------------------------------------
# 3. 启动 SWIFT sft_main
# ---------------------------------------------------------------------------
from swift.pipelines import sft_main

if __name__ == "__main__":
    sft_main()
