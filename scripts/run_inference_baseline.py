#!/usr/bin/env python3
"""
TASU Baseline 推理脚本 — 不使用 chat template，直接构造 input_ids
模仿原始 PS-SLM 的推理方式。
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_DIR = os.path.dirname(_SCRIPT_DIR)
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

# Inject peft dummies
import peft
for _name in ['BOFTConfig', 'BOFTModel', 'LoftQConfig', 'LoHaConfig', 'LoKrConfig', 'OFTConfig', 'VeraConfig', 'VeraModel']:
    if not hasattr(peft, _name):
        _cls = type(_name, (), {})
        if _name in ('VeraModel', 'BOFTModel'):
            _cls._create_and_replace = lambda *a, **k: None
        setattr(peft, _name, _cls)

from swift.pipelines.infer.infer import SwiftInfer, InferArguments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, default='/workspace/model/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--task_prompt', type=str, default='请识别语音并转写为英文,只生成转写结果:')
    return parser.parse_args()


def main():
    args = parse_args()

    infer_args = InferArguments(
        model=args.model,
        model_type='tasu',
        adapters=[],
        external_plugins=['/workspace/tasu_swift/scripts/tasu_plugin.py'],
        max_new_tokens=args.max_new_tokens,
        attn_impl='sdpa',
        torch_dtype='bfloat16',
        stream=False,
        infer_backend='transformers',
    )
    infer_args.eval_human = False

    print("Loading model...")
    infer = SwiftInfer(infer_args)
    infer.jsonl_writer = None
    template = infer.template
    tokenizer = template.tokenizer
    model = infer.infer_engine.model
    device = next(model.parameters()).device

    # 获取 <speech> token ID
    speech_token_id = tokenizer.default_speech_token
    print(f"<speech> token id: {speech_token_id}")

    # 读取数据
    with open(args.data, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
    if args.limit:
        lines = lines[:args.limit]
    print(f"Total samples: {len(lines)}")

    results = []
    for sample in tqdm(lines, desc="Inferencing"):
        # 构造 prompt：直接 encode，不使用 chat template
        prompt_text = args.task_prompt + "<speech>"
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = torch.tensor([input_ids], device=device)
        attention_mask = torch.ones_like(input_ids)

        # 通过 template 提取音频特征
        inputs_dict = {
            'messages': [{'role': 'user', 'content': prompt_text}],
            'audios': [sample['path']],
        }
        encoded = template.encode(inputs_dict, return_template_inputs=False, return_length=False)
        input_features = encoded['input_features'].unsqueeze(0).to(device)
        input_feature_length = torch.tensor([encoded['input_feature_length']], device=device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                input_feature_length=input_feature_length,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=False,
            )

        if hasattr(output, 'sequences'):
            # 只取生成部分（去掉 prompt）
            gen_ids = output.sequences[0][input_ids.shape[1]:].tolist()
        elif isinstance(output, torch.Tensor):
            gen_ids = output[0][input_ids.shape[1]:].tolist()
        else:
            gen_ids = output

        pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append({'key': sample.get('key', ''), 'gt': sample.get('target', ''), 'pred': pred})

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\nSaved to {args.output}")
    for i, r in enumerate(results[:10]):
        print(f"\n[{i+1}] {r['key']}")
        print(f"    GT:   {r['gt']}")
        print(f"    PRED: {r['pred']}")


if __name__ == '__main__':
    main()
