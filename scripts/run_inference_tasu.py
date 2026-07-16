#!/usr/bin/env python3
"""
TASU (PS-SLM) 推理脚本 — 绕过 SWIFT TransformersEngine，直接调用 model.generate()
支持命令行参数，自动计算 WER。

用法:
    python run_inference_tasu.py \
        --adapter /workspace/path/to/checkpoint \
        --data /workspace/data/asr/test-clean/multitask.jsonl \
        --output /tmp/results_test_clean.jsonl
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

# ------------------------------------------------------------------------------
# 0. 路径设置 + peft dummy 注入
# ------------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_DIR = os.path.dirname(_SCRIPT_DIR)
if _WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, _WORKSPACE_DIR)

import peft
for _name in ['BOFTConfig', 'BOFTModel', 'LoftQConfig', 'LoHaConfig',
              'LoKrConfig', 'OFTConfig', 'VeraConfig', 'VeraModel']:
    if not hasattr(peft, _name):
        _cls = type(_name, (), {})
        if _name in ('VeraModel', 'BOFTModel'):
            _cls._create_and_replace = lambda *a, **k: None
        setattr(peft, _name, _cls)

from swift.pipelines.infer.infer import SwiftInfer, InferArguments


# ------------------------------------------------------------------------------
# 1. 参数解析
# ------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='TASU batch inference')
    parser.add_argument('--adapter', type=str, required=False, default=None,
                        help='Path to LoRA adapter checkpoint dir (contains adapter_model.safetensors)')
    parser.add_argument('--ckpt_path', type=str, required=False, default=None,
                        help='Path to TASU projector checkpoint (pytorch_model.bin). '
                             'Overrides TASU_CKPT_PATH env var.')
    parser.add_argument('--encoder_dim', type=int, default=None,
                        help='Encoder projector input dim. Use 25055 for native TASU CTC-posterior projector.')
    parser.add_argument('--ctc_posterior', action='store_true',
                        help='Use CTC posterior (vocab prob) as projector input. Required for native TASU projector.')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to PS-SLM JSONL (task/target/path)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSONL path')
    parser.add_argument('--model', type=str,
                        default='/workspace/model/Qwen2.5-1.5B-Instruct',
                        help='Base LLM path')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                        help='Max new tokens to generate (64 causes truncation on long utterances)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (currently only 1 is well-tested)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device override, e.g. npu:0. Auto-detected if None.')
    parser.add_argument('--num_beams', type=int, default=1,
                        help='Number of beams for generation')
    parser.add_argument('--do_sample', action='store_true',
                        help='Use sampling instead of greedy decoding')
    parser.add_argument('--length_penalty', type=float, default=None,
                        help='Length penalty for beam search (default: 1.0)')
    parser.add_argument('--repetition_penalty', type=float, default=None,
                        help='Repetition penalty (default: 1.0)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature (default: 1.0)')
    parser.add_argument('--top_p', type=float, default=None,
                        help='Nucleus sampling top_p (default: 1.0)')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=None,
                        help='No repeat n-gram size (default: 0)')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping for beam search')
    parser.add_argument('--no_punctuation', action='store_true',
                        help='Suppress punctuation/contraction tokens (.,?!) during generation')
    parser.add_argument('--plugin', type=str,
                        default='/workspace/tasu_swift/scripts/tasu_plugin.py',
                        help='Path to tasu_plugin.py')
    parser.add_argument('--task_prompt', type=str, default='Transcribe the speech into text.',
                        help='Task prompt prefix before <audio> token')
    parser.add_argument('--system', type=str, default=None,
                        help='System prompt. Use empty string \"\" to disable the default system message.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples for quick test')
    return parser.parse_args()


# ------------------------------------------------------------------------------
# 2. 推理主函数
# ------------------------------------------------------------------------------
def main():
    args = parse_args()

    # 2.1 构建 InferArguments
    model_kwargs = {}
    if args.ckpt_path:
        model_kwargs['ckpt_path'] = args.ckpt_path
    if args.encoder_dim is not None:
        model_kwargs['encoder_dim'] = args.encoder_dim
    if args.ctc_posterior:
        model_kwargs['ctc_posterior'] = True

    infer_kwargs = dict(
        model=args.model,
        model_type='tasu',
        adapters=[args.adapter] if args.adapter else [],
        external_plugins=[args.plugin],
        max_new_tokens=args.max_new_tokens,
        attn_impl='sdpa',
        torch_dtype='bfloat16',
        stream=False,
        infer_backend='transformers',
        model_kwargs=model_kwargs if model_kwargs else None,
    )
    if args.system is not None:
        infer_kwargs['system'] = args.system
    infer_args = InferArguments(**infer_kwargs)
    infer_args.eval_human = False

    # 2.2 加载模型和模板
    print("Loading model and template...")
    infer = SwiftInfer(infer_args)
    infer.jsonl_writer = None
    template = infer.template
    tokenizer = template.tokenizer
    model = infer.infer_engine.model

    device = args.device or str(next(model.parameters()).device)
    print(f"Model loaded. Device: {device}")

    # 2.3 读取数据
    print(f"Reading data from {args.data} ...")
    with open(args.data, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
    if args.limit:
        lines = lines[:args.limit]
    print(f"Total samples: {len(lines)}")

    # 2.4 逐条推理
    results = []
    for sample in tqdm(lines, desc="Inferencing"):
        inputs_dict = {
            'messages': [
                {'role': 'user', 'content': f"{args.task_prompt}<audio>"},
            ],
            'audios': [sample['path']],
        }

        encoded = template.encode(inputs_dict, return_template_inputs=False, return_length=False)

        input_ids = torch.tensor([encoded['input_ids']], device=device)
        attention_mask = torch.tensor([encoded['attention_mask']], device=device) if 'attention_mask' in encoded else None
        input_features = encoded['input_features'].unsqueeze(0).to(device) if 'input_features' in encoded else None
        input_feature_length = torch.tensor([encoded['input_feature_length']], device=device) if 'input_feature_length' in encoded else None

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
        )
        if args.length_penalty is not None:
            gen_kwargs['length_penalty'] = args.length_penalty
        if args.repetition_penalty is not None:
            gen_kwargs['repetition_penalty'] = args.repetition_penalty
        if args.temperature is not None:
            gen_kwargs['temperature'] = args.temperature
        if args.top_p is not None:
            gen_kwargs['top_p'] = args.top_p
        if args.no_repeat_ngram_size is not None:
            gen_kwargs['no_repeat_ngram_size'] = args.no_repeat_ngram_size
        if args.early_stopping:
            gen_kwargs['early_stopping'] = True
        if args.no_punctuation:
            # Suppress common punctuation/contraction tokens
            bad_ids = []
            for tid in [6, 11, 13, 30]:
                bad_ids.append([tid])
            gen_kwargs['bad_words_ids'] = bad_ids

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                input_feature_length=input_feature_length,
                **gen_kwargs,
            )

        if hasattr(output, 'sequences'):
            output_ids = output.sequences[0].tolist()
        elif isinstance(output, torch.Tensor):
            output_ids = output[0].tolist()
        else:
            output_ids = output

        pred = tokenizer.decode(output_ids, skip_special_tokens=True)

        results.append({
            'key': sample.get('key', ''),
            'gt': sample.get('target', ''),
            'pred': pred,
        })

    # 2.5 保存结果
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\nResults saved to {args.output}")
    print(f"Total samples: {len(results)}")

    # 2.6 计算 WER
    try:
        from jiwer import wer
        refs = [r['gt'] for r in results]
        hyps = [r['pred'] for r in results]
        wer_value = wer(refs, hyps)
        print(f"WER: {wer_value * 100:.2f}%")
    except ImportError:
        print("jiwer not installed, skipping WER calculation.")
        print("Install with: pip install jiwer")

    # 2.7 打印前 10 条对比
    print('\n' + '=' * 80)
    print('INFERENCE RESULTS (first 10 samples)')
    print('=' * 80)
    for i, r in enumerate(results[:10]):
        match = "✓" if r['gt'].strip().upper() == r['pred'].strip().upper() else "✗"
        print(f"\n[{i+1}] key: {r['key']} {match}")
        print(f"    GT:   {r['gt']}")
        print(f"    PRED: {r['pred']}")
    print('\n' + '=' * 80)


if __name__ == '__main__':
    main()
