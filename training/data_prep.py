"""
TASU Data Preparation Utilities

Convert original JSONL format to SWIFT-compatible JSONL.
"""
import json
import os
from typing import List, Optional


def load_multiprompt(path: str) -> dict:
    """Load multiprompt.jsonl into task->prompt dict."""
    prompts = {}
    if not os.path.exists(path):
        return prompts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompts[item["task"]] = item["prompt"]
    return prompts


def convert_sample(item: dict, prompt_dict: dict) -> dict:
    """Convert a single original sample to SWIFT format."""
    task = item.get("task", "ASR")
    task_prompt = prompt_dict.get(task, "")
    target = item.get("target", "")
    audio_path = item.get("path", "")

    query = (
        f"<|im_start|>user\n"
        f"{task_prompt}<speech><|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    return {
        "query": query,
        "response": target,
        "audio": [audio_path],
    }


def convert_jsonl(
    input_path: str,
    output_path: str,
    multiprompt_path: str = "conf/multiprompt.jsonl",
) -> int:
    """Convert original JSONL to SWIFT JSONL. Returns number of samples."""
    prompt_dict = load_multiprompt(multiprompt_path)
    count = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            converted = convert_sample(item, prompt_dict)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
    return count


def create_subset(input_path: str, output_path: str, n: int = 100) -> int:
    """Create a subset of n samples for quick validation."""
    count = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if count >= n:
                break
            fout.write(line)
            count += 1
    return count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original JSONL file")
    parser.add_argument("--output", required=True, help="Output SWIFT JSONL file")
    parser.add_argument("--multiprompt", default="conf/multiprompt.jsonl")
    parser.add_argument("--subset", type=int, default=0, help="Create subset of N samples")
    args = parser.parse_args()

    if args.subset > 0:
        n = create_subset(args.input, args.output, args.subset)
        print(f"Created subset with {n} samples -> {args.output}")
    else:
        n = convert_jsonl(args.input, args.output, args.multiprompt)
        print(f"Converted {n} samples -> {args.output}")
