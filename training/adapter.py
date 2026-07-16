"""
TASU Dataset Adapter for SWIFT

Handles multi-task JSONL data with multiprompt lookup.
Output format follows SWIFT's ``messages`` + ``audios`` convention so that
:class:`model.tasu_template.TASUTemplate` can process it end-to-end.
"""
import json
import os
from typing import Dict, List, Optional

from swift.dataset import register_dataset, DatasetMeta


def _load_multiprompt(path: str) -> Dict[str, str]:
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


def _build_messages(task_prompt: str, target: str, audio_path: str) -> dict:
    """
    Build a SWIFT-compatible dataset sample.

    The user message contains the task prompt with the ``<speech>`` token.
    The assistant message contains the target transcription / translation.
    The audio file path is passed via the ``audios`` field so that
    :class:`TASUTemplate` can load and preprocess it.
    """
    query = f"{task_prompt}<speech>"
    return {
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": target},
        ],
        "audios": [audio_path] if audio_path else [],
    }


def preprocess_tasu(examples: dict, **kwargs) -> dict:
    """
    Preprocess function for TASU dataset.

    Input fields (various naming conventions supported):
      - ``task``, ``target``, ``path``  (original PS-SLM JSONL)
      - ``query``, ``response``, ``audio``  (SWIFT-style)
      - ``messages``, ``audios``  (already SWIFT format)

    Output fields (SWIFT standard):
      - ``messages``: List[{"role": "user"/"assistant", "content": str}]
      - ``audios``: List[str]  (audio file paths)
    """
    prompt_path = kwargs.get("multitask_prompt_path", "conf/multiprompt.jsonl")
    prompt_dict = _load_multiprompt(prompt_path)

    messages_list = []
    audios_list = []

    n = len(examples.get("messages", examples.get("query", examples.get("task", []))))

    for i in range(n):
        # Case 1: already in SWIFT messages format
        if "messages" in examples:
            messages_list.append(examples["messages"][i])
            audios_list.append(examples.get("audios", examples.get("audio", []))[i])
            continue

        # Case 2: from original PS-SLM format
        task = examples.get("task", ["ASR"])[i]
        task_prompt = prompt_dict.get(task, "")
        target = examples.get("target", examples.get("response", [""]))[i]
        audio_path = examples.get("path", examples.get("audio", [""]))[i]

        sample = _build_messages(task_prompt, target, audio_path)
        messages_list.append(sample["messages"])
        audios_list.append(sample["audios"])

    return {
        "messages": messages_list,
        "audios": audios_list,
    }


# Register datasets for each task type
for task_name in ["asr_task", "gr_task", "s2tt_task", "ser_task", "slu_task"]:
    register_dataset(
        DatasetMeta(
            dataset_name=task_name,
            preprocess_func=preprocess_tasu,
        )
    )
