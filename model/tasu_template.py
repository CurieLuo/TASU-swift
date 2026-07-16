"""
TASU Template for SWIFT 4.2.1

Handles:
  - Text encoding with Qwen2.5 chat template
  - Audio preprocessing via SenseVoice WavFrontend (fbank + LFR = 560-dim)
  - Padding of audio features in the data collator
  - Compatibility with raw PS-SLM JSONL format (task/target/path)
"""
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from swift.template.base import Template
from swift.template.template_inputs import StdTemplateInputs


class TASUTemplate(Template):
    """
    Template for TASU (PS-SLM) multi-task speech understanding.

    Prompt format (identical to original):
        <|im_start|>user
        {task_prompt}<speech><|im_end|>
        <|im_start|>assistant

    The ``<speech>`` token is added as a special token to the Qwen tokenizer
    in :class:`model.register.TASULoader`.  During encoding the audio file
    associated with the sample is loaded, passed through SenseVoice's
    WavFrontend, and the resulting fbank features (``[T, 560]``) are placed
    into the batch dict as ``input_features`` / ``input_feature_length``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frontend = None          # lazy initialised SenseVoice frontend
        self._frontend_device = "cpu"  # frontend runs on CPU to avoid NPU mem

    # ------------------------------------------------------------------
    # Lazy SenseVoice frontend initialisation
    # ------------------------------------------------------------------
    def _init_frontend(self):
        if self._frontend is not None:
            return
        from funasr import AutoModel
        _, kwargs = AutoModel.build_model(model="iic/SenseVoiceSmall", device=self._frontend_device)
        self._frontend = kwargs["frontend"]

    # ------------------------------------------------------------------
    # Audio feature extraction (per sample)
    # ------------------------------------------------------------------
    def _extract_audio(self, audio_path: str):
        """
        Extract fbank features from a single audio file.
        Returns (input_features, input_feature_length) where
        input_features is [T, 560] and input_feature_length is scalar.
        """
        self._init_frontend()
        from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank

        # Handle Kaldi ark files (path format: file.ark:offset)
        if ".ark:" in audio_path or audio_path.endswith(".ark"):
            import kaldiio
            import numpy as np
            _, data = kaldiio.load_mat(audio_path)
            # Normalize int16 PCM to float32 in [-1, 1]
            data = data.astype(np.float32) / 32768.0
            audio_list = [data]
        else:
            audio_list = load_audio_text_image_video(
                [audio_path],
                fs=self._frontend.fs,
                audio_fs=16000,
                data_type="sound",
            )
        input_features, input_feature_length = extract_fbank(
            audio_list,
            data_type="sound",
            frontend=self._frontend,
        )
        # extract_fbank returns batched results; take first (and only) sample
        return input_features[0], input_feature_length[0]

    # ------------------------------------------------------------------
    # Encode: text tokenisation + audio feature extraction
    # ------------------------------------------------------------------
    def encode(self, inputs, return_template_inputs: bool = False, return_length: bool = False):
        # Compatible with raw PS-SLM JSONL: convert (task, target, path) -> (messages, audios)
        if isinstance(inputs, dict) and 'path' in inputs and 'task' in inputs:
            task = inputs.get('task', 'ASR')
            target = inputs.get('target', '')
            audio_path = inputs['path']
            # Match original PS-SLM prompt format: instruction + <audio> (replaced to <speech> by replace_tag)
            instruction_map = {
                "ASR": "Transcribe the speech into text.",
                "GR": "Transcribe the speech into text.",
                "S2TT": "Translate the speech into the target language text.",
                "SER": "Recognize the emotion from the speech.",
                "SLU": "Understand the speech and extract the slot values.",
            }
            instruction = instruction_map.get(task, task)
            inputs = {
                'messages': [
                    {'role': 'user', 'content': instruction + "<audio>"},
                    {'role': 'assistant', 'content': target},
                ],
                'audios': [audio_path],
            }
        return super().encode(inputs, return_template_inputs=return_template_inputs, return_length=return_length)

    def replace_tag(self, media_type, index, inputs):
        """Override SWIFT's default audio placeholder to use TASU's <speech> token."""
        if media_type == 'audio':
            return ['<speech>']
        return super().replace_tag(media_type, index, inputs)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        # 1. Encode text with the Qwen chat template
        encoded = super()._encode(inputs)

        # 2. If there is an audio file attached, extract fbank features
        if inputs.audios:
            # TASU supports exactly one audio per sample
            audio_path = inputs.audios[0]
            input_features, input_feature_length = self._extract_audio(audio_path)
            encoded["input_features"] = input_features
            encoded["input_feature_length"] = input_feature_length

        return encoded

    # ------------------------------------------------------------------
    # Data collator: pad audio features temporally
    # ------------------------------------------------------------------
    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        # 1. Let the base class handle input_ids / labels / attention_mask
        res = super()._data_collator(batch, padding_to=padding_to)

        # 2. Handle input_features temporal padding
        input_features = [
            b["input_features"] for b in batch
            if b.get("input_features") is not None
        ]
        input_feature_length = [
            b["input_feature_length"] for b in batch
            if b.get("input_feature_length") is not None
        ]

        if input_features:
            max_t = max(feat.size(0) for feat in input_features)
            padded = []
            for feat in input_features:
                pad_len = max_t - feat.size(0)
                if pad_len > 0:
                    feat = F.pad(feat, (0, 0, 0, pad_len), value=0.0)
                padded.append(feat)
            res["input_features"] = torch.stack(padded, dim=0)          # [B, T_max, 560]
            res["input_feature_length"] = torch.stack(input_feature_length)  # [B]

        return res
