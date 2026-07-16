"""
TASUPreTrainedModel

A thin ``PreTrainedModel`` wrapper around :class:`TASUModel` so that
SWIFT / peft can apply LoRA adapters.  Only the projector weights are
saved when doing full fine-tuning; for LoRA training peft handles adapter
saving independently.
"""
import os
from typing import Optional

import torch
from transformers import PreTrainedModel, PretrainedConfig


class TASUPreTrainedModel(PreTrainedModel):
    """
    Wraps :class:`model.swift_model.TASUModel` inside a HuggingFace
    ``PreTrainedModel`` shell.  All forward / generate calls are delegated
    to the inner TASUModel.
    """

    # Disable the default tied-weights check – TASU does not tie embeddings.
    _tied_weights_keys = []

    def __init__(self, config: PretrainedConfig, tasu_model):
        super().__init__(config)
        self.tasu_model = tasu_model
        # Attributes SWIFT expects
        self.model_info = None
        self.model_meta = None
        self.model_dir = None

    # ------------------------------------------------------------------
    # Forward / generate delegation
    # ------------------------------------------------------------------
    def forward(self, **kwargs):
        return self.tasu_model(**kwargs)

    def generate(self, **kwargs):
        return self.tasu_model.generate(**kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    # ------------------------------------------------------------------
    # Embeddings proxy (needed by peft / SWIFT)
    # ------------------------------------------------------------------
    def get_input_embeddings(self):
        return self.tasu_model.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.tasu_model.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.tasu_model.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.tasu_model.llm.set_output_embeddings(new_embeddings)

    # ------------------------------------------------------------------
    # Saving (full fine-tuning only – saves projector only)
    # ------------------------------------------------------------------
    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        # For full fine-tuning we only need the projector weights;
        # encoder and LLM are frozen / loaded from their own checkpoints.
        state_dict = {
            k: v for k, v in self.state_dict().items()
            if "encoder_projector" in k
        }
        if state_dict:
            torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        # Also save the config so that HF can identify the model later
        self.config.save_pretrained(save_directory)
