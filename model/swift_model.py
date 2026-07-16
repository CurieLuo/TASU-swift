"""
TASU (PS-SLM) SWIFT Model Implementation

Architecture: SenseVoiceSmall Encoder -> CTC -> LinearSiLU Projector -> Qwen2.5 LLM
"""
import math
import re
import types
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from model.projector import EncoderProjectorLinearSiLU


def _get_projector_k(projector):
    """兼容 PEFT ModulesToSaveWrapper 的属性访问。"""
    return getattr(
        projector, 'k',
        getattr(getattr(projector, 'original_module', None), 'k', 1)
    )


class TASUModel(nn.Module):
    """
    Text-only Alignment for Speech Understanding (TASU)
    Adapted for SWIFT framework.
    """

    def __init__(
        self,
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    ):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = encoder_projector
        self.tokenizer = tokenizer

        # Proxy generation_config from LLM (required by SWIFT)
        self.generation_config = getattr(llm, "generation_config", None)

        # Behavior flags
        self.ctc_posterior = getattr(train_config, "ctc_posterior", True)
        self.do_psd = getattr(train_config, "do_psd", True)
        self.voca_trans = getattr(train_config, "voca_trans", False)
        self.gt_emb = getattr(train_config, "gt_emb", False)
        self.gt_emb_noise = getattr(train_config, "gt_emb_noise", False)
        self.top1_emb = getattr(train_config, "top1_emb", False)
        self.cross_attn = getattr(model_config, "encoder_projector", "") == "cross-attention"
        self.model_config = model_config

        # SenseVoice tokenizer for GT embedding
        from model.tokenizer import SenseVoiceTokenizer
        self.encoder_tokenizer = SenseVoiceTokenizer(model_config.encoder_path)

        # DeepSpeed LayerNorm patch (if needed)
        if getattr(train_config, "enable_deepspeed", False):
            def new_forward(self, input):
                output = F.layer_norm(
                    input.float(),
                    self.normalized_shape,
                    self.weight.float() if self.weight is not None else None,
                    self.bias.float() if self.bias is not None else None,
                    self.eps,
                )
                return output.type_as(input)
            for item in self.modules():
                if isinstance(item, nn.LayerNorm):
                    item.forward = types.MethodType(new_forward, item)

    def psd(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ctc_posterior: torch.Tensor,
        blank_id: int = 0,
        blank_threshold: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        import os
        if blank_threshold is None:
            blank_threshold = float(os.getenv("TASU_PSD_BLANK_THRESHOLD", "0.90"))
        """
        Phoneme-Synchronous Decoding.
        1. Merge adjacent identical non-blank frames
        2. Filter blank frames with threshold
        """
        B, T, D = encoder_out.shape
        device = encoder_out.device
        is_log_prob = ctc_posterior.max() <= 0
        ctc_probs = ctc_posterior.exp() if is_log_prob else ctc_posterior
        keep_frames, new_lens = [], []

        for b in range(B):
            L = encoder_out_lens[b].item()
            if L == 0:
                keep_frames.append(encoder_out.new_zeros(0, D))
                new_lens.append(0)
                continue
            ids = ctc_probs[b, :L].argmax(dim=-1)

            merged_feats, merged_blank_probs = [], []
            start = 0
            for end in range(1, L + 1):
                if end == L or ids[end] != ids[start]:
                    seg_len = end - start
                    char_id = ids[start].item()
                    if char_id == blank_id:
                        for t in range(start, end):
                            merged_feats.append(encoder_out[b, t])
                            merged_blank_probs.append(ctc_probs[b, t, blank_id])
                    else:
                        if seg_len > 5:
                            print(f"[PSD] Warning: batch={b}, char={char_id}, continuous frames={seg_len} (>5)")
                        merged_feats.append(encoder_out[b, start:end].mean(dim=0))
                        avg_blank_prob = ctc_probs[b, start:end, blank_id].mean()
                        merged_blank_probs.append(avg_blank_prob)
                    start = end

            merged_feats = torch.stack(merged_feats, dim=0)
            merged_blank_probs = torch.tensor(merged_blank_probs, device=device)
            mask = merged_blank_probs < blank_threshold
            keep = mask.nonzero(as_tuple=False).squeeze(-1)
            feats_after_blank = merged_feats[keep]
            keep_frames.append(feats_after_blank)
            new_lens.append(feats_after_blank.size(0))

        max_len = max(new_lens) if new_lens else 0
        if max_len == 0:
            return encoder_out.new_zeros(B, 0, D), encoder_out.new_zeros(B, dtype=torch.long, device=device)

        padded = []
        for feat in keep_frames:
            pad_len = max_len - feat.size(0)
            if pad_len > 0:
                feat = F.pad(feat, (0, 0, 0, pad_len), value=0.0)
            padded.append(feat)
        encoder_outs = torch.stack(padded, dim=0)
        new_lens = torch.tensor(new_lens, dtype=torch.long, device=device)
        return encoder_outs, new_lens

    def ctc_pseudo_posterior(self, texts):
        tok = self.encoder_tokenizer
        ids_list = [tok.encode(t) for t in texts]
        lens = torch.tensor([len(ids) for ids in ids_list], dtype=torch.long)
        max_len = lens.max().item()
        vocab_size = tok.vocab_size
        batch_size = len(texts)
        posterior = torch.zeros(batch_size, max_len, vocab_size, dtype=torch.float32)
        for b, ids in enumerate(ids_list):
            for i, idx in enumerate(ids):
                posterior[b, i, idx] = 1.0
        return posterior, lens

    def ctc_pseudo_posterior_noise(self, texts):
        posterior, lens = self.ctc_pseudo_posterior(texts)
        noise = torch.randn_like(posterior) * 0.1
        posterior = posterior + noise
        posterior = F.softmax(posterior, dim=-1)
        return posterior, lens

    def _compute_encoder_features(
        self,
        input_features: torch.Tensor,
        input_feature_length: torch.Tensor,
        targets: Optional[List[str]] = None,
    ):
        speech = input_features
        B = speech.size(0)

        language_query = self.encoder.embed(
            torch.tensor([[0]], device=speech.device)
        ).repeat(B, 1, 1)
        textnorm_query = self.encoder.embed(
            torch.tensor([[2]], device=speech.device)
        ).repeat(B, 1, 1)
        event_emo_query = self.encoder.embed(
            torch.tensor([[1, 2]], device=speech.device)
        ).repeat(B, 1, 1)

        speech = torch.cat([language_query, event_emo_query, textnorm_query, speech], dim=1)
        speech_lengths = input_feature_length + 4

        raw_encoder_out, raw_encoder_out_lens = self.encoder.encoder(speech, speech_lengths)
        if isinstance(raw_encoder_out, tuple):
            raw_encoder_out = raw_encoder_out[0]

        raw_logits = self.encoder.ctc.ctc_lo(raw_encoder_out)
        raw_ctc_posterior = torch.softmax(raw_logits, dim=-1)
        ctc_posterior = raw_ctc_posterior[:, 4:, :]
        encoder_out = raw_encoder_out[:, 4:, :]
        encoder_out_lens = torch.clamp(raw_encoder_out_lens - 4, min=0)
        return encoder_out, encoder_out_lens, ctc_posterior

    def _run_projector(
        self,
        encoder_out,
        encoder_out_lens,
        ctc_posterior,
        targets=None,
        device=None,
    ):
        if self.ctc_posterior:
            if not self.voca_trans:
                if self.gt_emb and targets is not None:
                    if self.gt_emb_noise:
                        encoder_outs, encoder_feature_length = self.ctc_pseudo_posterior_noise(targets)
                    else:
                        encoder_outs, encoder_feature_length = self.ctc_pseudo_posterior(targets)
                    if device is not None:
                        encoder_outs = encoder_outs.to(device, non_blocking=True)
                        encoder_feature_length = encoder_feature_length.to(device, non_blocking=True)
                else:
                    if self.do_psd:
                        encoder_outs, encoder_feature_length = self.psd(
                            ctc_posterior, encoder_out_lens, ctc_posterior, self.encoder.blank_id
                        )
                    else:
                        encoder_outs, encoder_feature_length = ctc_posterior, encoder_out_lens

                if self.cross_attn:
                    with torch.no_grad():
                        llm_embedding = self.llm.get_input_embeddings().weight
                    projector_outs = self.encoder_projector(encoder_outs, llm_embedding.detach())
                    projector_feature_length = encoder_feature_length
                else:
                    projector_outs = self.encoder_projector(encoder_outs)
                    projector_feature_length = encoder_feature_length // _get_projector_k(self.encoder_projector)
            else:
                if self.do_psd:
                    projector_outs = self.encoder_projector(encoder_out)
                    projector_feature_length = encoder_out_lens // _get_projector_k(self.encoder_projector)
                    ctc_post = torch.softmax(projector_outs, dim=-1)
                    projector_outs, projector_feature_length = self.psd(
                        projector_outs, projector_feature_length, ctc_post, self.encoder.blank_id
                    )
                else:
                    projector_outs = self.encoder_projector(encoder_out)
                    projector_feature_length = encoder_out_lens // _get_projector_k(self.encoder_projector)

                llm_embedding = self.llm.get_input_embeddings()
                embed_matrix = llm_embedding.weight
                V_real = projector_outs.size(-1) - 1
                logits_no_blank = projector_outs[..., :V_real]
                ctc_outs = torch.softmax(logits_no_blank, dim=-1)
                projector_outs = torch.einsum("btv,vh->bth", ctc_outs, embed_matrix[:V_real])
                if self.top1_emb:
                    top1_ids = ctc_outs.argmax(dim=-1).to(torch.int32)
                    projector_outs = embed_matrix[top1_ids]
        else:
            if self.do_psd:
                encoder_outs, encoder_feature_length = self.psd(
                    encoder_out, encoder_out_lens, ctc_posterior, self.encoder.blank_id
                )
            else:
                encoder_outs, encoder_feature_length = encoder_out, encoder_out_lens
            projector_outs = self.encoder_projector(encoder_outs)
            projector_feature_length = encoder_feature_length // _get_projector_k(self.encoder_projector)

        return projector_outs, projector_feature_length

    def _get_model_device(self):
        return next(self.llm.parameters()).device

    def _move_to_device(self, tensor, device):
        if tensor is not None and hasattr(tensor, 'to'):
            return tensor.to(device)
        return tensor

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_feature_length: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        device = self._get_model_device()
        input_ids = self._move_to_device(input_ids, device)
        input_features = self._move_to_device(input_features, device)
        attention_mask = self._move_to_device(attention_mask, device)
        input_feature_length = self._move_to_device(input_feature_length, device)
        labels = self._move_to_device(labels, device)

        encoder_out, encoder_out_lens, ctc_posterior = self._compute_encoder_features(
            input_features, input_feature_length
        )

        projector_outs, projector_feature_length = self._run_projector(
            encoder_out, encoder_out_lens, ctc_posterior,
            targets=kwargs.get("GT"), device=labels.device if labels is not None else None,
        )

        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds, attention_mask, labels, position_ids, _ = self._merge_input_ids_with_audio_features(
            projector_outs, projector_feature_length, inputs_embeds, input_ids, attention_mask, labels
        )

        model_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
        return model_outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_feature_length: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        targets: Optional[List[str]] = None,
        **gen_kwargs,
    ):
        device = self._get_model_device()
        input_ids = self._move_to_device(input_ids, device)
        input_features = self._move_to_device(input_features, device)
        attention_mask = self._move_to_device(attention_mask, device)
        input_feature_length = self._move_to_device(input_feature_length, device)
        labels = self._move_to_device(labels, device)

        encoder_out, encoder_out_lens, ctc_posterior = self._compute_encoder_features(
            input_features, input_feature_length
        )

        projector_outs, projector_feature_length = self._run_projector(
            encoder_out, encoder_out_lens, ctc_posterior, targets=targets,
            device=input_ids.device,
        )

        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds, attention_mask, labels, position_ids, _ = self._merge_input_ids_with_audio_features(
            projector_outs, projector_feature_length, inputs_embeds, input_ids, attention_mask, labels
        )

        llm_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": gen_kwargs.get("max_new_tokens", 200),
            "num_beams": gen_kwargs.get("num_beams", 4),
            "do_sample": gen_kwargs.get("do_sample", False),
            "min_length": gen_kwargs.get("min_length", 1),
            "top_p": gen_kwargs.get("top_p", 1.0),
            "repetition_penalty": gen_kwargs.get("repetition_penalty", 1.0),
            "length_penalty": gen_kwargs.get("length_penalty", 1.0),
            "temperature": gen_kwargs.get("temperature", 1.0),
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
        }
        for key in ["no_repeat_ngram_size", "early_stopping", "bad_words_ids"]:
            if key in gen_kwargs:
                llm_kwargs[key] = gen_kwargs[key]
        outputs = self.llm.generate(**llm_kwargs)
        return outputs

    def _merge_input_ids_with_audio_features(
        self, audio_features, num_audio_tokens, inputs_embeds, input_ids, attention_mask, labels
    ):
        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_features_mask = torch.arange(max_audio_tokens).expand(num_audios, max_audio_tokens).to(
            num_audio_tokens.device
        ) < num_audio_tokens.unsqueeze(1)
        masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)
        batch_size, sequence_length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long, device=input_ids.device)
        _left_padding = torch.any(attention_mask[:, 0] == 0)
        _right_padding = torch.any(attention_mask[:, -1] == 0)

        left_padding = True
        if batch_size > 1:
            if _left_padding and not _right_padding:
                left_padding = True
            elif not _left_padding and _right_padding:
                left_padding = False
            elif not _left_padding and not _right_padding:
                left_padding = True
            else:
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

        special_audio_token_mask = input_ids == self.tokenizer.default_speech_token
        num_special_audio_tokens = torch.sum(special_audio_token_mask, dim=-1)

        target_device = inputs_embeds.device
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)
        num_audio_tokens = num_audio_tokens.to(target_device)
        batch_indices, non_audio_indices = torch.where(
            (input_ids != self.tokenizer.default_speech_token) & (attention_mask == 1)
        )

        token_placeholder_num = torch.zeros_like(input_ids)
        token_placeholder_num[special_audio_token_mask] = num_audio_tokens.long() - 1
        token_placeholder_num = token_placeholder_num + 1
        new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
        max_token_num = token_placeholder_num.sum(-1).max()
        nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_audio_pad[:, None]
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        final_embedding = torch.zeros(
            batch_size, max_token_num, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_token_num, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_token_num), self.tokenizer.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )

        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_audio_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full((batch_size, max_token_num), self.tokenizer.default_ignore_token, dtype=input_ids.dtype, device=inputs_embeds.device).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_audio_indices]

        audio_to_overwrite = torch.full(
            (batch_size, max_token_num), True, dtype=torch.bool, device=inputs_embeds.device
        )
        audio_to_overwrite[batch_indices, text_to_overwrite] = False
        seq_indices = torch.arange(max_token_num).unsqueeze(0).to(target_device)
        seq_indices = seq_indices.expand(batch_size, max_token_num)

        if left_padding:
            max_token_num = max_token_num.to(target_device)
            val = (max_token_num - seq_indices) <= (
                token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1)
            )[:, None]
        else:
            val = seq_indices < (token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1))[:, None]

        audio_to_overwrite &= val

        if audio_to_overwrite.sum() != num_audio_tokens.sum():
            raise ValueError(
                f"The input provided to the model are wrong. The number of audio tokens is {num_special_audio_tokens} while"
                f" the number of audio given to the model is {num_audios}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[audio_to_overwrite] = (
            masked_audio_features.contiguous().reshape(-1, embed_dim).to(target_device).to(final_embedding.dtype)
        )
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids
