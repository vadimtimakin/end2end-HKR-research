import warnings
from typing import Optional, Iterable, Callable, List, Union, Tuple

import numpy as np
import torch
import torchvision
import torch.nn as nn
from packaging import version
from torch.distributions.constraints import Constraint
from torch.nn import TransformerEncoderLayer, TransformerEncoder, LayerNorm, TransformerDecoder, TransformerDecoderLayer, Embedding, Transformer
import torch.nn.functional as F
from transformers import BeamScorer, LogitsProcessorList, StoppingCriteriaList, MaxLengthCriteria, MaxTimeCriteria, HammingDiversityLogitsProcessor, RepetitionPenaltyLogitsProcessor, NoRepeatNGramLogitsProcessor, NoBadWordsLogitsProcessor, \
    MinLengthLogitsProcessor, PrefixConstrainedLogitsProcessor, InfNanRemoveLogitsProcessor, ForcedEOSTokenLogitsProcessor, BeamSearchScorer, StoppingCriteria, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from transformers.generation_utils import BeamSearchOutput, BeamSearchEncoderDecoderOutput, GreedySearchOutput, GreedySearchEncoderDecoderOutput, SampleOutput, SampleEncoderDecoderOutput, BeamSampleOutput, BeamSampleEncoderDecoderOutput

from coatnet import coatnet_0, coatnet_2, coatnet_1
from convnext import convnext_small, convnext_base
from data import TRANSFORMER_EOS_TOKEN, TRANSFORMER_PAD_TOKEN, TRANSFORMER_SOS_TOKEN


def torch_int_div(tensor1, tensor2):
    """
    A function that performs integer division across different versions of PyTorch.
    """
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        return tensor1 // tensor2
    else:
        return torch.div(tensor1, tensor2, rounding_mode="floor")


def get_resnet34_backbone(pretrained=True):
    """Get resnet34 backbone."""
    m = torchvision.models.resnet34(pretrained=True)
    input_conv = nn.Conv2d(1, 64, 7, 1, 3)
    blocks = [input_conv, m.bn1, m.relu,
              m.maxpool, m.layer1, m.layer2, m.layer3]
    return nn.Sequential(*blocks)


def get_coatnet_backbone():
    m = coatnet_0(shape=(96, 384), in_chans=1)
    last_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=0)
    return nn.Sequential(m, last_pool)


class BiLSTM(nn.Module):
    """BiLSTM layer."""

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class PositionalEncoding(nn.Module):
    """Character position encoding."""

    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        if not batch_first:
            pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, x):
        if self.batch_first:
            pe = self.pe[:, :x.size(1)]
        else:
            pe = self.pe[:x.size(0), :]
        x = x + self.scale * pe
        return self.dropout(x)


class CRNN(nn.Module):
    """CRNN model."""

    def __init__(
            self, n_ctc=None, n_transformer_decoder=None, transformer_decoding_params=None
    ):
        super().__init__()
        self.enable_ctc = n_ctc is not None
        self.enable_transformer_decoder = n_transformer_decoder is not None

        d_model = 768
        encoder_layers = 4
        decoder_layers = 4
        nhead = 4

        self.feature_extractor = convnext_small(pretrained=True, in_chans=1)
        self.feature_extractor_projection = nn.Linear(768, d_model)  # FIRST ARGUMENT IS CHANNEL COUNT OF BACKBONE

        self.transformer_encoder_pos = PositionalEncoding(d_model=d_model)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1,
                                                  activation=F.gelu, layer_norm_eps=1e-6, batch_first=True),
            num_layers=encoder_layers,
            norm=LayerNorm(d_model, eps=1e-6)
        )

        if self.enable_transformer_decoder:
            self.transformer_decoder_embedding = Embedding(n_transformer_decoder, d_model)
            self.transformer_decoder_pos = PositionalEncoding(d_model=d_model)
            self.transformer_decoder = TransformerDecoder(
                decoder_layer=TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1,
                                                      activation=F.gelu, layer_norm_eps=1e-6, batch_first=True),
                num_layers=decoder_layers,
                norm=LayerNorm(d_model, eps=1e-6)
            )
            self.classifier_transformer = nn.Sequential(
                nn.Linear(d_model, n_transformer_decoder)
            )
            self.trg_mask = None
            self.transformer_decoding_params = transformer_decoding_params

        if self.enable_ctc:
            self.classifier_ctc = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, n_ctc)
            )

    def forward(self, x, x_mask, trg_transformer_decoder=None):
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = x.transpose(1, 2)
        x = self.feature_extractor_projection(x)
        x = self.transformer_encoder_pos(x)
        x = self.transformer_encoder(x)
        out_dict = {}
        if self.enable_ctc:
            y_ctc = self.classifier_ctc(x)
            y_ctc = nn.functional.log_softmax(y_ctc, dim=2).permute(1, 0, 2)
            out_dict['ctc'] = y_ctc
        if self.enable_transformer_decoder:
            if trg_transformer_decoder is not None:
                # TRAINING MODE
                if self.trg_mask is None or self.trg_mask.shape[0] != trg_transformer_decoder.shape[1]:
                    self.trg_mask = Transformer.generate_square_subsequent_mask(trg_transformer_decoder.shape[1]).to(trg_transformer_decoder.device)
                trg = trg_transformer_decoder.clone()
                trg[trg == 2] = 0  # replace EOS to padding for inputs
                trg = self.transformer_decoder_embedding(trg)
                trg = self.transformer_decoder_pos(trg)
                trg_pad_mask = trg_transformer_decoder == 0
                y_transformer = self.transformer_decoder(tgt=trg, memory=x, tgt_mask=self.trg_mask, tgt_key_padding_mask=trg_pad_mask)
                y_transformer = self.classifier_transformer(y_transformer)
                out_dict['transformer'] = y_transformer
            else:
                generated = self.generate(
                    x,
                    num_return_sequences=1,
                    **self.transformer_decoding_params
                )
                out_dict['transformer'] = generated.sequences[:, 1:]  # cut out <SOS> token

        return out_dict

    def _beam_search(
            self,
            decoder_inputs: torch.Tensor,
            memory: torch.Tensor,
            beam_scorer: BeamScorer,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            pad_token_id: int,
            eos_token_id: int
    ) -> BeamSearchOutput:
        # init values
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = decoder_inputs.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = ()
        beam_indices = (tuple(() for _ in range(batch_beam_size)))

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=decoder_inputs.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:
            tgt = self.transformer_decoder_pos(self.transformer_decoder_embedding(decoder_inputs))
            next_token_logits = self.classifier_transformer(self.transformer_decoder(tgt=tgt, memory=memory))[:, -1, :]
            # # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            # next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(decoder_inputs, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = (next_tokens / vocab_size).long()
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                decoder_inputs,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            decoder_inputs = torch.cat([decoder_inputs[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            # )
            # if model_kwargs["past"] is not None:
            #     model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(decoder_inputs, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            decoder_inputs,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        num_return_sequences = beam_scorer.num_beam_hyps_to_keep
        # return only as many indices as sequences
        beam_indices = tuple(
            (beam_indices[i * num_beams: i * num_beams + num_return_sequences] for i in range(batch_size))
        )
        beam_indices = sum(beam_indices, ())

        return BeamSearchEncoderDecoderOutput(
            sequences=sequence_outputs["sequences"],
            sequences_scores=sequence_outputs["sequence_scores"],
            scores=scores
        )

    def _greedy_search(
            self,
            decoder_inputs: torch.LongTensor,
            memory: torch.FloatTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            pad_token_id: int = None,
            eos_token_id: int = None,
    ) -> GreedySearchOutput:
        # init values
        # init attention / hidden states / scores tuples
        scores = ()

        # keep track of which sequences are already finished
        unfinished_sequences = decoder_inputs.new(decoder_inputs.shape[0]).fill_(1)
        cur_len = decoder_inputs.shape[-1]

        while True:
            # prepare model inputs
            tgt = self.transformer_decoder_pos(self.transformer_decoder_embedding(decoder_inputs))
            next_token_logits = self.classifier_transformer(self.transformer_decoder(tgt=tgt, memory=memory))[:, -1, :]
            scores += (next_token_logits,)

            # pre-process distribution
            next_tokens_scores = logits_processor(decoder_inputs, next_token_logits)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            decoder_inputs = torch.cat([decoder_inputs, next_tokens[:, None]], dim=-1)
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(decoder_inputs, scores):
                break

        return GreedySearchEncoderDecoderOutput(
            sequences=decoder_inputs,
            scores=scores
        )

    def _sample(
            self,
            decoder_inputs: torch.LongTensor,
            memory: torch.FloatTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None
    ) -> SampleOutput:
        # init attention / hidden states / scores tuples
        scores = ()

        # keep track of which sequences are already finished
        unfinished_sequences = decoder_inputs.new(decoder_inputs.shape[0]).fill_(1)
        cur_len = decoder_inputs.shape[-1]

        # auto-regressive generation
        while True:
            tgt = self.transformer_decoder_pos(self.transformer_decoder_embedding(decoder_inputs))
            next_token_logits = self.classifier_transformer(self.transformer_decoder(tgt=tgt, memory=memory))[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(decoder_inputs, next_token_logits)
            next_token_scores = logits_warper(decoder_inputs, next_token_scores)

            # Store scores, attentions and hidden_states when required
            scores += (next_token_scores,)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            decoder_inputs = torch.cat([decoder_inputs, next_tokens[:, None]], dim=-1)
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(decoder_inputs, scores):
                break

        return SampleEncoderDecoderOutput(
            sequences=decoder_inputs,
            scores=scores
        )

    def _beam_sample(
            self,
            decoder_inputs: torch.LongTensor,
            memory: torch.FloatTensor,
            beam_scorer: BeamScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None
    ) -> BeamSampleOutput:
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = decoder_inputs.shape

        # init attention / hidden states / scores tuples
        scores = ()
        beam_indices = (tuple(() for _ in range(batch_beam_size)))

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=decoder_inputs.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:
            tgt = self.transformer_decoder_pos(self.transformer_decoder_embedding(decoder_inputs))
            next_token_logits = self.classifier_transformer(self.transformer_decoder(tgt=tgt, memory=memory))[:, -1, :]

            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(decoder_inputs, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = logits_warper(decoder_inputs, next_token_scores)

            # Store scores, attentions and hidden_states when required
            scores += (logits_warper(decoder_inputs, next_token_scores_processed),)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = nn.functional.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                decoder_inputs,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            decoder_inputs = torch.cat([decoder_inputs[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(decoder_inputs, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            decoder_inputs,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        num_return_sequences = beam_scorer.num_beam_hyps_to_keep
        # return only as many indices as sequences
        beam_indices = tuple(
            (beam_indices[i * num_beams: i * num_beams + num_return_sequences] for i in range(batch_size))
        )
        beam_indices = sum(beam_indices, ())

        return BeamSampleEncoderDecoderOutput(
            sequences=sequence_outputs["sequences"],
            sequences_scores=sequence_outputs["sequence_scores"],
            scores=scores
        )

    def _group_beam_search(
            self,
            decoder_inputs: torch.LongTensor,
            memory: torch.FloatTensor,
            beam_scorer: BeamScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None
    ):
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        device = decoder_inputs.device

        batch_beam_size, cur_len = decoder_inputs.shape

        beam_indices = [tuple(() for _ in range(num_sub_beams * batch_size)) for _ in range(num_beam_groups)]

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = ()

        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:
            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=decoder_inputs.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            # do one decoder step on all beams of all sentences in batch
            tgt = self.transformer_decoder_pos(self.transformer_decoder_embedding(decoder_inputs))
            logits = self.classifier_transformer(self.transformer_decoder(tgt=tgt, memory=memory))

            processed_score = torch.zeros_like(logits[:, -1, :])

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )
                group_input_ids = decoder_inputs[batch_group_indices]

                # select outputs of beams of current group only
                next_token_logits = logits[batch_group_indices, -1, :]

                next_token_scores = nn.functional.log_softmax(
                    next_token_logits, dim=-1
                )  # (batch_size * group_size, vocab_size)
                vocab_size = next_token_scores.shape[-1]

                next_token_scores_processed = logits_processor(
                    group_input_ids, next_token_scores, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )
                next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

                processed_score[batch_group_indices] = next_token_scores_processed

                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = torch_int_div(next_tokens, vocab_size)
                next_tokens = next_tokens % vocab_size

                # stateless
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                beam_indices[beam_group_idx] = tuple(
                    beam_indices[beam_group_idx][beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices[0]))
                )

                decoder_inputs[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                        num_beams * torch_int_div(beam_idx, group_size) + group_start_idx + (beam_idx % group_size)
                )

            # Store scores, attentions and hidden_states when required
            scores += (processed_score,)

            decoder_inputs = torch.cat([decoder_inputs, current_tokens.unsqueeze(-1)], dim=-1)

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(decoder_inputs, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            decoder_inputs,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        beam_indices = sum(beam_indices, ())
        num_return_sequences = beam_scorer.num_beam_hyps_to_keep
        # return only as many indices as sequences
        beam_indices = tuple(
            (beam_indices[i * num_beams: i * num_beams + num_return_sequences] for i in range(batch_size))
        )
        beam_indices = sum(beam_indices, ())

        return BeamSearchEncoderDecoderOutput(
            sequences=sequence_outputs["sequences"],
            sequences_scores=sequence_outputs["sequence_scores"],
            scores=scores
        )

    def _get_stopping_criteria(
            self, max_length: Optional[int], max_time: Optional[float], stopping_criteria: Optional[StoppingCriteriaList]
    ) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if max_length is not None:
            criteria.append(MaxLengthCriteria(max_length=max_length))
        if max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=max_time))
        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        return criteria

    def _merge_criteria_processor_list(
            self,
            default_list: Union[LogitsProcessorList, StoppingCriteriaList],
            custom_list: Union[LogitsProcessorList, StoppingCriteriaList],
    ) -> Union[LogitsProcessorList, StoppingCriteriaList]:
        if len(custom_list) == 0:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "logits processor"
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to `generate`, "
                        f"but it has already been created with the values {default}. {default} has been created by passing the "
                        "corresponding arguments to generate or by the model's config default values. "
                        f"If you just want to change the default values of {object_type} consider passing them as arguments "
                        f"to `generate` instead of using a custom {object_type}."
                    )
        default_list.extend(custom_list)
        return default_list

    def _get_logits_processor(
            self,
            forced_eos_token_id: int,
            repetition_penalty: float,
            no_repeat_ngram_size: int,
            bad_words_ids: Optional[List[List[int]]],
            min_length: int,
            max_length: int,
            eos_token_id: int,
            prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
            num_beams: int,
            num_beam_groups: int,
            diversity_penalty: float,
            remove_invalid_values: bool,
            logits_processor: Optional[LogitsProcessorList],
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        processors = LogitsProcessorList()
        # instantiate processors list

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if diversity_penalty is not None and diversity_penalty > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
                )
            )
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if bad_words_ids is not None:
            processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
        if prefix_allowed_tokens_fn is not None:
            processors.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams // num_beam_groups))
        if forced_eos_token_id is not None:
            processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
        if remove_invalid_values is True:
            processors.append(InfNanRemoveLogitsProcessor())
        processors = self._merge_criteria_processor_list(processors, logits_processor)
        return processors

    @staticmethod
    def _expand_inputs_for_generation(
            input_ids: torch.LongTensor,
            memory: torch.FloatTensor,
            expand_size: int = 1
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx).to(input_ids.device)
        memory = memory.index_select(0, expanded_return_idx.to(memory.device))
        return input_ids, memory

    def _prepare_decoder_input_ids_for_generation(
            self,
            device: torch.device,
            batch_size: int,
            bos_token_id: int = None,
    ) -> torch.LongTensor:
        return torch.ones((batch_size, 1), dtype=torch.long, device=device) * bos_token_id

    def _get_logits_warper(
            self,
            top_k: int,
            top_p: float,
            typical_p: float,
            temperature: float,
            num_beams: int,
    ) -> LogitsProcessorList:
        # instantiate warpers list
        warpers = LogitsProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        if typical_p is not None and typical_p < 1.0:
            # TODO from master in huggingface, currently not supported
            raise NotImplementedError()
            # warpers.append(TypicalLogitsWarper(mass=typical_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        return warpers

    def generate(
            self,
            memory: torch.FloatTensor,
            max_length: Optional[int] = None,
            min_length: int = 10,
            do_sample: bool = False,
            early_stopping: bool = False,
            num_beams: int = 1,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 1.0,
            typical_p: Optional[float] = None,
            repetition_penalty: float = 1.0,
            bad_words_ids: Optional[Iterable[int]] = None,
            length_penalty: float = 1.0,
            no_repeat_ngram_size: int = 0,
            num_return_sequences: int = 1,
            max_time: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
            num_beam_groups: int = 1,
            diversity_penalty: float = 0.0,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
            stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
            constraints: Optional[List[Constraint]] = None,
            remove_invalid_values: bool = False
    ) -> Union[BeamSearchOutput, GreedySearchOutput, SampleOutput, BeamSampleOutput]:
        # 1. Set generation parameters if not already defined
        bos_token_id = 1
        pad_token_id = 0
        eos_token_id = 2

        batch_size = memory.shape[0]

        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        input_ids = self._prepare_decoder_input_ids_for_generation(
            memory.device,
            batch_size,
            bos_token_id=bos_token_id
        )

        # 5. Prepare `max_length` depending on other stopping criteria
        # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
        if max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + input_ids.shape[-1]
        elif max_length is not None and max_new_tokens is not None:
            # Both are set, this is odd, raise a warning
            warnings.warn(
                "Both `max_length` and `max_new_tokens` have been set "
                f"but they serve the same purpose. `max_length` {max_length} "
                f"will take priority over `max_new_tokens` {max_new_tokens}.",
                UserWarning,
            )
        elif max_length is not None and max_new_tokens is None:
            pass
        else:
            raise ValueError('мда треш')

        # 6. determine generation mode
        is_constraint_gen_mode = constraints is not None
        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
        is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
        is_beam_sample_gen_mode = (
                (num_beams > 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
        )
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1) and constraints is None

        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # 7. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_eos_token_id=eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            logits_processor=logits_processor,
        )

        # 8. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
        )

        # 9. go into different generation modes
        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # 10. run greedy search
            return self._greedy_search(
                decoder_inputs=input_ids,
                memory=memory,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id
            )

        elif is_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, typical_p=typical_p, temperature=temperature, num_beams=num_beams
            )

            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, memory = self._expand_inputs_for_generation(input_ids, memory, expand_size=num_beams)

            # 12. run sample
            return self._sample(
                decoder_inputs=input_ids,
                memory=memory,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id
            )

        elif is_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=memory.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, memory = self._expand_inputs_for_generation(input_ids, memory, expand_size=num_beams)
            # 12. run beam search
            return self._beam_search(
                decoder_inputs=input_ids,
                memory=memory,
                beam_scorer=beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id
            )

        elif is_beam_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, typical_p=typical_p, temperature=temperature, num_beams=num_beams
            )

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size * num_return_sequences,
                num_beams=num_beams,
                device=memory.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
            )

            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, memory = self._expand_inputs_for_generation(input_ids, memory, expand_size=num_beams * num_return_sequences)

            # 13. run beam sample
            return self._beam_sample(
                decoder_inputs=input_ids,
                memory=memory,
                beam_scorer=beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

        elif is_group_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if num_beams % num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                max_length=stopping_criteria.max_length,
                device=memory.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
                num_beam_groups=num_beam_groups,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, memory = self._expand_inputs_for_generation(input_ids, memory, expand_size=num_beams)
            # 12. run beam search
            return self._group_beam_search(
                decoder_inputs=input_ids,
                memory=memory,
                beam_scorer=beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

        elif is_constraint_gen_mode:
            raise NotImplementedError()
