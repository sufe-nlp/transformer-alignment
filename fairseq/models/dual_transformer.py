# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
from itertools import accumulate
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import register_model, register_model_architecture
from .fairseq_model import BaseFairseqModel
from fairseq.models.transformer import (
    Embedding,
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
)

import random

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('dual_transformer')
class DualTransformerModel(BaseFairseqModel):
    def __init__(self, args, modelf, modelb):
        super().__init__()
        self.args = args
        self.modelf = modelf
        self.modelb = modelb
        self.pad = 1
        self.eos = 2

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        
        # fmt: on
        
    @classmethod
    def build_submodel(cls, args, task, reverse=False):
        dual_transformer_small(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, 'max_source_positions', None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        if not reverse:
            src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        else:
            args.source_lang, args.target_lang = args.target_lang, args.source_lang
            tgt_dict, src_dict  = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens, src_dict=src_dict)
        return TransformerModel(args, encoder, decoder)

    @classmethod
    def build_model(cls, args, task):
        # import pdb;pdb.set_trace()
        modelf = cls.build_submodel(args, task)
        modelb = cls.build_submodel(args, task, reverse=True)
        args.source_lang, args.target_lang = args.target_lang, args.source_lang
        return cls(args, modelf, modelb)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, src_dict=None):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False),
            src_dict=src_dict,
        )
    
    def forward(self, sample, **kwargs):
        src_tokens = sample['net_input']['src_tokens']
        src_lengths = sample['net_input']['src_lengths']
        prev_output_tokens = sample['net_input']['prev_output_tokens']
        f_decoder_out = self.modelf(src_tokens, src_lengths, prev_output_tokens)

        src_tokens = sample['dual_reverse']['src_tokens']
        src_lengths = sample['dual_reverse']['src_lengths']
        prev_output_tokens = sample['dual_reverse']['prev_output_tokens']
        b_decoder_out = self.modelb(src_tokens, src_lengths, prev_output_tokens)

        return f_decoder_out, b_decoder_out

    def get_normalized_probs(self, decoder_out):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = decoder_out[0].float() 
        return F.log_softmax(logits, dim=-1)
    
    def extract_merge_then_align_alignment(self, sample, net_output):
        f_net_output, b_net_output = net_output
        f_attn = f_net_output[1]['suphead_attn'][:,1:,:-1]
        b_attn = b_net_output[1]['suphead_attn'][:,1:,:-1].transpose(1,2)

        # f_attn = f_net_output[1]['attn'][:,1:,:-1]
        # b_attn = b_net_output[1]['attn'][:,1:,:-1].transpose(1,2)
            
        tgt_sent=sample['dual_reverse']['prev_output_tokens'][:,1:]
        src_sent=sample['dual_reverse']['src_tokens'][:,:-1]    
        src_valid = ((src_sent != self.pad) & (src_sent != self.eos)).unsqueeze(dim=1).float()
        tgt_valid = ((tgt_sent != self.pad) & (tgt_sent != self.eos)).unsqueeze(dim=-1).float()
        b_mask = torch.einsum('bik,bkj->bij',tgt_valid,src_valid).bool().transpose(1,2)
        b_attn = torch.masked_select(b_attn, b_mask)

        tgt_sent=sample['net_input']['prev_output_tokens'][:,1:]
        src_sent=sample['net_input']['src_tokens'][:,:-1]    
        src_valid = ((src_sent != self.pad) & (src_sent != self.eos)).unsqueeze(dim=1).float()
        tgt_valid = ((tgt_sent != self.pad) & (tgt_sent != self.eos)).unsqueeze(dim=-1).float()
        f_mask = torch.einsum('bik,bkj->bij',tgt_valid,src_valid).bool()
        f_attn = torch.masked_select(f_attn, f_mask)

        attn = torch.zeros_like(f_net_output[1]['attn'][:,1:,:-1])
        attn[f_mask] = f_attn * b_attn 
        bsz, _, srclen = attn.size()
        alignments = {}
        src_length = (src_sent != self.pad).sum(dim=-1)
        tgt_length = (tgt_sent != self.pad).sum(dim=-1)
        src_valid = (src_sent != self.pad).int()
        tgt_valid = (tgt_sent != self.pad).int()
        for i in range(bsz):
            id = str(int(sample['id'][i]))
            k_size = min(src_length[i],tgt_length[i]) 
            _, index = torch.topk(attn[i].flatten(), k=k_size)
            row, column = index / srclen, index % srclen
            res = []
            for x,y in zip(row,column):
                bpe_tgt = int(tgt_valid[i,:x+1].sum()) - 1
                bpe_src = int(src_valid[i,:y+1].sum()) - 1
                assert bpe_tgt > -1 and bpe_src > -1
                res.append((bpe_src,bpe_tgt))
            res = sorted(res,key=lambda x: x[0], reverse=False)
            alignments[id] = ' '.join([str(x)+'-'+str(y) for x,y in res])
            ## alignments are all 0-indexed with src-tgt format.
        return alignments

    def extract_align_then_merge_alignment(self, sample, net_output, src_punc_tokens, reverse=False):
        f_net_output, b_net_output = net_output
        f_attn = f_net_output[1]['suphead_attn'][:,1:,:-1]
        b_attn = b_net_output[1]['suphead_attn'][:,1:,:-1]

        # f_attn = f_net_output[1]['attn'][:,1:,:-1]
        # b_attn = b_net_output[1]['attn'][:,1:,:-1]

        if not reverse:
            attn = f_attn 
            tgt_sent=sample['net_input']['prev_output_tokens'][:,1:]
            src_sent=sample['net_input']['src_tokens'][:,:-1]  
        else:
            attn = b_attn 
            tgt_sent=sample['dual_reverse']['prev_output_tokens'][:,1:]
            src_sent=sample['dual_reverse']['src_tokens'][:,:-1]     

        def get_token_to_word_mapping(tokens, exclude_list):
            n = len(tokens)
            word_start = [int(token not in exclude_list) for token in tokens]
            word_idx = list(accumulate(word_start))
            token_to_word = {i: word_idx[i] for i in range(n)}
            return token_to_word

        # import pdb;pdb.set_trace()
        alignments = {} ## is 0-index 
        for idx, sample_id in enumerate(sample['id'].tolist()):
            alignments[sample_id] = []
            tgt_valid = (tgt_sent[idx] != self.pad).nonzero().squeeze(dim=-1) 
            src_invalid = (src_sent[idx] == self.pad).nonzero().squeeze(dim=-1)           
            src_token_to_word = get_token_to_word_mapping(src_sent[idx], [self.pad])
            tgt_token_to_word = get_token_to_word_mapping(tgt_sent[idx], [self.pad])

            if len(tgt_valid) != 0 and len(src_invalid) < len(src_sent[idx]):  
                attn_valid = attn[idx,tgt_valid,:]
                if src_sent[idx,-1] in src_punc_tokens:
                    if tgt_sent[idx,tgt_valid[-1]] in src_punc_tokens:
                        attn_valid[:-1, -1] = float('-inf')   
                        attn_valid[ -1, -1] = float('+inf')
                    else:
                        attn_valid[:, -1] = float('-inf') 

                attn_valid[:, src_invalid] = float('-inf')
                _, src_indices = attn_valid.max(dim=1) 
                for tgt_idx, src_idx in zip(tgt_valid, src_indices):
                    src_align=src_token_to_word[src_idx.item()] - 1 
                    tgt_align=tgt_token_to_word[tgt_idx.item()] - 1
                    align_str=str(src_align)+'-'+str(tgt_align)
                    alignments[sample_id].append(align_str)
                alignments[sample_id] = ' '.join(alignments[sample_id])
            ## alignments are all 0-indexed with src-tgt format.
        return alignments

@register_model_architecture('dual_transformer', 'dual_transformer_base')
def dual_transformer_base(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.no_cross_attention = getattr(args, 'no_cross_attention', False)
    args.cross_self_attention = getattr(args, 'cross_self_attention', False)
    args.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', False)


@register_model_architecture('dual_transformer', 'dual_transformer_small')
def dual_transformer_small(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    dual_transformer_base(args)

