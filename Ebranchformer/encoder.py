import torch
import logging
import yaml
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from typeguard import typechecked

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from modules.nets_utils import make_pad_mask, get_activation, Swish
from modules.subsampling import Conv2dSubsampling
from modules.embedding import RelPositionalEncoding
from modules.attention import RelPositionMultiHeadedAttention
from modules.cgmlp import ConvolutionalGatingMLP
from modules.repeat import repeat
from modules.layer_norm import LayerNorm

with open("/home/rndlwjs/qhdd14/hdd14/kyujin/241125_asr_project/libris_asr/main/branchformer/train_asr_e_branchformer_size256_mlp1024_linear1024_e12_mactrue_edrop0.0_ddrop0.0.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)['encoder_conf']

class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class EBranchformerEncoderLayer(torch.nn.Module):
    """E-Branchformer encoder layer module.

    Args:
        size (int): model dimension
        attn: standard self-attention or efficient attention
        cgmlp: ConvolutionalGatingMLP
        feed_forward: feed-forward module, optional
        feed_forward: macaron-style feed-forward module, optional
        dropout_rate (float): dropout probability
        merge_conv_kernel (int): kernel size of the depth-wise conv in merge module
    """

    def __init__(
        self,
        size: int,
        attn: torch.nn.Module,
        cgmlp: torch.nn.Module,
        feed_forward: Optional[torch.nn.Module],
        feed_forward_macaron: Optional[torch.nn.Module],
        dropout_rate: float,
        merge_conv_kernel: int = 3,
    ):
        super().__init__()

        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp

        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.ff_scale = 1.0
        if self.feed_forward is not None:
            self.norm_ff = LayerNorm(size)
        if self.feed_forward_macaron is not None:
            self.ff_scale = 0.5
            self.norm_ff_macaron = LayerNorm(size)

        self.norm_mha = LayerNorm(size)  # for the MHA module
        self.norm_mlp = LayerNorm(size)  # for the MLP module
        self.norm_final = LayerNorm(size)  # for the final output of the block

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.depthwise_conv_fusion = torch.nn.Conv1d(
            size + size,
            size + size,
            kernel_size=merge_conv_kernel,
            stride=1,
            padding=(merge_conv_kernel - 1) // 2,
            groups=size + size,
            bias=True,
        )
        self.merge_proj = torch.nn.Linear(size + size, size)

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
        """

        if cache is not None:
            raise NotImplementedError("cache is not None, which is not tested")

        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        if self.feed_forward_macaron is not None:
            residual = x
            x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))

        # Two branches
        x1 = x
        x2 = x

        # Branch 1: multi-headed attention module
        x1 = self.norm_mha(x1)

        #if isinstance(self.attn, FastSelfAttention):
        #    x_att = self.attn(x1, mask)
        if True: #else:
            if pos_emb is not None:
                x_att = self.attn(x1, x1, x1, pos_emb, mask)
            else:
                x_att = self.attn(x1, x1, x1, mask)

        x1 = self.dropout(x_att)

        # Branch 2: convolutional gating mlp
        x2 = self.norm_mlp(x2)

        if pos_emb is not None:
            x2 = (x2, pos_emb)
        x2 = self.cgmlp(x2, mask)
        if isinstance(x2, tuple):
            x2 = x2[0]

        x2 = self.dropout(x2)

        # Merge two branches
        x_concat = torch.cat([x1, x2], dim=-1)
        x_tmp = x_concat.transpose(1, 2)
        x_tmp = self.depthwise_conv_fusion(x_tmp)
        x_tmp = x_tmp.transpose(1, 2)
        x = x + self.dropout(self.merge_proj(x_concat + x_tmp))

        if self.feed_forward is not None:
            # feed forward module
            residual = x
            x = self.norm_ff(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        x = self.norm_final(x)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask

class EBranchformerEncoder(torch.nn.Module):
    """E-Branchformer encoder module."""

    #@typechecked
    def __init__(
        self,
        input_size: int = 80,
        output_size: int = 256,
        attention_heads: int = 4,
        attention_layer_type: str = "rel_selfattn",
        pos_enc_layer_type: str = "rel_pos",
        rel_pos_type: str = "latest",
        cgmlp_linear_units: int = 2048,
        cgmlp_conv_kernel: int = 31,
        use_linear_after_conv: bool = False,
        gate_activation: str = "identity",
        num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        zero_triu: bool = False,
        padding_idx: int = -1,
        layer_drop_rate: float = 0.0,
        max_pos_emb_len: int = 5000,
        use_ffn: bool = False,
        macaron_ffn: bool = False,
        ffn_activation_type: str = "swish",
        linear_units: int = 2048,
        positionwise_layer_type: str = "linear",
        merge_conv_kernel: int = 3,
        interctc_layer_idx=None,
        interctc_use_conditioning: bool = False,
        qk_norm: bool = False,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self._output_size = output_size

        ### RelPosEnc
        self.pos_enc_class = RelPositionalEncoding

        ### Conv2dSubsampling
        self.embed = Conv2dSubsampling(
            idim=80, 
            odim=config['output_size'], 
            dropout_rate=['dropout_rate'], 
            pos_enc=self.pos_enc_class(d_model=config['output_size'], dropout_rate=config['positional_dropout_rate'], max_len=5000)
            )

        self.activation = get_activation(ffn_activation_type)

        ### MHSA
        self.encoder_selfattn_layer = RelPositionMultiHeadedAttention
        self.encoder_selfattn_layer_args = (config['attention_heads'], config['output_size'], config['dropout_rate'], False) #n_head, n_feat, dropout_rate, zero_triu=False

        ### CGMLP
        self.cgmlp_layer = ConvolutionalGatingMLP
        self.cgmlp_layer_args = (config['output_size'], config['cgmlp_linear_units'], config['cgmlp_conv_kernel'], config['dropout_rate'], False, config['gate_activation'],)

        self.positionwise_layer = PositionwiseFeedForward
        self.positionwise_layer_args = (output_size, linear_units, dropout_rate, self.activation,)

        self.encoders = repeat(
            config['num_blocks'],
            lambda lnum: EBranchformerEncoderLayer(
                size=config['output_size'],
                attn=self.encoder_selfattn_layer(*self.encoder_selfattn_layer_args),
                cgmlp=self.cgmlp_layer(*self.cgmlp_layer_args),
                feed_forward=self.positionwise_layer(*self.positionwise_layer_args) if use_ffn else None,
                feed_forward_macaron=(
                    self.positionwise_layer(*self.positionwise_layer_args)
                    if use_ffn and macaron_ffn
                    else None
                ),
                dropout_rate=config['dropout_rate'],
                merge_conv_kernel=config['merge_conv_kernel'],
            ),
            config['layer_drop_rate'],
        )
        self.after_norm = LayerNorm(output_size)

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc=None,
        max_layer: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.
            ctc (CTC): Intermediate CTC module.
            max_layer (int): Layer depth below which InterCTC is applied.
        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.
        """

        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        xs_pad, masks = self.embed(xs_pad, masks)

        xs_pad, masks = self.encoders(xs_pad, masks)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]

        xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1)

        return xs_pad, xs_pad, olens #encoder_log_probs, outputs, output_lengths
        #return xs_pad, olens, None