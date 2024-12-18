import torch
import torchaudio
import torch.nn as nn
import pytorch_lightning as pl
from torch import Tensor
from typing import Dict, Union
from omegaconf import DictConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

import yaml
from eb import EBranchformerEncoder

encoder = EBranchformerEncoder(
    input_size=80,
    output_size=256,
    attention_heads=4,
    attention_layer_type="rel_selfattn",
    pos_enc_layer_type="rel_pos",
    rel_pos_type="latest",
    cgmlp_linear_units=2048,
    cgmlp_conv_kernel=31,
    use_linear_after_conv=False,
    gate_activation="identity",
    num_blocks=12,
    dropout_rate=0.1,
    positional_dropout_rate=0.1,
    attention_dropout_rate=0.0,
    input_layer="conv2d",
    zero_triu=False,
    padding_idx=-1,
    layer_drop_rate=0.0,
    max_pos_emb_len=5000,
    use_ffn=False,
    macaron_ffn=False,
    ffn_activation_type="swish",
    linear_units=2048,
    positionwise_layer_type="linear",
    merge_conv_kernel=3,
    interctc_layer_idx=None,
    interctc_use_conditioning=False,
    qk_norm=False,
    use_flash_attn=True,
    )

print(encoder)