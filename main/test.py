from pytorch_lightning.utilities.model_summary import LayerSummary
from models.encoder import ConformerEncoder, ConformerBlock

encoder = ConformerEncoder(
    num_classes=30,
    input_dim=80,
    encoder_dim=256,
    num_layers=1,
    num_attention_heads=4,
    feed_forward_expansion_factor=1024,
    conv_expansion_factor=2,
    input_dropout_p=0.1,
    feed_forward_dropout_p=0.1,
    attention_dropout_p=0.1,
    conv_dropout_p=0.1,
    conv_kernel_size=31,
    half_step_residual=True,
    joint_ctc_attention=True,
    )

q = ConformerBlock(
    encoder_dim=256,
    num_attention_heads=4,
    feed_forward_expansion_factor=1024,
    conv_expansion_factor=2,
    feed_forward_dropout_p=0.1,
    attention_dropout_p=0.1,
    conv_dropout_p=0.1,
    conv_kernel_size=31,
    half_step_residual=True,
    )

def msize(m):
    print(LayerSummary(m).num_parameters / 1000000)

import torch
from models.modules import Linear
msize(torch.nn.Linear(512, 512*4))
q = Linear
msize(Linear(512, 512))