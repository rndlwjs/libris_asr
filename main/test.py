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

from models.convolution import (
    ConformerConvModule,
    Conv2dSubampling,
)
from models.modules import (
    ResidualConnectionModule,
    LayerNorm,
    Linear, Transpose,
)
from models.feed_forward import FeedForwardModule

encoder = ConformerEncoder(
    num_classes=30,
    input_dim=80,
    encoder_dim=256,
    num_layers=16,
    num_attention_heads=4,
    feed_forward_expansion_factor=4,
    conv_expansion_factor=2,
    input_dropout_p=0.1,
    feed_forward_dropout_p=0.1,
    attention_dropout_p=0.1,
    conv_dropout_p=0.1,
    conv_kernel_size=31,
    half_step_residual=True,
    joint_ctc_attention=True,
)
msize(encoder)