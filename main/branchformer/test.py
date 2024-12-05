import yaml
import torch
from branchformer.nets_utils import make_pad_mask, get_activation, Swish
from branchformer.subsampling import Conv2dSubsampling
from branchformer.embedding import RelPositionalEncoding
from branchformer.eb import EBranchformerEncoderLayer
from branchformer.attention import RelPositionMultiHeadedAttention
from branchformer.cgmlp import ConvolutionalGatingMLP
from branchformer.repeat import repeat
from branchformer.layer_norm import LayerNorm

### Config

path = 'train_asr_e_branchformer_size256_mlp1024_linear1024_e12_mactrue_edrop0.0_ddrop0.0.yaml'
with open(path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)['encoder_conf']

### start here

xs_pad = torch.rand(1, 80, 200)
xs_pad = xs_pad.transpose(1, 2) #입력은 [B, T, F] 이다.
ilens = torch.LongTensor([200])

masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

### RelPosEnc
pos_enc_class = RelPositionalEncoding

### subsampling

embed = Conv2dSubsampling(idim=80, 
                          odim=config['output_size'], 
                          dropout_rate=['dropout_rate'], 
                          pos_enc=pos_enc_class(d_model=config['output_size'], dropout_rate=config['positional_dropout_rate'], max_len=5000)
                          )

xs_pad, masks = embed(xs_pad, masks)

### MHSA
encoder_selfattn_layer = RelPositionMultiHeadedAttention
encoder_selfattn_layer_args = (config['attention_heads'], config['output_size'], config['dropout_rate'], False) #n_head, n_feat, dropout_rate, zero_triu=False

### CGMLP
cgmlp_layer = ConvolutionalGatingMLP
cgmlp_layer_args = (config['output_size'], config['cgmlp_linear_units'], config['cgmlp_conv_kernel'], config['dropout_rate'], False, config['gate_activation'],)

### FFN
class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

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

positionwise_layer = PositionwiseFeedForward
positionwise_layer_args = (config['output_size'], config['linear_units'], config['positional_dropout_rate'], get_activation('swish'))

### encoder
use_ffn = config['use_ffn']
macaron_ffn = config['macaron_ffn']

encoders = repeat(
    config['num_blocks'],
    lambda lnum: EBranchformerEncoderLayer(
        size=config['output_size'],
        attn=encoder_selfattn_layer(*encoder_selfattn_layer_args),
        cgmlp=cgmlp_layer(*cgmlp_layer_args),
        feed_forward=positionwise_layer(*positionwise_layer_args) if use_ffn else None,
        feed_forward_macaron=(
            positionwise_layer(*positionwise_layer_args)
            if use_ffn and macaron_ffn
            else None
        ),
        dropout_rate=config['dropout_rate'],
        merge_conv_kernel=config['merge_conv_kernel'],
    ),
    config['layer_drop_rate'],
)

from pytorch_lightning.utilities.model_summary import LayerSummary

param = LayerSummary(encoders).num_parameters / 1000000
print(param); exit()

xs_pad, masks = encoders(xs_pad, masks)

if isinstance(xs_pad, tuple):
    xs_pad = xs_pad[0]

### LayerNorm
LayerNorm = LayerNorm(256)

xs_pad = LayerNorm(xs_pad)
olens = masks.squeeze(1).sum(1)

print(xs_pad.shape, olens.shape)
print(olens)
