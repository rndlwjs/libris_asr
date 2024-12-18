import torch
from torch import nn, Tensor
from encoder import EBranchformerEncoder
from decoder import DecoderRNNT

class EbranchformerASR(nn.Module):
    def __init__(self,):
        super().__init__()
        self.encoder = EBranchformerEncoder(
            input_size=80,
            output_size=256,
            attention_heads=4,
            attention_layer_type="rel_selfattn",
            pos_enc_layer_type="rel_pos",
            rel_pos_type="latest",
            cgmlp_linear_units=1024,
            cgmlp_conv_kernel=31,
            use_linear_after_conv=False,
            gate_activation="identity",
            num_blocks=12,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            input_layer="conv2d",
            zero_triu=False,
            padding_idx=-1,
            layer_drop_rate=0.0,
            max_pos_emb_len=5000,
            use_ffn=True,
            macaron_ffn=True,
            ffn_activation_type="swish",
            linear_units=1024,
            positionwise_layer_type="linear",
            merge_conv_kernel=3,
            interctc_layer_idx=None,
            interctc_use_conditioning=False,
            qk_norm=False,
            use_flash_attn=True,)

        self.decoder = DecoderRNNT(
            num_classes=27,
            hidden_state_dim=320,
            output_dim=256,
            num_layers=1,
            rnn_type='lstm',
            sos_id=1,
            eos_id=2,
            dropout_p=0.1,)

        self.fc = nn.Sequential(
            nn.Linear(256 << 1, 256),
            nn.Tanh(),
            nn.Linear(256, 27, bias=False),
        )
    
    def joint(self, encoder_outputs, decoder_outputs):
        if encoder_outputs.dim() == 3 and decoder_outputs.dim() == 3:
            input_length = encoder_outputs.size(1)
            target_length = decoder_outputs.size(1)

            encoder_outputs = encoder_outputs.unsqueeze(2)
            decoder_outputs = decoder_outputs.unsqueeze(1)

            encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])
            decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])

        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=-1)
        outputs = self.fc(outputs)

        return outputs

    def forward(self, inputs, input_lengths, targets, target_lengths):
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs, _ = self.decoder(targets, target_lengths)

        outputs = self.joint(encoder_outputs, decoder_outputs)

        return outputs, encoder_output_lengths

    @torch.no_grad()
    def decode(self, encoder_output: Tensor, max_length: int) -> Tensor:
        """
        Decode `encoder_outputs`.
        Args:
            encoder_output (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(seq_length, dimension)``
            max_length (int): max decoding time step
        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        pred_tokens, hidden_state = list(), None
        decoder_input = encoder_output.new_tensor([[1]], dtype=torch.long) ### replacing self.decoder.sos_id as 1

        for t in range(max_length):

            decoder_output, hidden_state = self.decoder(
                decoder_input, hidden_states=hidden_state
            )
            step_output = self.joint(
                encoder_output[t].view(-1), decoder_output.view(-1)
            )
            step_output = step_output.softmax(dim=0)
            pred_token = step_output.argmax(dim=0)
            pred_token = int(pred_token.item())
            pred_tokens.append(pred_token)
            decoder_input = step_output.new_tensor([[pred_token]], dtype=torch.long)

        return torch.LongTensor(pred_tokens)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor):
        """
        Recognize input speech. This method consists of the forward of the encoder and the decode() of the decoder.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        outputs = list()

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            decoded_seq = self.decode(encoder_output, max_length)
            outputs.append(decoded_seq)

        outputs = torch.stack(outputs, dim=1).transpose(0, 1)

        return outputs