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
from branchformer.nets_utils import make_pad_mask, get_activation, Swish
from branchformer.repeat import repeat
from branchformer.layer_norm import LayerNorm
from branchformer.subsampling import Conv2dSubsampling
from branchformer.embedding import RelPositionalEncoding
from branchformer.eb import EBranchformerEncoder
from branchformer.attention import RelPositionMultiHeadedAttention
from branchformer.cgmlp import ConvolutionalGatingMLP

from models.joint_ctc_cross_entropy import JointCTCCrossEntropyLoss
from models.decoder import DecoderRNN
from pytorch_lightning.loggers import TensorBoardLogger


class BranchformerLSTMModel(pl.LightningModule):
    def __init__(
            self,
            num_classes=30,
    ):

        super(BranchformerLSTMModel, self).__init__()

        self.teacher_forcing_ratio = 1.0

        self.criterion = self.configure_criterion(
            num_classes,
            ignore_index=0,
            blank_id=3,
            ctc_weight=0.3,
            cross_entropy_weight=0.7,
        )

        if True:
            self.encoder = EBranchformerEncoder(
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

        if False:
            self.encoder = ConformerEncoder(
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

        self.decoder = DecoderRNN(
            num_classes=30,
            max_length=128, #have to look
            hidden_state_dim=256,
            pad_id=0, #self.vocab.pad_id, #0, 1, 2
            sos_id=1, #self.vocab.sos_id,
            eos_id=2, #self.vocab.eos_id,
            num_heads=4,
            dropout_p=0.1,
            num_layers=1,
            rnn_type='lstm',
            use_tpu=False,
        )

    def _log_states(
            self,
            stage: str,
            loss: float,
            cross_entropy_loss: float,
            ctc_loss: float,
    ) -> None:
        self.log(f"{stage}_loss", loss)
        self.log(f"{stage}_cross_entropy_loss", cross_entropy_loss)
        self.log(f"{stage}_ctc_loss", ctc_loss)
        return

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for inference.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * y_hats (torch.FloatTensor): Result of model predictions.
        """
        _, encoder_outputs, _ = self.encoder(inputs, input_lengths)
        y_hats = self.decoder(encoder_outputs=encoder_outputs, teacher_forcing_ratio=0.0)
        return y_hats

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        inputs, targets, input_lengths, target_lengths = batch

        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs = self.decoder(targets, encoder_outputs, teacher_forcing_ratio=self.teacher_forcing_ratio)

        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        outputs = outputs[:, :max_target_length, :]

        loss, ctc_loss, cross_entropy_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=outputs.contiguous().view(-1, outputs.size(-1)),
            output_lengths=encoder_output_lengths,
            targets=targets[:,1:],
            target_lengths=target_lengths,
        )

        y_hats = outputs.max(-1)[1]

        self._log_states('train', loss, cross_entropy_loss, ctc_loss)

        return loss

    def configure_optimizers(self) -> Dict[str, Union[torch.optim.Optimizer, object, str]]:

        optimizer = Adam(self.parameters(), lr=1e-04)
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=1, #self.configs.lr_patience,
            factor=0.3, #self.configs.lr_factor,
        )

        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'monitor': 'metric_to_track',
        }

    def configure_criterion(
            self,
            num_classes: int,
            ignore_index: int,
            blank_id: int,
            cross_entropy_weight: float,
            ctc_weight: float,
    ) -> nn.Module:
        """ Configure criterion """
        criterion = JointCTCCrossEntropyLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            reduction="mean",
            blank_id=blank_id,
            dim=-1,
            cross_entropy_weight=cross_entropy_weight,
            ctc_weight=ctc_weight,
        )
        return criterion