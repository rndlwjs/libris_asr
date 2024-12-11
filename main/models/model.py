import torch
import torchaudio
import torch.nn as nn
import pytorch_lightning as pl
from torch import Tensor
from typing import Dict, Union
from omegaconf import DictConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from models.decoder import DecoderRNN
from models.encoder import ConformerEncoder

from models.joint_ctc_cross_entropy import JointCTCCrossEntropyLoss

from pytorch_lightning.loggers import TensorBoardLogger

from transform import TextTransform
#from lightning_asr.vocabs import LibriSpeechVocabulary
#from lightning_asr.vocabs.vocab import Vocabulary

TextTransform = TextTransform()

class ConformerLSTMModel(pl.LightningModule):
    def __init__(
            self,
            num_classes=30,
    ):

        super(ConformerLSTMModel, self).__init__()

        self.teacher_forcing_ratio = 1.0
        #self.vocab = vocab

        self.criterion = self.configure_criterion(
            num_classes,
            ignore_index=0, #self.vocab.pad_id,
            blank_id=3, #self.vocab.blank_id,
            ctc_weight=0.3,
            cross_entropy_weight=0.7,
        )
        self.encoder = ConformerEncoder(
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

        self.decoder = DecoderRNN(
            num_classes=30,
            max_length=578, #choose between 128 and 578 (max len in test-clean dataset)
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

        if batch_idx == 0:
            targets = targets[0].squeeze().cpu().detach().numpy()
            y_hats = y_hats[0].squeeze().cpu().detach().numpy()
            
            #print(targets.shape, y_hats.shape)
            print("target", TextTransform.int_to_text(targets))
            print("prediction", TextTransform.int_to_text(y_hats))


        self._log_states('train', loss, cross_entropy_loss, ctc_loss)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, targets, input_lengths, target_lengths = batch

        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs = self.decoder(encoder_outputs=encoder_outputs, teacher_forcing_ratio=0.0)

        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        outputs = outputs[:, :max_target_length, :]

        loss, ctc_loss, cross_entropy_loss = self.criterion(
            encoder_log_probs=encoder_log_probs.transpose(0, 1),
            decoder_log_probs=outputs.contiguous().view(-1, outputs.size(-1)),
            output_lengths=encoder_output_lengths,
            targets=targets[:, 1:],
            target_lengths=target_lengths,
        )

        y_hats = outputs.max(-1)[1]

        targets = targets.squeeze().cpu().detach().numpy()
        y_hats = y_hats.squeeze().cpu().detach().numpy()
        
        print("target", TextTransform.int_to_text(targets))
        print("prediction", TextTransform.int_to_text(y_hats))

        self._log_states('valid', loss, cross_entropy_loss, ctc_loss)

        return loss
    
    def test_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for test.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.FloatTensor): Loss for training.
        """
        inputs, targets = batch
        encoder_log_probs, encoder_outputs, encoder_output_lengths = self.encoder(inputs, inputs.size(1))
        outputs = self.decoder(encoder_outputs=encoder_outputs, teacher_forcing_ratio=0.0)

        max_target_length = targets.size(1) - 1  # minus the start of sequence symbol
        outputs = outputs[:, :max_target_length, :]

        y_hats = outputs.max(-1)[1]

        targets = targets.squeeze().cpu().detach().numpy()
        y_hats = y_hats.squeeze().cpu().detach().numpy()
        
        print("target", TextTransform.int_to_text(targets))
        print("prediction", TextTransform.int_to_text(y_hats))

        #cer = self.cer_metric(targets[:, 1:], y_hats)
    
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



'''
### Model Test
enc = ConformerEncoder(
    num_classes=30,
    input_dim=80,
    encoder_dim=256,
    num_layers=14,
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

dec = DecoderRNN(
    num_classes=30,
    max_length=128, #have to look
    hidden_state_dim=256,
    pad_id=0, #self.vocab.pad_id, #0, 1, 2
    sos_id=1, #self.vocab.sos_id,
    eos_id=2, #self.vocab.eos_id,
    num_heads=4,
    dropout_p=0.1,
    num_layers=2,
    rnn_type='lstm',
    use_tpu=False,
)



inp = torch.rand(1, 200, 80)
ilens = torch.LongTensor([200])

_, encoder_outputs, _ = enc(inp, ilens)
y_hats = dec(encoder_outputs=encoder_outputs, teacher_forcing_ratio=0.0)

print(y_hats)
'''