# E-Branchformer: Transducer

import yaml
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.text import CharErrorRate as cer
from itertools import groupby
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
from model import Ebranchformer

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from transform import TextTransform

TextTransform = TextTransform()

class EBranchformerEncoderASR(pl.LightningModule):
    def __init__(self,):

        super().__init__()
        self.model = Ebranchformer()
        self.criterion = T.RNNTLoss(blank=0) #torch.nn.CTCLoss(blank=0)

        self.cer_metric = cer()
        self.decoded_preds = []
        self.decoded_targets = []
        self.cer = []
        
    def get_batch(self, batch):
        inputs, input_lengths, targets, target_lengths = batch

        batch_size = inputs.size(0)

        zeros = torch.zeros((batch_size, 1)).to(device=self.device)
        compute_targets = torch.cat((zeros, targets), dim=1).to(
            device=self.device, dtype=torch.int
        )
        compute_target_lengths = (target_lengths + 1).to(device=self.device)

        return (inputs, input_lengths, targets, target_lengths, compute_targets, compute_target_lengths,)

    def forward(self, inputs, input_lengths):
        # spec [B, T, F]    
        return self.model.recognize(inputs, input_lengths)

    def training_step(self, batch, batch_idx):
        inputs, labels, input_lengths, label_lengths, compute_targets, compute_target_lenths = self.get_batch(batch)

        outputs, output_lengths = self.model(inputs, input_lengths, compute_targets, compute_target_lengths)
        loss = self.criterion(outputs, targets, output_lengths, target_lengths)

        metrics = {'train_loss': loss}
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        (inputs, input_lengths, targets, target_lengths, compute_targets, compute_target_lengths,) = self.get_batch(batch)

        outputs, output_lengths = self.conformer(inputs, input_lengths, compute_targets, compute_target_lengths)

        loss = self.criterion(outputs, targets, output_lengths, target_lengths)

        predicts = self.forward(inputs, input_lengths)
        if batch_idx == 0:
            print(predicts)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=[0.9, 0.98], weight_decay=1e-3)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=4000, num_training_steps=100000)
        
        sch_config = {
                'scheduler': scheduler,
                'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheulder':sch_config,}