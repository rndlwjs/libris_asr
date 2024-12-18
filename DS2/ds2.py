# DeepSpeech2 CTC model

import torch
import torchaudio
import torch.nn as nn
from itertools import groupby
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torchmetrics.text import CharErrorRate as cer
import torch.nn.functional as F
from .model import DeepSpeech2
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from transform import TextTransform

TextTransform = TextTransform()
hparams = {
    "n_cnn_layers": 3,
    "n_rnn_layers": 5,
    "rnn_dim": 512,
    "n_class": 27,
    "n_feats": 128,
    "stride": 2,
    "dropout": 0.1,
    "learning_rate": 5e-4,
    "batch_size": 16,
    "epochs": 20,
    "total_files":28539,
}

def GreedyDecoder(output, labels, label_lengths, blank_label=0, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(TextTransform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index == blank_label:
                decode.append(index.item())
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        
        decode = [i[0] for i in groupby(decode)]
        decodes.append(TextTransform.int_to_text(decode))
    return decodes, targets

class DeepSpeech2ASR(pl.LightningModule):
    def __init__(self,):

        super(DeepSpeech2ASR, self).__init__()
        self.model = DeepSpeech2(
                        n_cnn_layers=hparams['n_cnn_layers'], 
                        n_rnn_layers=hparams['n_rnn_layers'], 
                        rnn_dim=hparams['rnn_dim'],
                        n_class=hparams['n_class'], 
                        n_feats=hparams['n_feats'], 
                        stride=hparams['stride'], 
                        dropout=hparams['dropout']
                        )
        self.criterion = torch.nn.CTCLoss(blank=0)
        self.cer_metric = cer()
        self.decoded_preds = []
        self.decoded_targets = []
        self.cer = []

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, labels, input_lengths, label_lengths = batch 
        output = self(inputs)          # (time, batch, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        metrics = {'train_loss': loss}
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, input_lengths, label_lengths = batch 
        output = self(inputs)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)
        y_hats = torch.argmax(output, dim=2)

        if batch_idx == 0:
            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
            print("\ntarget:", decoded_targets[0][:20])
            print("prediction:", decoded_preds[0][:20]) #logits.max(-1)[1])

    def test_step(self, batch, batch_idx):
        inputs, labels, input_lengths, label_lengths = batch 
        #output = self(inputs.unsqueeze(1).transpose(2, 3))          # (time, batch, n_class)
        output = self(inputs)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)
        y_hats = torch.argmax(output, dim=2)

        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
        cer = self.cer_metric(decoded_preds, decoded_targets)

        self.decoded_preds.append(decoded_preds)
        self.decoded_targets.append(decoded_targets)
        self.cer.append(cer)

        metrics = {'test_cer': cer}
        self.log_dict(metrics) 

        if batch_idx ==  0:
            print("\ntarget:", decoded_preds[0][:])
            print("prediction:", decoded_targets[0][:]) #logits.max(-1)[1])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=hparams['learning_rate'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=hparams['learning_rate'],
                                                        steps_per_epoch=(hparams['total_files'] // hparams['batch_size']), #int(len(train_loader)),
                                                        epochs=hparams['epochs'],
                                                        anneal_strategy='linear')
        sch_config = {
                'scheduler': scheduler,
                'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheulder':sch_config,}