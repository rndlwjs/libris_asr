import torch
import torchaudio
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import torch.nn.functional as F
from .model import DeepSpeech2

hparams = {
    "n_cnn_layers": 3,
    "n_rnn_layers": 5,
    "rnn_dim": 512,
    "n_class": 30,
    "n_feats": 128,
    "stride": 2,
    "dropout": 0.1,
    "learning_rate": 5e-4,
    "batch_size": 4,
    "epochs": 20
}

class DeepSpeech2ASR(pl.LightningModule):
    def __init__(self,):

        super(DeepSpeech2ASR, self).__init__()
        #self.hparams = hparams
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

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, labels, input_lengths, label_lengths = batch 
        output = self(inputs)          # (time, batch, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)
        #print(output.shape); exit()
        loss = self.criterion(output, labels, input_lengths, label_lengths)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, input_lengths, label_lengths = batch 
        logits = self(x)
        print(logits.shape)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=hparams['learning_rate'])


#optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
#                                            steps_per_epoch=int(len(train_loader)),
#                                            epochs=hparams['epochs'],
#                                            anneal_strategy='linear')