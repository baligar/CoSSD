import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchsummary import summary
import random

from asteroid_filterbanks import STFTFB, Encoder, Decoder, make_enc_dec
from asteroid import ConvTasNet
from asteroid.data import LibriMix
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.masknn import TDConvNet, TDConvNetpp
from asteroid.engine.system import System

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

print(device)

'''
The given code snippet defines a function named my_returnNormedVector_oneSam_efficient that takes a 2-dimensional tensor A as input (usually considered as a batch of samples where the first dimension represents the batch size, and the second dimension represents the feature values or channels).

Here's the step-by-step explanation of what the code does:

Subtract the Minimum: The line A -= A.min(dim=1, keepdim=True)[0] subtracts the minimum value of A along the first dimension (i.e., for each row or sample) from all elements in the corresponding row. The dim=1 specifies that the operation is to be done along the second dimension (indexing starts at 0), and the keepdim=True ensures that the result retains the original number of dimensions, allowing for proper broadcasting. This effectively translates the values in each row so that the minimum value in that row becomes zero.

Divide by the Maximum: The line A /= A.max(dim=1, keepdim=True)[0] divides A by its maximum value along the first dimension (i.e., for each row or sample). Similar to the subtraction step, this operation is performed for each row individually, scaling the values in that row so that the maximum value becomes one.

The result of these two operations is a normalized version of the input tensor A, where, for each row or sample, the values are rescaled to fall within the range [0, 1]. This kind of normalization is often used in machine learning and data preprocessing to standardize the scale of features, making them comparable across different dimensions or channels.
'''

def my_returnNormedVector_oneSam_efficient(A):
    # Subtract the minimum value of A along the first dimension (across channels)
    A -= A.min(dim=1, keepdim=True)[0]
    
    # Divide A by its maximum value along the first dimension (across channels)
    A /= A.max(dim=1, keepdim=True)[0]
    
    return A


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Define filterbanks and encoders
        fb = STFTFB(n_filters=256, kernel_size=128, stride=64)
        self.enc = Encoder(fb)
        self.cond_enc = Encoder(fb)
        decoder_fb = STFTFB(n_filters=514, kernel_size=128, stride=64)
        self.dec = Decoder(decoder_fb)

        # Define masker and linear layers
        self.masker = TDConvNet(in_chan=516, n_src=1)
        self.m = nn.Linear(14, 64)
        self.m2 = nn.Linear(64, 499)
        self.dropout = nn.Dropout(0.25)

        # Define convolution layers
        self.c1 = nn.Conv1d(in_channels=1, out_channels=258, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.bn1 = nn.BatchNorm1d(258)
        self.c1d_1 = nn.Conv1d(in_channels=516, out_channels=516, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.co_fc_1 = nn.Conv1d(in_channels=516, out_channels=1, kernel_size=3, stride=1, padding=1, padding_mode='zeros')

        # Define fully connected layers
        self.fc1 = nn.Linear(499, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout_2 = nn.Dropout(p=0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, wav, cond_wav):
        tf_rep = self.enc(wav)
        m_out = self.dropout(F.relu(self.m(cond_wav)))
        m2_out = self.dropout(F.relu(self.m2(m_out)))
        c1_out = F.relu(self.bn1(self.c1(m2_out)))
        concat_in = torch.cat((tf_rep, c1_out), dim=1)
        c1d_1_out = F.relu(self.c1d_1(concat_in))
        masks = self.masker(c1d_1_out)
        con_usq = c1d_1_out.unsqueeze(1) * masks
        co_fc_1_out = self.co_fc_1(con_usq.squeeze(1))
        x = self.sigmoid(self.fc2(self.dropout_2(F.relu(self.fc1(co_fc_1_out.view(-1, 499))))))  # Pres/abs
        wavs_out = self.dec(con_usq)  # Output sep waveform

        return wavs_out, x


'''
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        fb = STFTFB(n_filters=256, kernel_size=128, stride=64)
        self.enc = Encoder(fb)
        decoder_fb = STFTFB(n_filters=514, kernel_size=128, stride=64)
        self.dec = Decoder(decoder_fb)
        self.masker = TDConvNet(in_chan=516, n_src=1)

        self.m = nn.Linear(14, 64)
        self.m2 = nn.Linear(64, 499)
        self.dropout = nn.Dropout(0.25)
        
        self.c1 = nn.Conv1d(1, 258, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(258)
        
        self.c1d_1 = nn.Conv1d(516, 516, kernel_size=3, stride=1, padding=1)
        self.co_fc_1 = nn.Conv1d(516, 1, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(499, 32)
        self.fc2 = nn.Linear(32, 1)

        self.dropout_2 = nn.Dropout(p=0.1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, wav, cond_wav):
        tf_rep = self.enc(wav)

        m_out = self.dropout(F.relu(self.m(cond_wav)))
        m2_out = self.dropout(F.relu(self.m2(m_out)))
        
        c1_out = F.relu(self.bn1(self.c1(m2_out)))
                
        concat_in = torch.cat((tf_rep, c1_out), dim=1)
        
        c1d_1_out = F.relu(self.c1d_1(concat_in))
        masks = self.masker(c1d_1_out)
        
        con_usq = c1d_1_out.unsqueeze(1) * masks
        
        co_fc_1_out = self.co_fc_1(con_usq.squeeze(1))
        
        x = co_fc_1_out.view(-1, 499)
        x = self.dropout_2(F.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x)) # Pres/abs
        
        wavs_out = self.dec(con_usq) # output sep waveform
        
        return wavs_out, x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

cond_conv_tasnet = Model()
cond_conv_tasnet.to(device)

wav_out, pres_abs = cond_conv_tasnet(
    wav=torch.randn(10, 1, 32000).to(device), 
    cond_wav=torch.randn(10, 1, 14).to(device)
)

print(wav_out.shape, pres_abs.shape)


'''


print(device)

# Define and forward
cond_conv_tasnet = Model()
cond_conv_tasnet.to(device)
wav_out, pres_abs = cond_conv_tasnet(wav=torch.randn(10, 1, 32000).to(device), cond_wav=torch.randn(10, 1, 14).to(device))
print(wav_out.shape, pres_abs.shape)



def count_parameters(cond_conv_tasnet):
    return sum(p.numel() for p in cond_conv_tasnet.parameters() if p.requires_grad)

n = count_parameters(cond_conv_tasnet)
print("Number of parameters: %s" % n)


