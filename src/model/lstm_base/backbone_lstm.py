from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from collections import namedtuple
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from easydict import EasyDict as edict

from utils import *
 
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class EncoderRNN(nn.Module):
    def __init__(self, args, input_size, hidden_size, device, n_layers=1):
        super(EncoderRNN, self).__init__()  
        self.device = device
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.state_rg = args.state_rg
        # self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True).to(self.device)

    def blank_state(self):
        hidden0 = torch.zeros(1, 1, self.hidden_size)
        # hidden0 = hidden0.to(self.device)
        return nn.Parameter(hidden0, requires_grad = self.state_rg).to(self.device)

    def forward(self, input_state, hidden):
        output, hidden = self.lstm(input_state, (hidden, hidden))
        return output, hidden

class EncoderRNN_batch(nn.Module):
    def __init__(self, args, input_size, hidden_size, device, n_layers=1):
        super(EncoderRNN_batch, self).__init__()  
        self.device = device
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.state_rg = args.state_rg
        # self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True).to(self.device)

    def blank_state(self, batch_size):
        hidden0 = torch.zeros(1, batch_size, self.hidden_size)
        # hidden0 = hidden0.to(self.device)
        return nn.Parameter(hidden0, requires_grad = self.state_rg).to(self.device)

    def forward(self, input_state, hm, cm):
        output, (hn, cn) = self.lstm(input_state, (hm, cm))
        return output, hn, cn

      
class KGState_LSTM(nn.Module):
    def __init__(self, args, history_len=1):
        super(KGState_LSTM, self).__init__()
        self.policy_lstm = EncoderRNN_batch(args, args.embed_size * 2, args.embed_size * 2, args.device)

    def set_up_hidden_state(self, batch_size):
        self.zero_hm = self.policy_lstm.blank_state(batch_size)
        self.zero_cm = self.policy_lstm.blank_state(batch_size)
        return self.zero_hm, self.zero_cm

    def __call__(self, history_seq, hm, cm):
        output, hn, cn = self.policy_lstm(history_seq, hm, cm)
        return output, hn, cn


class KGState_LSTM_ERU(nn.Module):
    def __init__(self, args, history_len=1):
        super(KGState_LSTM_ERU, self).__init__()
        self.policy_lstm = EncoderRNN_batch(args, args.embed_size * 3, args.embed_size * 2, args.device)

    def set_up_hidden_state(self, batch_size):
        self.zero_hm = self.policy_lstm.blank_state(batch_size)
        self.zero_cm = self.policy_lstm.blank_state(batch_size)
        return self.zero_hm, self.zero_cm

    def __call__(self, history_seq, hm, cm):

        output, hn, cn = self.policy_lstm(history_seq, hm, cm)
        return output, hn, cn


      
class KGState_LSTM_no_rela(nn.Module):
    def __init__(self, args, history_len=1):
        super(KGState_LSTM_no_rela, self).__init__()
        self.policy_lstm = EncoderRNN_batch(args, args.embed_size, args.embed_size, args.device)

    def set_up_hidden_state(self, batch_size):
        self.zero_hm = self.policy_lstm.blank_state(batch_size)
        self.zero_cm = self.policy_lstm.blank_state(batch_size)
        return self.zero_hm, self.zero_cm

    def __call__(self, history_seq, hm, cm):
        # print('hm = ', hm.shape)
        # print('cm = ', cm.shape)
        # print('history_seq = ', history_seq.shape)
        # input()
        output, hn, cn = self.policy_lstm(history_seq, hm, cm)
        return output, hn, cn