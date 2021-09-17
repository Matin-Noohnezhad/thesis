import torch
import torch.nn.functional as F
from torch import nn

from utils import layer_sampler, connection_sampler


class ControllerNetwork(nn.Module):
    def __init__(self, hidden_size, num_layers, num_layer_types):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_layer_types = num_layer_types
        #####
        self.fc1 = nn.Linear(self.hidden_size, self.num_layer_types)
        self.fc2 = nn.Linear(self.hidden_size, self.num_layers)
        self.dropout = nn.Dropout(0.2)
        #
        self.lstm = nn.LSTM(input_size=self.num_layers + self.num_layer_types, hidden_size=self.hidden_size,
                            num_layers=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        ####
        # initialized uniformly between [-0.1, 0.1]
        torch.nn.init.uniform_(self.fc1.weight, a=-0.1, b=0.1)
        torch.nn.init.uniform_(self.fc2.weight, a=-0.1, b=0.1)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, a=-0.1, b=0.1)
        ####

    def forward(self, input_for_training=None, tanh_const=2.5, temperature=5):
        """
        :param input_for_training: concated tensor of target1 and target2 from previous trainings
        :return: return soft
        """
        # torch.autograd.set_detect_anomaly(True)
        # initialize zero inputs
        previous_h = torch.zeros(1, 1, self.hidden_size)
        previous_c = torch.zeros(1, 1, self.hidden_size)
        previous_target_1 = torch.zeros(1, 1, self.num_layer_types)
        previous_target_2 = torch.zeros(1, 1, self.num_layers)
        #
        soft_1_list = []
        soft_2_list = []
        target_1_list = []
        target_2_list = []
        #
        n = (
                    self.num_layers * 2)  # first layer connected to input (no other layer connected to input) & the last layer is (global average pooling + softmax layer)
        for i in range(n):
            if input_for_training == None:
                input = torch.cat((previous_target_1, previous_target_2), 2)
            else:
                input = input_for_training[i:i + 1]
            out, (previous_h, previous_c) = self.lstm(input, (previous_h, previous_c))
            out = out.view(-1, self.hidden_size)
            out = self.relu(out)
            out = self.dropout(out)
            ##### block one --> choosing layer type #####
            previous_soft_1 = F.softmax(tanh_const * self.tanh(self.fc1(out) / temperature), dim=1).reshape(1, 1, -1)
            previous_target_1 = layer_sampler(previous_soft_1, i)
            target_1_list.append(previous_target_1)
            previous_soft_1 = torch.log(previous_soft_1)
            soft_1_list.append(previous_soft_1)
            ##### block two --> choosing connections to previous layers #####
            sigmoid_const = int(i / 2)
            previous_soft_2 = torch.log(1 / (1 + sigmoid_const * torch.exp(-self.fc2(out)))).reshape(1, 1, -1)
            # previous_soft_2 = F.logsigmoid(self.fc2(out)).reshape(1, 1, -1)
            previous_target_2 = connection_sampler(previous_soft_2, i)
            target_2_list.append(previous_target_2)
            soft_2_list.append(previous_soft_2)
        soft_1 = torch.cat(soft_1_list, dim=0)
        soft_2 = torch.cat(soft_2_list, dim=0)
        target_1 = torch.cat(target_1_list, dim=0)
        target_2 = torch.cat(target_2_list, dim=0)
        return soft_1, soft_2, target_1, target_2
