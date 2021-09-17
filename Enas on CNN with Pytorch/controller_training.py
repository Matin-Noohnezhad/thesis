import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from childnet import ChildNetwork
from utils import exponential_moving_average

class ControllerTrainer:

    def __init__(self, controller_net, cifar10_trainer, num_hidden_layers, num_filters, num_classes, batch_size):
        #
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #
        self.controller = controller_net
        self.cifar10_trainer = cifar10_trainer
        self.num_layers = num_hidden_layers
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.val_accuracy_list = []
        ###
        self.shuffle_and_load_data_again()
        ###

    def shuffle_and_load_data_again(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.48216, 0.44653),
                                  (0.24703, 0.24349, 0.26159))])
        #
        indices = np.arange(50000)
        np.random.shuffle(indices)
        #
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                       shuffle=False, num_workers=2,
                                                       sampler=torch.utils.data.SubsetRandomSampler(indices[:45000]))
        self.valloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                     shuffle=False, num_workers=2,
                                                     sampler=torch.utils.data.SubsetRandomSampler(indices[45000:50000]))
        # self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        #                                             download=True, transform=transform)
        # self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
        #                                               shuffle=False, num_workers=2)

    def train(self, num_of_steps=2000, grad_clip_value=None):
        optimizer = optim.Adam(self.controller.parameters(), lr=35e-5, eps=1e-3)

        start = time.time()
        iters = len(self.valloader)
        no_epoch = int(num_of_steps/iters) + 1
        step_no = 0
        for epoch in range(no_epoch):
            for i, data in enumerate(self.valloader, 0):
                step_no += 1
                if(step_no > num_of_steps):
                    continue
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                soft_1, soft_2, layers_types, skip_connections = self.controller()
                ############### create new child network and validate it
                child_net = ChildNetwork(layers_types, skip_connections, load_previous_network_weights=True,
                                         num_hidden_layers=self.num_layers,
                                         num_filters=self.num_filters, num_classes=self.num_classes)
                child_net.to(self.device)
                child_net.eval()
                with torch.no_grad():
                    outputs = child_net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    # predictions.append(outputs)
                    total = labels.size(0)
                    correct = (predicted == labels).sum().item()
                    val_accuracy = 100 * correct / total
                self.val_accuracy_list.append(val_accuracy)
                baseline = exponential_moving_average(self.val_accuracy_list)
                policy_gradient_multiplier = val_accuracy - baseline
                self.controller.train()
                optimizer.zero_grad()
                reward1 = soft_1 * layers_types * policy_gradient_multiplier
                reward2 = soft_2 * skip_connections * policy_gradient_multiplier
                total_loss = - (reward1.mean() + reward2.mean())
                total_loss.backward()
                if not (grad_clip_value == None):
                    nn.utils.clip_grad_norm_(self.controller.parameters(), max_norm=grad_clip_value)
                optimizer.step()
            if (step_no > num_of_steps):
                continue
        end = time.time()
        total_time = end - start
        print('The time for training the controller for {} steps is: {:.2f} seconds'.format(num_of_steps, total_time))
        #######################

