import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


#

class Cifar10Trainer:

    def __init__(self, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        print('what is the device? cpu or gpu ? ', self.device)
        #
        self.shuffle_and_load_data_again()

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

    def train(self, net, num_of_steps=400, saving_weights=True):
        net.to(self.device)
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(net.parameters(), lr=3e-4)
        optimizer = optim.SGD(net.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-4, nesterov=True)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2,
                                                                                eta_min=1e-3)

        ######################## train for (num_of_steps) steps #########################
        iters = len(self.trainloader)
        no_epoch = int(num_of_steps / iters) + 1
        step_no = 0
        start = time.time()
        for epoch in range(no_epoch):
            net.to(self.device)
            net.train()
            for i, data in enumerate(self.trainloader, 0):
                step_no += 1
                if (step_no > num_of_steps):
                    continue
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                cosine_scheduler.step()
                # print(cosine_scheduler.get_lr())
            if (step_no > num_of_steps):
                continue
        end = time.time()
        train_time = end - start
        print('The training time for {} steps is: {:.2f} seconds'.format(num_of_steps, train_time))
        #################################################################################
        ########## validation accuracy on selected architecture after training ##########
        start = time.time()
        val_accuracy = self.validation(net)
        end = time.time()
        validation_time = end - start
        print('The validation time on whole validation set is {:.2f} seconds'.format(validation_time))
        print('The validation set accuracy of the sampled network is: {:.2f} %'.format(val_accuracy))
        #################################################################################
        if saving_weights:
            print('Saving child network\'s weights...')
            torch.save(net.state_dict(), 'weights.pth')
        #
        return val_accuracy, train_time

    def validation(self, net, single_step=False):
        """
        :param net: the network which we want to validate
        :param single_step: only enumerate over one batch not all of them. (use for updating controller)
        :return: validation accurary
        """
        correct, total = 0, 0
        predictions = []
        net.eval()
        with torch.no_grad():
            for i, data in enumerate(self.valloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predictions.append(outputs)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if single_step:
                    continue
            val_accuracy = 100 * correct / total
        return val_accuracy
