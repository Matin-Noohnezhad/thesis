import torch
import torch.nn as nn
import torch.nn.functional as F

class ChildNetwork(nn.Module):
    def __init__(self, layers_types, skip_connections, load_previous_network_weights=True, num_hidden_layers=7,
                 num_filters=32, num_classes=10):
        super(ChildNetwork, self).__init__()
        self.layers_types = layers_types
        self.skip_connections = skip_connections
        self.num_hidden_layers = num_hidden_layers
        ##### creating first possible layers
        conv3_num = 'conv3_1'
        conv5_num = 'conv5_1'
        bn_num = 'bn_1'
        self.layers = nn.ModuleDict({
            conv3_num: nn.Conv2d(3, num_filters, kernel_size=(3, 3), padding=(1, 1)),
            conv5_num: nn.Conv2d(3, num_filters, kernel_size=(5, 5), padding=(2, 2)),
            bn_num: nn.BatchNorm2d(num_filters, eps=1e-05, momentum=0.9)
        })
        ##### creating second to (n-1)th possible layers
        for i in range(2, num_hidden_layers + 1):
            #
            conv3_num = 'conv3_' + str(i)
            conv5_num = 'conv5_' + str(i)
            bn_num = 'bn_' + str(i)
            #
            next_layers = {conv3_num: nn.Conv2d(num_filters, num_filters, kernel_size=(3, 3), padding=(1, 1)),
                           conv5_num: nn.Conv2d(num_filters, num_filters, kernel_size=(5, 5), padding=(2, 2)),
                           bn_num: nn.BatchNorm2d(num_filters, eps=1e-05, momentum=0.9)}
            #
            self.layers.update(next_layers)
            #
        ##### creating conv 1x1 layers for reducing the channel numbers after concatenation layers
        self.conv1x1 = nn.ModuleDict({})
        for i in range(2, num_hidden_layers + 2):  # for all hidden layers and one before global average pooling
            n_skip_conn = int(torch.sum(skip_connections[(i * 2 - 3), 0, :(i - 1)]))
            if (n_skip_conn > 1 or i == num_hidden_layers+1):
                #
                conv1_num = 'conv1_' + str(i)
                bn_c1_num = 'bn_c1_' + str(i)
                #
                if (i == num_hidden_layers + 1):
                    next_layers = {conv1_num: nn.Conv2d(num_filters * n_skip_conn, num_classes, kernel_size=(1, 1)),
                                   bn_c1_num: nn.BatchNorm2d(num_classes, eps=1e-05, momentum=0.9)}
                else:
                    next_layers = {conv1_num: nn.Conv2d(num_filters * n_skip_conn, num_filters, kernel_size=(1, 1)),
                                   bn_c1_num: nn.BatchNorm2d(num_filters, eps=1e-05, momentum=0.9)}
                #
                self.conv1x1.update(next_layers)
                #
        ##### creating pooling layers (they don't have any weights)
        self.pool_3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.avgpool_3 = nn.AvgPool2d(3, stride=1, padding=1)
        self.globalavgpool = nn.AvgPool2d(32)
        #####
        if load_previous_network_weights:
            # load weights
            # print('Load previous network\'s weights to this network...')
            state_dict = torch.load('weights.pth')
            for i in range(1, num_hidden_layers + 1):
                #
                conv3_num = 'conv3_' + str(i)
                conv5_num = 'conv5_' + str(i)
                bn_num = 'bn_' + str(i)
                # initialize weights to previous networks weights
                with torch.no_grad():
                    self.layers[conv3_num].weight.copy_(state_dict['layers.' + conv3_num + '.weight'])
                    self.layers[conv3_num].bias.copy_(state_dict['layers.' + conv3_num + '.bias'])
                    #
                    self.layers[conv5_num].weight.copy_(state_dict['layers.' + conv5_num + '.weight'])
                    self.layers[conv5_num].bias.copy_(state_dict['layers.' + conv5_num + '.bias'])
                    #
                    self.layers[bn_num].weight.copy_(state_dict['layers.' + bn_num + '.weight'])
                    self.layers[bn_num].bias.copy_(state_dict['layers.' + bn_num + '.bias'])
                #
        else:
            # He initialization
            for m in self.modules():
                if isinstance(m, (nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def forward(self, x):
        #
        maxpool3x3 = 0
        avgpool3x3 = 1
        conv3x3 = 2
        conv5x5 = 3
        depthsepconv3x3 = 4
        depthsepconv5x5 = 5
        ###### feature extractor ######
        layers_output = []
        # first layer
        if self.layers_types[0, 0, maxpool3x3] == 1:
            layers_output.append(self.pool_3(x))
        elif self.layers_types[0, 0, avgpool3x3] == 1:
            layers_output.append(self.avgpool_3(x))
        elif self.layers_types[0, 0, conv3x3] == 1:
            layers_output.append(self.layers['bn_1'](self.layers['conv3_1'](F.relu(x))))
        elif self.layers_types[0, 0, conv5x5] == 1:
            layers_output.append(self.layers['bn_1'](self.layers['conv5_1'](F.relu(x))))
        #         elif self.layers_types[0, 0, depthsepconv3x3] == 1:
        #             pass
        #         elif self.layers_types[0, 0, depthsepconv5x5] == 1:
        #             pass
        ##### second to (n-1)th hidden layer
        for i in range(2, self.num_hidden_layers + 1):
            ##### concating selected connections together
            conn_to_prev = self.skip_connections[(i * 2 - 3), 0, :(i - 1)]
            n = 0
            x = None
            for j in range(len(conn_to_prev)):
                sel = conn_to_prev[j]
                if (sel == 1):
                    n += 1
                    if (n == 1):
                        x = layers_output[-(j + 1)]
                    else:
                        x = torch.cat((x, layers_output[-(j + 1)]), dim=1)
            if (n > 1):
                x = self.conv1x1['conv1_' + str(i)](x)
                x = self.conv1x1['bn_c1_' + str(i)](x)
            #####
            conv3_num = 'conv3_' + str(i)
            conv5_num = 'conv5_' + str(i)
            bn_num = 'bn_' + str(i)
            #
            if self.layers_types[(i - 1) * 2, 0, maxpool3x3] == 1:
                layers_output.append(self.pool_3(x))
            elif self.layers_types[(i - 1) * 2, 0, avgpool3x3] == 1:
                layers_output.append(self.avgpool_3(x))
            elif self.layers_types[(i - 1) * 2, 0, conv3x3] == 1:
                layers_output.append(self.layers[bn_num](self.layers[conv3_num](F.relu(x))))
            elif self.layers_types[(i - 1) * 2, 0, conv5x5] == 1:
                layers_output.append(self.layers[bn_num](self.layers[conv5_num](F.relu(x))))
            #
        #####
        ##### skip connections for last layer (global average pooling)
        conn_to_prev = self.skip_connections[-1, 0, :]
        n = 0
        x = None
        for j in range(len(conn_to_prev)):
            sel = conn_to_prev[j]
            if (sel == 1):
                n += 1
                if (n == 1):
                    x = layers_output[-(j + 1)]
                else:
                    x = torch.cat((x, layers_output[-(j + 1)]), dim=1)
        # if (n > 1):
        x = self.conv1x1['conv1_' + str(self.num_hidden_layers + 1)](x)
        x = self.conv1x1['bn_c1_' + str(self.num_hidden_layers + 1)](x)
        #####
        x = self.globalavgpool(x)
        # reshape
        x = x.view(-1, 10)

        return x