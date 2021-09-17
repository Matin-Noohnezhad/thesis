import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch

from childnet import ChildNetwork
from cifar10_training import Cifar10Trainer
from controller import ControllerNetwork
from controller_training import ControllerTrainer


##

def run(outer_iterations=310, controller_iterations=2000, child_iterations=400, hidden_size=100, num_layer_types=4,
        num_layers=5, num_filters=32, num_classes=10, batch_size=128, max_num_of_best_samples=3):
    val_accuracy_list = []
    train_time_list = []

    best_samples = OrderedDict({})
    # initialize our dict and force it to have size max_num_of_best_samples, which we keep it that size till the end.
    for i in range(-max_num_of_best_samples + 1, 1):
        best_samples.update({i: 0})

    ##

    cifar10_trainer = Cifar10Trainer(batch_size=batch_size)
    controller = ControllerNetwork(hidden_size, num_layers, num_layer_types)
    controller_trainer = ControllerTrainer(controller, cifar10_trainer, num_hidden_layers=num_layers,
                                           num_filters=num_filters,
                                           num_classes=num_classes, batch_size=batch_size)
    load_previous_network_weights = None
    ##
    for i in range(outer_iterations):
        start = time.time()
        print("ITERATION NUMBER %d" % (i + 1))
        ############## sample model ##############
        print('Controller is sampling architecture...')
        soft_1, soft_2, layers_types, skip_connections = controller()
        #
        print(torch.exp(soft_1))
        print(layers_types)
        # print(torch.exp(soft_2))
        # print(skip_connections)
        ############## initialize weights ##############
        print('Creating new child network...')
        if i == 0:
            load_previous_network_weights = False
        child_net = ChildNetwork(layers_types, skip_connections, load_previous_network_weights,
                                 num_hidden_layers=num_layers,
                                 num_filters=num_filters, num_classes=num_classes)
        load_previous_network_weights = True

        ############## train model ##############
        print('Training child network...')
        val_accuracy, train_time = cifar10_trainer.train(child_net, num_of_steps=child_iterations)
        val_accuracy_list.append(val_accuracy)
        train_time_list.append(train_time)
        ## adding to best samples
        worst_one_in_the_list = list(best_samples.keys())[0]
        if val_accuracy > worst_one_in_the_list:
            best_samples.popitem(0)
            best_samples.update({val_accuracy: (layers_types, skip_connections)})
            best_samples = OrderedDict(sorted(best_samples.items()))
        ##
        ############## train controller ##############
        print('Training controller network')
        controller_trainer.train(num_of_steps=controller_iterations)
        ##############################################
        end = time.time()
        total_time = end - start
        print('total time of 1 outer iteration is {:.2f}'.format(total_time))
        print('##############################################')
        print('##############################################')
    ##
    # plt.plot(val_accuracy_list)
    # plt.figure()
    # plt.plot(train_time_list)

    return best_samples
