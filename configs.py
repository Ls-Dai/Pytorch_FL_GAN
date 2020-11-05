import torch
import numpy as np
import torch.nn as nn


class TrainConfig:
    def __init__(self):

        self.num_of_clients = 1

        # train dataset setup
        self.batch_size = 5
        self.shuffle = True
        self.collate_fn = None
        self.batch_sampler = None
        self.sampler = None
        self.num_workers = 0
        self.pin_memory = False
        self.drop_last = True
        self.timeout = 0
        self.worker_init_fn = None

        # train network setup
        self.epochs = 100

        # fed setup
        self.com_epochs = 10

        # train dataset setup
        # self.dataset = "MNIST"
        # self.dataset = "CIFAR10"
        self.dataset = "vangogh2photo"

        # generated samples store
        self.sample_rate = 1

        # Model setup
        self.lr = 2 * 1e-4
        self.momentum_d1 = 0.5
        self.momentum_d2 = 0.999

        # others
        self.valid_loss_min = np.Inf

        # for cycle gan 
        self.input_nc = 3 # number of channels of input data 
        self.output_nc = 3 # number of channels of output data 
        self.n_residual_blocks = 9 

        self.seed = 1

        # self.itisjusttomakepatienceinatraingepoch
        self.epoch_eta = 500

    # @property

    @property
    def order(self):
        return not self.shuffle

    @property
    def num_data_owned_setup(self):
        return int(50000 / self.num_of_clients)

    # model setup
    @property
    def latent_dim(self):
        if self.dataset == "MNIST":
            return 100
        elif self.dataset == "CIFAR10":
            return 100

    @property
    def n_classes(self):
        if self.dataset == "MNIST":
            return 10
        elif self.dataset == "CIFAR10":
            return 10

    @property
    def img_size(self):
        if self.dataset == "MNIST":
            return 28
        elif self.dataset == "CIFAR10":
            return 32
        elif self.dataset == "vangogh2photo":
            return 256

    @property
    def channels(self):
        if self.dataset == "MNIST":
            return 1
        elif self.dataset == "CIFAR10":
            return 3
        elif self.dataset == "vangogh2photo":
            return 3 

    @property
    def img_shape(self):
        if self.dataset == "MNIST":
            return (self.channels, self.img_size, self.img_size)
        elif self.dataset == "CIFAR10":
            return (self.channels, self.img_size, self.img_size)
        elif self.dataset == "vangogh2photo":
            return (self.channels, self.img_size, self.img_size)

    # CUDA setup

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def no_cuda(self):
        return not torch.cuda.is_available()

    # generated samples store
    @property
    def if_img(self):
        if self.sample_rate != 0:
            return True
        else:
            return False

    # decay of learning rate, where to start at which epoch 
    @property
    def decay_epoch(self):
        return self.epochs / 2

class TestConfig:
    def __init__(self):
        pass
