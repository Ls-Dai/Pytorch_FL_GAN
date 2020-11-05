import torch
import numpy as np
from torch.autograd import Variable
from torch.utils import data
from torchvision.utils import save_image
import itertools
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

import copy
from utils import *
import pandas as pd

from models.cycle_GAN import Generator, Discriminator
import glob
import random
import os
import PIL


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class Client:
    def __init__(self, id, config):
        self.id = id

        self.local_dir = 'clients/' + str(id) + '/'
        dir_setup(self.local_dir)

        # Now dataset is a united OBJECT in this version
        self.dataset = None
        dir_setup(self.local_dir + 'dataset/')
        # add something new

        # Only used in former vision
        # self.label = []
        # dir_setup(self.local_dir + 'label/')

        # New models
        self.generator_A2B = Generator(input_nc=config.input_nc, output_nc=config.output_nc).to(config.device)
        self.generator_B2A = Generator(input_nc=config.output_nc, output_nc=config.input_nc).to(config.device)
        dir_setup(self.local_dir + 'model/')
        self.generator_name_A2B = "generator_A2B.pkl"
        self.generator_name_B2A = "generator_B2A.pkl"

        self.discriminator_A = Discriminator(input_nc=config.input_nc).to(config.device)
        self.discriminator_B = Discriminator(input_nc=config.output_nc).to(config.device)
        dir_setup(self.local_dir + 'model/')
        self.discriminator_name_A = "discriminator_A.pkl"
        self.discriminator_name_B = "discriminator_B.pkl"

        # The number of samples the client owns (before really load data)
        # self.num_data_owned_setup = 0
        # It won't be used in cycle gan no more
        dir_setup(self.local_dir + 'dataset/trainA/')
        dir_setup(self.local_dir + 'dataset/trainB/')
        dir_setup(self.local_dir + 'dataset/testA/')
        dir_setup(self.local_dir + 'dataset/testB/')

        # self config
        self.config = config

        # generated samples store
        self.store_generated_root = 'results/'
        dir_setup(self.local_dir + self.store_generated_root)

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.generator_A2B.parameters(), self.generator_B2A.parameters()),
                                            lr=self.config.lr, betas=(0.5, 0.999))
        
        self.optimizer_D_A = torch.optim.Adam(self.discriminator_A.parameters(), self.config.lr, betas=(0.5, 0.999))

        self.optimizer_D_B = torch.optim.Adam(self.discriminator_B.parameters(), self.config.lr, betas=(0.5, 0.999))

        # Learning scheduler setup
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(config.epochs, 0, config.decay_epoch).step)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=LambdaLR(config.epochs, 0, config.decay_epoch).step)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=LambdaLR(config.epochs, 0, config.decay_epoch).step)

    """# optimizer for training
    # @property
    @property
    def optimizer_G(self):
        return torch.optim.Adam(itertools.chain(self.generator_A2B.parameters(), self.generator_B2A.parameters()),
                                lr=self.config.lr, betas=(0.5, 0.999))

    @property
    def optimizer_D_A(self):
        return torch.optim.Adam(self.discriminator_A.parameters(), self.config.lr, betas=(0.5, 0.999))

    @property
    def optimizer_D_B(self):
        return torch.optim.Adam(self.discriminator_B.parameters(), self.config.lr, betas=(0.5, 0.999))
    """
    # Not gonna be used
    # def load_data(self, data_label_list):
    #    self.dataset.append(data_label_list[0])
    #    self.label.append(data_label_list[1])

    def load_dataset_from_dir(self, dir):

        transforms_ = [transforms.Resize(int(self.config.img_size * 1.12), PIL.Image.BICUBIC),
                       transforms.RandomCrop(self.config.img_size),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        train_dataset = ImageDataset(dir, transforms_=transforms_, unaligned=True)

        self.dataset = data.DataLoader(dataset=train_dataset,
                                       batch_size=self.config.batch_size,
                                       shuffle=self.config.shuffle,
                                       collate_fn=self.config.collate_fn,
                                       batch_sampler=self.config.batch_sampler,
                                       num_workers=self.config.num_workers,
                                       pin_memory=self.config.pin_memory,
                                       drop_last=self.config.drop_last,
                                       timeout=self.config.timeout,
                                       worker_init_fn=self.config.worker_init_fn)

    def load_model_from_path(self, model_path):
        self.generator_A2B = torch.load(model_path + self.generator_name_A2B)
        self.generator_B2A = torch.load(model_path + self.generator_name_B2A)
        self.discriminator_A = torch.load(model_path + self.discriminator_name_A)
        self.discriminator_B = torch.load(model_path + self.discriminator_name_B)

    def load_model(self, generator_A2B, generator_B2A, discriminator_A, discriminator_B):
        self.generator_A2B = copy.deepcopy(generator_A2B)
        self.generator_B2A = copy.deepcopy(generator_B2A)
        self.discriminator_A = copy.deepcopy(discriminator_A)
        self.discriminator_B = copy.deepcopy(discriminator_B)

    def num_data_owned(self):
        return len(self.dataset)

    # client writes logs
    def log_write(self, epoch, loss_D, loss_G, loss_G_GAN, loss_G_identity, loss_G_cycle):
        loss_data_frame = pd.DataFrame(columns=None, index=[epoch], data=[[loss_D, loss_G, loss_G_GAN, loss_G_identity, loss_G_cycle]])
        loss_data_frame.to_csv("clients/" + str(self.id) + "/" + "log.csv", mode='a', header=False)

    # store generated samples during train process by some rate

    def store_train_samples(self, epoch, img_dict):
        dir = os.path.abspath(os.getcwd()) + "/clients/" + str(self.id) + "/results/"
        for image_name, image_tensor in img_dict.items():
            path = dir + image_name + "_" + str(epoch) + ".png"
            save_image(image_tensor, path, normalize=True)
        print("Client {}'s epoch {} result has been saved as {}".format(self.id, epoch, dir))

    """def store_train_samples(self, sample, root, img_name):
        shape = self.config.img_shape
        sample_images = sample.reshape(sample.size(0), shape[0], shape[1], shape[2])
        save_image(sample_images.data,
                   os.path.abspath(os.getcwd()) + "/clients/" + str(self.id) + "/" + root + img_name, normalize=True)"""

    # update learning rates 
    def lr_update(self):
        self.lr_scheduler_G.step()
        self.lr_scheduler_D_A.step()
        self.lr_scheduler_D_B.step()