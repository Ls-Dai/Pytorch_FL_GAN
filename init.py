import os
import shutil
from torch.utils import data
import cv2
import torch
import numpy as np
from torchvision import datasets, transforms
import copy

from client import Client
from server import Server
from utils import parse, gif_maker


def dataset_sel(dataset_name):

    root = 'datasets/'

    return {
        "MNIST": datasets.MNIST(root, download=True, train=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])),
        "CIFAR10": datasets.CIFAR10(root, train=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ), target_transform=None, download=True)  # transforms.Normalize((0.1307,), (0.3081,))
    }.get(dataset_name)


def load_datasets(clients, config):

    train_dataset = dataset_sel(config.dataset)

    num_data_total = int(config.num_data_owned_setup * len(clients))

    train_distributer = data.DataLoader(dataset=train_dataset,
                                        batch_size=num_data_total,
                                        shuffle=config.shuffle
                                        )

    images, labels = next(iter(train_distributer))

    # Heterogeneous
    if config.order:

        images_array = images.numpy()
        labels_array = labels.numpy()
        images_list = []
        labels_list = []
        for i in range(config.n_classes):
            for j in range(num_data_total):
                if labels_array[j] == i:
                    images_list.append(images_array[j])
                    labels_list.append(labels_array[j])
        images_array = np.array(images_list)
        labels_array = np.array(labels_list)
        images = torch.from_numpy(images_array)
        labels = torch.from_numpy(labels_array)

    else:
        images, labels = next(iter(train_distributer))

    for client in clients:
        for i in range(config.num_data_owned_setup):
            j = i + client.id * config.num_data_owned_setup
            client.load_data([images[j], labels[j]])


def init_federated():

    # clients list
    clients = []

    # load configs

    config = parse()

    # generate clients
    for i in range(config.num_of_clients):
        clients.append(Client(id=i, config=config))

    # generate server
    server = Server(id_num=0, config=config)

    if os.path.exists(server.model_dir + server.generator_name) and os.path.exists(
            server.model_dir + server.discriminator_name):
        server.load_model()
        print("Global model saved on the server has been restored!")
    elif not (os.path.exists(server.model_dir + server.generator_name) or os.path.exists(
            server.model_dir + server.discriminator_name)):
        print("Global model has been created!")
    else:
        raise EOFError
    # load datasets
    load_datasets(clients=clients, config=config)

    # load models
    for client in clients:
        client.load_model(generator=copy.deepcopy(server.generator),
                          discriminator=copy.deepcopy(server.discriminator))
        print("Client {}'s model has been updated from the server".format(client.id))

    return clients, server, config


if __name__ == '__main__':

    clients, server, config = init_federated()
    gif_maker(clients=clients, config=config)

    """pic = np.array([])
    
    for i in range(50):
        pic_h = np.array([])
        for j in range(50):
            if j == 0:
                pic_h = np.array(clients[9].dataset[i * 50 + j]).transpose(1, 2, 0)
            else:
                pic_h = np.hstack((pic_h, np.array(clients[9].dataset[i * 50 + j]).transpose(1, 2, 0)))
        if i == 0:
            pic = pic_h
        else:
            pic = np.vstack((pic, pic_h))

    cv2.imshow("pic", pic)
    cv2.waitKey(0)

    shutil.rmtree("clients")
    # shutil.rmtree("servers")"""

