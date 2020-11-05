import os
import shutil
from torch.utils import data
import torch
import numpy as np
from torchvision import datasets, transforms
import copy

from client import Client
from server import Server
from utils import Parser, gif_maker
import glob 
import PIL 


def load_datasets(clients, config):

    for client in clients:
        for i in range(config.num_data_owned_setup):
            j = i + client.id * config.num_data_owned_setup
            client.load_dataset_from_dir(dir = "clients/" + str(self.id) + '/dataset/')


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
        

def init_federated():

    # clients list
    clients = []

    # load configs
    parser = Parser()
    parser.parse()
    config = parser.config

    # generate clients
    for i in range(config.num_of_clients):
        clients.append(Client(id=i, config=config))

    # generate server
    server = Server(id=0, config=config)
    
    # foo 
    server.generator_A2B.apply(weights_init_normal)
    server.generator_B2A.apply(weights_init_normal)
    server.discriminator_A.apply(weights_init_normal)
    server.discriminator_B.apply(weights_init_normal)

    if (os.path.exists(server.model_dir + server.generator_name_A2B) and
             os.path.exists(server.model_dir + server.generator_name_B2A) and 
             os.path.exists(server.model_dir + server.discriminator_name_A) and
             os.path.exists(server.model_dir + server.discriminator_name_B)):
        server.load_model()
        print("Global model saved on the server has been restored!")

    elif not (os.path.exists(server.model_dir + server.generator_name_A2B) and
             os.path.exists(server.model_dir + server.generator_name_B2A) and 
             os.path.exists(server.model_dir + server.discriminator_name_A) and
             os.path.exists(server.model_dir + server.discriminator_name_B)):
        print("Global model has been created!")
    else:
        raise EOFError

    # load datasets
    # This method is detached from the init part 
    # load_datasets(clients=clients, config=config)
    
    # load models
    for client in clients:
        client.load_model(generator_A2B=server.generator_A2B,
                          generator_B2A=server.generator_B2A,
                          discriminator_A=server.discriminator_A,
                          discriminator_B=server.discriminator_B)
        print("Client {}'s model has been updated from the server".format(client.id))
    
    return clients, server, config


if __name__ == '__main__':

    clients, server, config = init_federated()
    #gif_maker(clients=clients, config=config)

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

