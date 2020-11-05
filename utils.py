import os
import argparse
import imageio
import torch 
import shutil
import math 
import sys

from configs import TrainConfig


class DatasetLoader():
    def __init__(self):
        self.root = "datasets/"
        self.clients_dir = "clients/"
        self.dataset_name = "vangogh2photo"
    
    @property
    def num_of_clients(self):
        num_of_clients = 0
        for _ in os.listdir(self.clients_dir):
            num_of_clients += 1 
        return num_of_clients
    
    @property
    def testA_dir(self):
        return self.root + self.dataset_name + "/testA/"

    @property
    def testB_dir(self):
        return self.root + self.dataset_name + "/testB/"

    @property
    def trainA_dir(self):
        return self.root + self.dataset_name + "/trainA/"

    @property
    def trainB_dir(self):
        return self.root + self.dataset_name + "/trainB/"

    @property
    def dir_lst(self):
        return [self.testA_dir, self.testB_dir, self.trainA_dir, self.trainB_dir]


    def load_dataset_default(self):
        for dir in self.dir_lst:
            dataset_lst = os.listdir(dir)
            dataset_len = len(dataset_lst)
            for i in range(self.num_of_clients):
                num_of_data_owned = math.floor(dataset_len / self.num_of_clients)
                for j in range(num_of_data_owned):
                    source = dir + dataset_lst[i * num_of_data_owned + j]
                    if dir == self.testA_dir:
                        target = "clients/" + str(i) + "/dataset/" + "testA/"
                    elif dir == self.testB_dir:
                        target = "clients/" + str(i) + "/dataset/" + "testB/"
                    elif dir == self.trainA_dir:
                        target = "clients/" + str(i) + "/dataset/" + "trainA/"
                    elif dir == self.trainB_dir:
                        target = "clients/" + str(i) + "/dataset/" + "trainB/"
                    try:
                        shutil.copy(source, target)
                    except IOError as e:
                        print("Uable to copy file. %s" % e)
                    except:
                        print("Unexcepted error:", sys.exc_info())
            print(dir + " has been copied to all clients!")


def gif_maker(clients, config):
    for client in clients:
        img_root = client.local_dir + client.store_generated_root
        gif_images = []
        for name_list in os.listdir(img_root):
            if name_list == os.listdir(img_root)[0]:
                for _ in range(config.num_of_clients - 1):
                    gif_images.append(imageio.imread(img_root + name_list))
            elif name_list == os.listdir(img_root).pop():
                for _ in range(config.num_of_clients - 1):
                    gif_images.append(imageio.imread(img_root + name_list))
            else:
                gif_images.append(imageio.imread(img_root + name_list))
        imageio.mimsave(client.local_dir + "Client_" + str(client.id) + ".gif", gif_images, fps=3)


def dir_setup(path):
    if not os.path.exists(path):
        os.makedirs(path)
    

class Parser:
    def __init__(self):
        self.config = TrainConfig()
    
    def main_para_echo(self):
        print("-------------------------------")
        print("Number of clients: {}".format(self.config.num_of_clients))
        print("Train batch size: {}".format(self.config.batch_size))
        print("Train epochs: {}".format(self.config.epochs))
        print("If shuffle: {}".format(self.config.shuffle))
        print("One communication round contain epochs: {}".format(self.config.com_epochs))
        print("Using dataset: {}".format(self.config.dataset))
        print("Generating output images in epochs: {}".format(self.config.sample_rate))
        print("Using device: {}".format(self.config.device))
        print("The learning rate: {}".format(self.config.lr))
        print("-------------------------------")

    def parse(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--epochs", type=int, default=self.config.epochs, help='Epochs for the training')
        parser.add_argument("--clients", type=int, default=self.config.num_of_clients, help='Number of clients')
        parser.add_argument("--shuffle", type=int, default=int(self.config.shuffle), help='If Shuffle (IID)')
        parser.add_argument("--fed_epochs", type=int, default=self.config.com_epochs,
                            help='How many epochs for communication round')
        parser.add_argument("--dataset", type=str, default=self.config.dataset, help='Dataset used for the training')
        parser.add_argument("--check_epochs", type=int, default=self.config.sample_rate,
                            help='Train process visualization sample rate')
        parser.add_argument("--lr", type=float, default=self.config.lr, help="Adam: learning rate")

        args = parser.parse_args()

        self.config.epochs = args.epochs
        self.config.num_of_clients = args.clients
        self.config.shuffle = bool(args.shuffle)

        if args.fed_epochs == 0:
            self.config.com_epochs = args.fed_epochs + 1
        else:
            self.config.com_epochs = args.fed_epochs

        self.config.dataset = args.dataset
        self.config.sample_rate = args.check_epochs
        self.config.lr = args.lr

        # echo, for linux > command to write to the logs, record the command
        print("python train.py --epochs {} --clients {} --shuffle {} --fed_epochs {} --dataset {} --check_epochs {} --lr {"
            "} > logs.txt".format(args.epochs,
                                    args.clients,
                                    args.shuffle,
                                    args.fed_epochs,
                                    args.dataset,
                                    args.check_epochs,
                                    args.lr))

        # echo
        self.main_para_echo()

if __name__ == "__main__":
    pass
