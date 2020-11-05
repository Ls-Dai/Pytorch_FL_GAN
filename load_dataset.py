import torch 
import os 
import shutil
import math 
import sys


def load_dataset_default():
    
    root = "datasets/"
    clients_dir = "clients/"

    num_of_clients = 0

    for _ in os.listdir(clients_dir):
        num_of_clients += 1 

    #print(num_of_clients)

    dataset_name = "vangogh2photo"

    testA_dir = root + dataset_name + "/testA/"
    testB_dir = root + dataset_name + "/testB/"
    trainA_dir = root + dataset_name + "/trainA/"
    trainB_dir = root + dataset_name + "/trainB/"
    
    dir_lst = [testA_dir, testB_dir, trainA_dir, trainB_dir]

    for dir in dir_lst:
        
        dataset_lst = os.listdir(dir)
        dataset_len = len(dataset_lst)
        for i in range(num_of_clients):
            num_of_data_owned = math.floor(dataset_len / num_of_clients)
            for j in range(num_of_data_owned):
                source = dir + dataset_lst[i * num_of_data_owned + j]
                if dir == testA_dir:
                    target = "clients/" + str(i) + "/dataset/" + "testA/"
                elif dir == testB_dir:
                    target = "clients/" + str(i) + "/dataset/" + "testB/"
                elif dir == trainA_dir:
                    target = "clients/" + str(i) + "/dataset/" + "trainA/"
                elif dir == trainB_dir:
                    target = "clients/" + str(i) + "/dataset/" + "trainB/"
                try:
                    shutil.copy(source, target)
                except IOError as e:
                    print("Uable to copy file. %s" % e)
                except:
                    print("Unexcepted error:", sys.exc_info())
        print(dir + " has been copied to all clients!")


if __name__ == "__main__":
    load_dataset_default()