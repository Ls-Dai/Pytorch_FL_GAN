import torch
import torch.nn as nn

from torch.autograd import Variable

from init_cycle_gan import init_federated
from models.fed_merge import fedavg
import clear
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import time 

from utils import DatasetLoader 

criterion_identity = torch.nn.L1Loss()
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if torch.random.uniform(0, 1) > 0.5:
                    i = torch.random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class TrainerFL():
    def __init__(self, config, clients, server):
        if config.no_cuda:
            from torch import FloatTensor, LongTensor
        else:
            from torch.cuda import FloatTensor, LongTensor

        self.count = 0
        self.config = config
        self.clients = clients
        self.server = server 

        self.input_A = FloatTensor(config.batch_size, config.input_nc, config.img_size, config.img_size)
        self.input_B = FloatTensor(config.batch_size, config.input_nc, config.img_size, config.img_size)
        self.target_real = Variable(FloatTensor(config.batch_size).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(FloatTensor(config.batch_size).fill_(0.0), requires_grad=False)

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            # A parameter collector
            para_collector_G_A2B = []
            para_collector_G_B2A = []
            para_collector_D_A = []
            para_collector_D_B = []

            # All clients update their local models
            for client in self.clients:

                # This func would return the parameters of the model trained in
                # this turn
                # timer
                time_start = time.time()
                
                loss_dict = self.train_epoch(epoch=epoch, client=client, config=config, time_start=time_start, input_A=input_A, input_B=input_B, target_real=target_real, target_fake=target_fake)
                
                time_end = time.time()
                self.config.epoch_eta = time_end - time_start

                # echo
                print(
                    'Client {}\tTrain Epoch: {}\tLoss: G (totally):{:6f}, D (totally):{:6f}, G-GAN:{:6f}, G-identity:{'
                    ':6f}, G-cycle:{:6f}, Timecost:{:2f}'.format(
                        client.id,
                        epoch,
                        loss_dict['loss_G'],
                        loss_dict['loss_D'],
                        loss_dict['loss_G_GAN'],
                        loss_dict['loss_G_identity'],
                        loss_dict['loss_G_cycle'],
                        time_end - time_start))

                # log write for this client
                # client.log_write(epoch=epoch, loss_g=train_loss_dict['g'], loss_d=train_loss_dict['d'])

                if epoch % config.com_epochs == 0:
                    para_collector_G_A2B.append(copy.deepcopy(client.generator_A2B.state_dict()))
                    para_collector_G_B2A.append(copy.deepcopy(client.generator_B2A.state_dict()))
                    para_collector_D_A.append(copy.deepcopy(client.discriminator_A.state_dict()))
                    para_collector_D_B.append(copy.deepcopy(client.discriminator_B.state_dict()))

            # federated!
            if epoch % config.com_epochs == 0:
                # merge + update global
                para_global_G_A2B = fedavg(para_collector_G_A2B)
                para_global_G_B2A = fedavg(para_collector_G_B2A)
                para_global_D_A = fedavg(para_collector_D_A)
                para_global_D_B = fedavg(para_collector_D_B)

                self.server.generator_A2B.load_state_dict(para_global_G_A2B)
                self.server.generator_B2A.load_state_dict(para_global_G_B2A)
                self.server.discriminator_A.load_state_dict(para_global_D_A)
                self.server.discriminator_B.load_state_dict(para_global_D_B)

                # echo
                print("Server's model has been update, Fed No.: {}".format(count))
                count += 1

                # model download local
                for client in self.clients:
                    client.load_model(generator_A2B=copy.deepcopy(server.generator_A2B),
                                    generator_B2A=copy.deepcopy(server.generator_B2A),
                                    discriminator_A=copy.deepcopy(server.discriminator_A),
                                    discriminator_B=copy.deepcopy(server.discriminator_B))
                    print("Client {}'s model has been updated from the server, Fed No.{}".format(client.id,
                                                                                                count))
        # Save the server model
        server.save_model()
        print("Global model has been saved as file on the server!")

    def train_epoch(self, epoch, client=client, config=config, time_start=time_start, input_A=input_A, input_B=input_B, target_real=target_real, target_fake=target_fake):
        # client.generator_A2B.train()
        # client.generator_B2A.train()
        # client.discriminator_A.train()
        # client.discriminator_B.train()

        loss_dict = {}
        # train_loader = client.train_data_load()
        # This is done in initialization part 

        # fake_A_buffer = ReplayBuffer()
        # fake_B_buffer = ReplayBuffer()

        for batch_idx, batch in enumerate(client.dataset):

            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            ###### Generators A2B and B2A ######
            client.optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = client.generator_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = client.generator_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*5.0

            # GAN loss
            fake_B = client.generator_A2B(real_A)
            pred_fake = client.discriminator_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = client.generator_B2A(real_B)
            pred_fake = client.discriminator_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = client.generator_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B = client.generator_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            
            client.optimizer_G.step()
            ###################################

            ###### Discriminator A ######
            client.optimizer_D_A.zero_grad()

            # Real loss
            pred_real = client.discriminator_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            # fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = client.discriminator_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            client.optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            client.optimizer_D_B.zero_grad()

            # Real loss
            pred_real = client.discriminator_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            # Fake loss
            # fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = client.discriminator_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            client.optimizer_D_B.step()
            ###################################

            ## Progress report
            
            # Make a progress bar :) desperate 
            batch_idx_real = batch_idx + 1
            capacity = len(client.dataset)
            ratio = batch_idx_real / capacity
            ratio_percentage = round(ratio * 100, 2)    
            epoch_percetage = round(100 * (epoch) / config.epochs, 2)

            time_current = time.time()
            eta = round(config.epoch_eta - (time_current - time_start), 2)

            print("\rClient: {} | Epochs completed: {:.2f}% | This batch: {:.2f}% | ETA: {:2f}".format(client.id, epoch_percetage, ratio_percentage, eta), end='')
            # :)
            
            if batch_idx_real + 1 == len(client.dataset):
                
                loss_G = loss_G
                loss_D = loss_D_A + loss_D_B
                loss_G_GAN = loss_GAN_A2B + loss_GAN_B2A
                loss_G_identity = loss_identity_A + loss_identity_B
                loss_G_cycle = loss_cycle_ABA + loss_cycle_BAB
                
                time.sleep(1.0)
                print('')
                
                ## logs
                client.log_write(epoch, loss_D=loss_D, loss_G=loss_G, loss_G_GAN=loss_G_GAN,
                                loss_G_identity=loss_G_identity, loss_G_cycle=loss_G_cycle)
                client.store_train_samples(epoch=epoch, img_dict={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A,
                                                                'fake_B': fake_B})

                loss_dict['loss_D'] = loss_D
                loss_dict['loss_G'] = loss_G
                loss_dict['loss_G_GAN'] = loss_G_GAN
                loss_dict['loss_G_identity'] = loss_G_identity
                loss_dict['loss_G_cycle'] = loss_G_cycle

                # Update learning rates
        client.lr_update()

        return loss_dict


if __name__ == '__main__':
    
    clear.clear_records(if_clients=True, if_servers=True, if_logs=True)

    clients, server, config = init_federated()

    datasetLoader = DatasetLoader()
    datasetLoader.load_dataset_default()

    for client in clients:
        client.load_dataset_from_dir("clients/" + str(client.id) + "/dataset/")
    train_federated(config, clients, server)
