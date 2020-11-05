import torch

from utils import dir_setup
from models.cycle_GAN import Generator, Discriminator 


class Server:
    def __init__(self, id, config):
        self.model_name = None
        self.id = id

        self.local_dir = "servers/" + str(self.id) + "/"
        dir_setup(self.local_dir)

        self.model_dir = self.local_dir + "model/"
        dir_setup(self.model_dir)

        self.generator_A2B = Generator(input_nc=config.input_nc, output_nc=config.output_nc).to(config.device)
        self.generator_B2A = Generator(input_nc=config.output_nc, output_nc=config.input_nc).to(config.device)
        self.discriminator_A = Discriminator(input_nc=config.input_nc).to(config.device)
        self.discriminator_B = Discriminator(input_nc=config.output_nc).to(config.device)

        self.generator_name_A2B = "generator_A2B.pkl"
        self.generator_name_B2A = "generator_B2A.pkl"

        self.discriminator_name_A = "discriminator_A.pkl"
        self.discriminator_name_B = "discriminator_B.pkl"

        self.config = config


    def save_model(self):
        torch.save(self.model_dir + self.generator_name_A2B)
        torch.save(self.model_dir + self.generator_name_B2A)
        torch.save(self.model_dir + self.discriminator_name_A)
        torch.save(self.model_dir + self.discriminator_name_B)

    def load_model(self):
        self.generator_A2B = torch.load(self.model_dir + self.generator_name_A2B, map_location=torch.device(self.config.device))
        self.generator_B2A = torch.load(self.model_dir + self.generator_name_B2A, map_location=torch.device(self.config.device))
        self.discriminator_A = torch.load(self.model_dir + self.discriminator_name_A, map_location=torch.device(self.config.device))
        self.discriminator_B = torch.load(self.model_dir + self.discriminator_name_B, map_location=torch.device(self.config.device))