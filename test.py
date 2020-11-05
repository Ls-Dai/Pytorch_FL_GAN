# test
import torch
import torch.nn.functional as func
import torch
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import syft as sy

from models.cnn import Cnn as Model

import os

def load_model(config, device):
    # initialize the model
    if (config.load_model and os.listdir('savedmodels')):
        model = Model().to(device)
        model.load_state_dict(torch.load(config.load_model_path + config.load_model_name))
    else:
        model = Model().to(device)

    # optimize
    optimizer = optim.SGD(model.parameters(), lr=config.lr)

    return model, optimizer

def test(config, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += func.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            print(type(output))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    config = settings.TestConfig()

    # Cuda setup
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Worker setup
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('datasets/', train=False,
                       transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.1307,), (0.3081,))
                                                     ])),
        batch_size=config.test_batch_size, shuffle=True, **kwargs)

    model, optimizer = load_model(config, device)
    test(config, model, device, test_loader)