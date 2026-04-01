import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from model import ModularSNN, get_model
from logger import SNNLogger
from train import train
import argparse
from spikingjelly.datasets.n_mnist import NMNIST

import numpy as np
import random

def set_seed(seed=4):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(4)  # use any number you like


# Hyperparameters
batch_size = 64
timesteps = 30
epochs = 10
lr = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 1. Parse command line arguments ===
parser = argparse.ArgumentParser()
parser.add_argument('--t',     type=int, default=10)
parser.add_argument('--model', type=str, default='lif', help='Neuron model: lif, plif, log_plif, leak_gated , attention')
args = parser.parse_args()

dt = args.t

# Dataset
train_dataset = NMNIST(root='/home/ahmed/datasets/n_mnist',  train=True, data_type='frame', frames_number=dt, split_by='time',    duration=None )
test_dataset = NMNIST(root='/home/ahmed/datasets/n_mnist',  train=False, data_type='frame', frames_number=dt, split_by='time',    duration=None )

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize
model = get_model(args.model).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.CrossEntropyLoss()
sv = 'output/' + str(dt) + '/'
logger = SNNLogger( model_name= args.model, save_dir=sv)

# Train
train(model, train_loader, test_loader, optimizer, scheduler, criterion, logger,
      timesteps=timesteps,  device=device, epochs=epochs)

# Export
logger.export_csv()
logger.save_model(model)
