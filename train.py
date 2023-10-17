import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from dataset import PillsDataset
from functions import get_transform, collate_fn, split_data, plot_batch

batch_size = 10
idx = 30
a= 0.6

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

trainset, valset, testset = split_data('pillsdetectiondataset\pillsPicture', device, train_perc=0.8, val_perc=0.15, test_perc=0.05)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

plot_batch(trainloader)