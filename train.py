import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from dataset import PillsDataset
from functions import get_transform, collate_fn, split_data, plot_batch
from model import get_model

batch_size = 10
idx = 30
a= 0.6
img_size = (224,224,3)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

trainset, valset, testset = split_data('pillsdetectiondataset\pillsPicture', device,
                                       train_perc=0.8, val_perc=0.15, test_perc=0.05,
                                       img_size=img_size)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# plot_batch(trainloader)

model = get_model(max_size=img_size[0], min_size=img_size[0], num_classes=2, device=device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.00009, momentum = 0.0009, weight_decay=0.0006)
# optimizer = torch.optim.Adam(params, lr=0.0005)



