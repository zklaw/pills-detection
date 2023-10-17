import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset import PillsDataset
from functions import get_transform

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = PillsDataset('pillsdetectiondataset/pillsPicture', get_transform(False), device)
img, ann = dataset.__getitem__(0)
plt.imshow(img.cpu().detach().numpy().astype(np.int32))
plt.show()