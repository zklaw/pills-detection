import albumentations as A
from dataset import PillsDataset
import torch
import math
import numpy as np
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def get_transform(train: bool, img_size: tuple):
  img_height, img_width, channels = img_size
  if train:
    transform = A.Compose([A.Resize(img_height, img_width, channels),
                           A.HorizontalFlip(p=0.5),
                           A.RandomBrightnessContrast(p=0.2),
                          #  A.Normalize(mean=(137, 143, 134), std=(50, 52, 48), max_pixel_value=255.0),
                          #  ToTensorV2(),
                           ],bbox_params=A.BboxParams(format='pascal_voc', label_fields = []))
  else:
    transform = A.Compose([A.Resize(img_height, img_width, channels),
                          #  ToTensorV2(),
                          #  A.Normalize(mean=(137, 143, 134), std=(50, 52, 48), max_pixel_value=255.0),
                           ],
                           bbox_params=A.BboxParams(format='pascal_voc', label_fields = []))
  return transform

def collate_fn(batch):
    return tuple(zip(*batch))

def split_data(path,device, train_perc, val_perc, test_perc, img_size):

    trainset = PillsDataset(path, get_transform(True, img_size), device)
    valset = PillsDataset(path, get_transform(False, img_size), device)

    indices = torch.randperm(len(trainset)).tolist()

    a = len(trainset)-math.floor(len(trainset)*train_perc)
    b = math.floor(len(trainset)*val_perc)

    trainset = Subset(trainset, indices[:-a])
    valset = Subset(valset, indices[-a:])

    indices = torch.randperm(len(valset)).tolist()

    testset = Subset(valset, indices[:-b])
    valset = Subset(valset, indices[-b:])

    return trainset, valset, testset

def plot_bboxes(bboxes, ax, linewidth=1, edgecolor='r', facecolor='none'):
  for bbox in bboxes:
    a = bbox[0]
    b = bbox[1]
    width = bbox[2]-bbox[0]
    height = bbox[3]-bbox[1]
    rect = Rectangle((a, b),width,height,linewidth=linewidth,edgecolor=edgecolor,facecolor=facecolor)
    ax.add_patch(rect)

def plot_batch(dataloader):

    x, y = next(iter(dataloader))
    batch_size = len(x)

    for i in range(batch_size):
        img = x[i].cpu().detach().numpy().astype(np.int32)
        ann = y[i]
        ann = ann['boxes'].cpu().detach().numpy()

        plt.imshow(img)
        ax = plt.gca()
        plot_bboxes(ann,ax)
        plt.show()


def plot_results(images, outputs, targets, threshold):
   
  for img, output, ground_truth in zip(images, outputs, targets):

    # output = outputs[i]
    # ground_truth = targets[i]
    ground_truth = ground_truth['boxes'].cpu().detach().numpy()


    boxes = output['boxes'].cpu().detach().numpy()
    scores = output['scores'].cpu().detach().numpy()
    output = boxes[scores>threshold]

    plt.imshow(img.astype(np.int32))
    ax = plt.gca()
    plot_bboxes(output,ax)
    plot_bboxes(ground_truth,ax,edgecolor='g')

    plt.show()