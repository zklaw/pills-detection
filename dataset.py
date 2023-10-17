import torch.nn as nn
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch


class PillsDataset(nn.Module):
    def __init__(self, root, transforms, device):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.ann = list(sorted(os.listdir(os.path.join(root, "annotations"))))
        self.device = device

    def __get_bboxes__(self, ann_path):

      tree = ET.parse(ann_path)
      root = tree.getroot()
      annotations = []
      for bndbox in root.iter('bndbox'):
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        if xmin>=xmax:
          xmin=xmax-1
        if ymin>=ymax:
          ymin=ymax-1

        annotations.append([xmin, ymin, xmax, ymax])
      return annotations

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        ann_path = os.path.join(self.root, "annotations", self.ann[idx])

        # img = cv2.imread(img_path)
        # print(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


        img = np.array(Image.open(img_path).convert("RGB"))


        boxes = self.__get_bboxes__(ann_path)
        # # convert everything into a torch.Tensor
        boxes = torch.tensor(boxes)
        # # there is only one class
        num_objs = len(boxes)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # labels_name = ['pill'*len(boxes)]


        image_id = torch.tensor([idx])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # if self.transforms is not None:

        transformed = self.transforms(image=img, bboxes = boxes)




        img = torch.tensor(transformed['image'], dtype=torch.float32)
        boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)

        img = img.to(self.device)
        target = {}
        target["boxes"] = boxes.to(self.device)
        target["labels"] = labels.to(self.device)
        target["image_id"] = image_id.to(self.device)
  
        return img, target


    def calc_mean(self):
      m_red = []
      m_green = []
      m_blue = []

      std_red = []
      std_green = []
      std_blue = []

      for idx in range(len(self.imgs)):
        img, _ = self.__getitem__(idx)
        img = img.cpu().detach().numpy()
        img = np.reshape(img, (3,np.shape(img)[0],np.shape(img)[1]))

        red = img[0]
        green = img[1]
        blue = img[2]

        sum_red = np.sum(red)
        sum_green = np.sum(green)
        sum_blue = np.sum(blue)

        m_red.append(sum_red/(np.shape(red)[0]*np.shape(red)[1]))
        m_green.append(sum_green/(np.shape(green)[0]*np.shape(green)[1]))
        m_blue.append(sum_blue/(np.shape(blue)[0]*np.shape(blue)[1]))

        std_red.append(np.std(red))
        std_green.append(np.std(green))
        std_blue.append(np.std(blue))


      m = [np.sum(m_red)/len(m_red), np.sum(m_green)/len(m_green), np.sum(m_blue)/len(m_blue)]
      std = [np.sum(std_red)/len(std_red), np.sum(std_green)/len(std_green), np.sum(std_blue)/len(std_blue)]
      print(m, std)


    def __len__(self):
        return len(self.imgs)

