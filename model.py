import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch

def load_fasterRCNN(max_size, min_size, num_classes, device):
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT", max_size=max_size, min_size=min_size)
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  model = model.to(device)
  return model

def get_model(max_size, min_size, num_classes, device, path_to_model=""):
  
  model = load_fasterRCNN(max_size, min_size, num_classes, device)

  if len(path_to_model) != 0:
    try:    
      model.load_state_dict(torch.load(path_to_model)['model_state_dict'])
      print('Loaded saved model')
    except FileNotFoundError:      
      print('File not found, loaded default model')

  else:
    print('Loaded default model')

  return model
