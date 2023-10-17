import albumentations as A

def get_transform(train):
  if train:
    transform = A.Compose([A.Resize(224,224,3),
                           A.HorizontalFlip(p=0.5),
                           A.RandomBrightnessContrast(p=0.2),
                          #  A.Normalize(mean=(137, 143, 134), std=(50, 52, 48), max_pixel_value=255.0),
                          #  ToTensorV2(),
                           ],bbox_params=A.BboxParams(format='pascal_voc', label_fields = []))
  else:
    transform = A.Compose([A.Resize(224,224,3),
                          #  ToTensorV2(),
                          #  A.Normalize(mean=(137, 143, 134), std=(50, 52, 48), max_pixel_value=255.0),
                           ],
                           bbox_params=A.BboxParams(format='pascal_voc', label_fields = []))
  return transform