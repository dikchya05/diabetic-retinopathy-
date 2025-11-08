import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, image_column='id_code', label_column='label'):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms
        self.image_column = image_column
        self.label_column = label_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = os.path.join(self.img_dir, f"{row[self.image_column]}.png")  # add extension if needed
        img = np.array(Image.open(img_path).convert("RGB"))
        label = int(row[self.label_column])

        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented['image']

        return img, label

def get_transforms(image_size=224):
    train_transforms = A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
        A.OneOf([A.GaussNoise(), A.MotionBlur(), A.MedianBlur(blur_limit=3)], p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.CLAHE(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])
    valid_transforms = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(),
        ToTensorV2()
    ])
    return train_transforms, valid_transforms
