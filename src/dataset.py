#this code is an edited version of the code found at https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import random
import os

class CustomImageDataset(Dataset):
    def __init__(self, labels_df, img_dir, transform=None, transform_rate=0.1):
        self.img_labels = labels_df
        self.img_dir = img_dir
        self.transform = transform
        self.transform_rate = transform_rate

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_idx = self.img_labels.index[idx]
        img_path = os.path.join(self.img_dir, "im"+str(img_idx)+".jpg")
        image = read_image(img_path, ImageReadMode.RGB)
        labels = torch.from_numpy(self.img_labels.iloc[idx].values).float()
        if torch.cuda.is_available():
            image = image.to("cuda")
            labels = labels.to("cuda")
        if self.transform:
            image = self._random_transform(image)
        image = image.float()
        return image, labels
    
    def _random_transform(self, image):
        for transform in self.transform:
            if random.random() < self.transform_rate:
                image = transform(image)
        return image
    
    
### Info About Custom Dataset:

## Using Data and Indexing System:
#The label data is kept in one-hotted Pandas dataframe "labels_df". This does not contain the image data. Label data is fairly small so can be kept directly in memory.
#Dataframes have two indexing systems. Hidden internal index (iloc, or .index[]), which always goes from 0 to len-1.
#Second indexing system is the visible index, which might be different. For our dataframe, the visible index follows image indexing, which can be used to load in image data.
#Pytorch DataLoaders call dataset __getitem__ method with idx values from 0 to __len__()-1. This corresponds to our dataframe hidden indexing.
#For one item within __getitem__ method, we are dealing with a single in our dataframe
#To get image index from hidden index we set img_idx = self.img_labels.index[idx]. This is used to get image data for the item.
#Corresponding one-hotted label data is obtained with hidden index: self.img_labels.iloc[idx].values

## Reading Image
#We use torchvision read_image method
#We force every image to be read in as color images with ImageReadMode.RGB
#This way every image has 3 channels, otherwise gray images have 1 channels and Dataloader fails
#Other way would be to grayscale everything

## Other
#Additional image transformers in self.transform are only applied if they exists (not None)