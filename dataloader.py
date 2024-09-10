import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torch.utils.data import Dataset, Subset
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
import torch


from PIL import Image
import os
import pandas as pd
import numpy as np

class CustomCrop(object):
    def __init__(self, size=(30,768)):
        self.size = size
    def __call__(self, img):
        # Get the dimensions of the image
        #w, h = img.size
        # Specify the starting indices for cropping
        left = 0
        top = 0
        # Perform the crop
        img = transforms.functional.crop(img, top, left, self.size[0], self.size[1])
        return img
    
custom_starting_crop = CustomCrop()

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, selected_labels=['A', 'NA'], merge= True):
        self.data_dir = data_dir
        self.basic_transform =transforms.Compose([transforms.ToTensor(), 
                                                  #custom_starting_crop,
                                                  CustomCrop(),
                                            #transforms.CenterCrop(size=(30,512)),
                                            #transforms.Resize(size=(30,512),antialias=True),
                                              ])
        self.transform = transform
        self.merge = merge
        self.images = os.listdir(data_dir)
        self.labels_csv = pd.read_csv(csv_file)
        self.selected_labels = selected_labels 
        self.image_label_dict = self.create_image_label_dict()
        #print(self.image_label_dict)
        self.label_to_index = {label: index for index, label in enumerate(selected_labels)}
        # {'N': 0, '~': 1, 'A': 2, 'O': 3}

    
    def create_image_label_dict(self):
        image_label_dict = {}
        for _, row in self.labels_csv.iterrows():
            image_name = row[0]
            label = row[1]
            if self.merge:
                if label in ['N', '~', 'O']:
                    new_label = 'NA'
                elif label in ['A']:
                    new_label = 'A'
                if new_label in self.selected_labels: # [A,NA]
                    image_path = os.path.join(self.data_dir, image_name+'.csv')
                    if os.path.exists(image_path):
                        image_label_dict[image_name] = new_label
            else: 
                if label in self.selected_labels: # [A,N]
                    image_path = os.path.join(self.data_dir, image_name+'.csv')
                    if os.path.exists(image_path):
                        image_label_dict[image_name] = label
        return image_label_dict

    def __len__(self):
        return len(self.image_label_dict)

    def __getitem__(self, idx):
        image_name = list(self.image_label_dict.keys())[idx]
        image_path = os.path.join(self.data_dir, image_name+'.csv')
        #print(image_path)
        image = np.loadtxt(image_path)# pd.read_csv(image_path,header=None) #.to_numpy()
        #print(type(image))
        #print(f"Image shape: {image.ndim}") 
        if image.ndim ==1:
            image = np.expand_dims(image, axis=-1) 
        image = np.array(image,dtype=np.float32)
        image = self.basic_transform(image)
        #print(f" t Image shape: {image.shape}") 
        if self.transform:
            image = self.transform(image)
        #print(f"tt Image shape: {image.shape}") 
        label = self.label_to_index[self.image_label_dict[image_name]]
        #print(image_name,label)
        return (image, label), image_name


class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)
    


from sklearn.model_selection import train_test_split

class ImageDataModule2(pl.LightningDataModule):
    def __init__(self,  img_dir, labels_csv,
                        batch_size=32, 
                        num_workers=8, 
                        validation_split=0.15,
                        test_split=0.1,
                        merge = True,
                        ):  # Add test_split argument
        super().__init__()
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split
        self.test_split = test_split  # Store test_split
        self.csv_file_path = labels_csv
        # Define the transforms for data augmentation (you can customize these)
        self.augment = transforms.Compose([
            
                v2.RandomAdjustSharpness(sharpness_factor=4,p=0.25),
                v2.RandomPerspective(distortion_scale=0.2, p=0.25),
                v2.RandomAutocontrast(p=0.25),
                #v2.RandomErasing(p=0.5,scale=(0.1,0.1),ratio=(0.5, 2.5)),
                v2.RandomApply( 
                   torch.nn.ModuleList([
                       v2.GaussianBlur(kernel_size=(3,3),sigma=(0.2,1.5)),
                       ]),p=0.25 )
        ])  
        self.merge = merge

    def prepare_data(self):
        self.dataset = CustomImageDataset(data_dir=self.img_dir,csv_file=self.csv_file_path,
                                          selected_labels=['A', 'NA'] if self.merge else ['A','N'])
        # Split data for each class
        train_idx, test_val_idx = train_test_split(list(range(len(self.dataset))), test_size=self.test_split+self.validation_split, random_state=42)
        val_test_ratio = self.validation_split / (self.test_split + self.validation_split)
        val_idx, test_idx = train_test_split(test_val_idx, test_size=val_test_ratio, random_state=42)
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        print(self.dataset.label_to_index)

    def setup(self, stage=None):      
        self.train_dataset = MyDataset(subset=Subset(self.dataset,self.train_idx),transform=self.augment)
        self.val_dataset = MyDataset(Subset(self.dataset,self.val_idx),transform=None)
        self.test_dataset = MyDataset(Subset(self.dataset,self.test_idx),transform=None)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          pin_memory=True
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          pin_memory=True
                          )


if __name__ == "__main__":

    data_module = ImageDataModule2(img_dir = '/home/saptarshi/Research/encoder_decoder/physionet2017challenge/training2017',
                                   #'Beat_level_ECG2017_images_jpg_format' ,
                                   labels_csv= '/home/saptarshi/Research/encoder_decoder/physionet2017challenge/training2017/REFERENCE.csv',  #'REFERENCE-v3.csv',
                                   batch_size=32)
    data_module.prepare_data()
    data_module.setup()
    # Create a LightningModule and a Lightning Trainer
    # Then, train your model using trainer.fit(your_model)