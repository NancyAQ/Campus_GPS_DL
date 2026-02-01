import os
import pandas as pd
import torch
from pathlib import Path 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
class Campus_GPS_Dataset(Dataset):
    def __init__(self,labels="dataset/gt.csv",splits=None,img_size=224,norms=None,transform=None,augment=False):
        self.data_frame=pd.read_csv(labels)
        self.norms=norms
        if transform is not None:
            self.transform=transform
        else :
            if augment:
                self.transform=transforms.Compose(
                    [transforms.Resize((img_size,img_size)),
                    transforms.ColorJitter(brightness=0.25,contrast=0.25,saturation=0.15,hue=0.02),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1,1.0))],p=0.15),
                    transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
                )
            
            else:
                self.transform=transforms.Compose(
                    [transforms.Resize((img_size,img_size)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
                )
        if splits is not None:
            paths=[line.strip() for line in Path(splits).read_text().splitlines() if line.strip()]
            split_group=set(paths)
            mask=self.data_frame["name"].isin(split_group)
            self.data_frame=self.data_frame[mask].reset_index(drop=True)
    def __len__(self):
        return len(self.data_frame)
    def __getitem__(self,idx):
        row=self.data_frame.iloc[idx]
        img_path=Path("dataset/images")/row["name"]
        img=Image.open(img_path).convert("RGB")
        lat,lon=float(row["Latitude"]),float(row["Longitude"])
        #here we normalize the lat and lon accotding to the training mean and dev
        if self.norms is not None: 
            mean_lat,dev_lat,mean_lon,dev_lon=self.norms
            lat=(lat-mean_lat)/dev_lat
            lon=(lon-mean_lon)/dev_lon
        if self.transform:
            image=self.transform(img)
        label=torch.tensor([lat,lon],dtype=torch.float32)
        return image,label
    def denorm(self,pred_norm):
        if self.norms is None:
            return pred_norm
        else:
            mean=pred_norm.new_tensor([self.norms[0],self.norms[2]])
            standard_dev=pred_norm.new_tensor([self.norms[1],self.norms[3]])
            return pred_norm*standard_dev+mean