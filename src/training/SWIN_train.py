import torch.nn as nn
import torchvision
import torch
import os
import torch.optim as optim
from torchvision import transforms
from torchvision.models import swin_t,Swin_T_Weights
from torch.utils.data import DataLoader
from src.data_procesing.dataset import Campus_GPS_Dataset
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from src.metrics.metric import haversine_distance

def calc_norms(labels, train_split):
    data_frame=pd.read_csv(labels)
    with open(train_split,"r") as f:
        train_samps={line.strip() for line in f if line.strip()}
    train_data_frame=data_frame[data_frame["name"].isin(train_samps)]
    lat_mean=float(train_data_frame["Latitude"].mean())
    lat_dev=float(train_data_frame["Latitude"].std())
    lon_mean=float(train_data_frame["Longitude"].mean())
    lon_dev=float(train_data_frame["Longitude"].std())
    return lat_mean,lat_dev,lon_mean,lon_dev
    
def main():
    labels_csv="dataset/gt.csv"
    training_split="dataset/splits/train.txt"
    val_split="dataset/splits/val.txt"
    norms=calc_norms(labels_csv,training_split)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size=16
    num_workers=1
    img_size=224
    num_epochs=20##low since we dont have a lot of data yet
    swin_tf = Swin_T_Weights.DEFAULT.transforms()
    training_data=Campus_GPS_Dataset(labels="dataset/gt.csv",
                                     splits="dataset/splits/train.txt",
                                     img_size=img_size,
                                     norms=norms,
                                     transform=swin_tf,
                                     augment=True
                                     )
    validation_data=Campus_GPS_Dataset(labels="dataset/gt.csv",
                                     splits="dataset/splits/val.txt",
                                     img_size=img_size,
                                     norms=norms,
                                     transform=swin_tf,
                                     augment=False
                                     )
    """We firstly experiment with a pretrained resnet model(finetune it)"""
    model=swin_t(weights=Swin_T_Weights.DEFAULT)
    num_features=model.head.in_features
    # model.fc=nn.Linear(num_features,2) ##we want the model to output lat and lon
    #trying another regression heaf
    model.head=nn.Sequential(
        nn.Linear(num_features, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(512, 2),
    )
    model=model.to(device)
    # criterion=nn.MSELoss()
    #trying another loss fun
    criterion=nn.SmoothL1Loss(beta=0.1)
    optimizer=optim.Adam(model.parameters(),lr=0.0001)
    #we want to save the best checkpoint and save its weight
    min_median=float("inf")
    best_chk_path="CheckPoints/swin_t_CampGPS_best.pt"
    ##defining the data loaders
    training_loader=DataLoader(training_data,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    validation_loader=DataLoader(validation_data,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    for epoch in tqdm(range(num_epochs),desc="epochs progress:"):
        model.train()
        training_loss=0.0
        for imgs,labels in tqdm(training_loader,desc=f"epoch {epoch+1})",leave=False):
            imgs=imgs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(imgs)
            loss=criterion(logits,labels)
            loss.backward()
            optimizer.step()
            training_loss=training_loss+loss.item()*imgs.size(0)
        training_loss=training_loss/len(training_data)
        model.eval()
        validation_loss=0.0
        val_dist=[]
        with torch.no_grad():
            for imgs,labels in tqdm(validation_loader,desc=f"val {epoch+1}",leave=False):
                imgs=imgs.to(device)
                labels=labels.to(device)
                logits=model(imgs)
                loss=criterion(logits,labels)
                validation_loss=validation_loss+loss.item()*imgs.size(0)
                logits_unnorms=validation_data.denorm(logits)
                labels_unnorms=validation_data.denorm(labels)
                dists=haversine_distance(logits_unnorms,labels_unnorms)
                val_dist.append(dists.to(device))
                print(
                f"[epoch {epoch+1}] "
                f"logits mean: {logits_unnorms.mean(dim=0).tolist()} | "
                f"gt mean: {labels_unnorms.mean(dim=0).tolist()}"
                    )

        validation_loss=validation_loss/len(validation_data)
        val_dist=torch.cat(val_dist)
        mean_err=val_dist.mean().item()
        median_err=val_dist.median().item()
        if median_err<min_median:
             min_median= median_err
             torch.save({
                 "epoch":epoch+1,
                 "model_state":model.state_dict(),
                 "min_median":min_median,
                 "norms":norms,
                 "img_size":img_size,
             },best_chk_path)
             tqdm.write(f"Best checkoint saved in epoch {epoch+1}")
        tqdm.write(f"epoch{epoch+1}|"f"Training loss:{training_loss:.6f}|"f"Validation loss:{validation_loss:.6f}|"f"mean error:{mean_err:.1f}m|"f"median error:{median_err:.1f}m")
    # torch.save(model.state_dict(),"rsnet50_gps.pt")
if __name__=="__main__":
    main()
