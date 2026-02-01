from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models import resnet18,ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from src.metrics.metric import haversine_distance
from src.data_procesing.dataset import Campus_GPS_Dataset
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
from torch.utils.data import DataLoader
best_check_point="CheckPoints/swin_t_CampGPS_best.pt"
labels="dataset/gt.csv"
split="dataset/splits/test.txt"
output="outputs/Swin_T/test_res.csv"
batch=16
workers=1

def load_model():   # for swin
    #we build a model identical to the one we trained(we just copiid it from train.py)
        model=swin_t(weights=Swin_T_Weights.DEFAULT)
        num_features=model.head.in_features
        model.head=nn.Sequential(
                nn.Linear(num_features,512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512,2)
            )
        return model
def load_checkpoint(best_check_point,device):
    best_ckpt=torch.load(best_check_point,map_location=device)
    norms=best_ckpt["norms"]
    img_size=best_ckpt.get("img_size",224)
    return best_ckpt,norms, img_size

def ds_loader(gt_csv,split,norms,batch_size,img_size, workers,transform=None):
    dataset=Campus_GPS_Dataset(gt_csv,split,img_size,norms,transform=transform,)
    loader=DataLoader(dataset,batch_size,shuffle=False,num_workers=workers,)
    return dataset,loader

def evaluate(model, dataset,loader,device): 
    model.eval()
    errors,rows=[],[]
    with torch.no_grad():
        for imgs,labels in loader:
            imgs,labels=imgs.to(device),labels.to(device)
            preds,gts=dataset.denorm(model(imgs)),dataset.denorm(labels)
            curr_errors=haversine_distance(preds,gts).cpu()
            errors.append(curr_errors)
            np_preds,np_gts=preds.cpu().numpy(),gts.cpu().numpy()
            for idx in range(len(curr_errors)):
                rows.append({"lat":float(np_preds[idx,0]),
                             "lon":float(np_preds[idx,1]),
                             "True_Lat":float(np_gts[idx,0]),
                             "True_Lon":float(np_gts[idx,1]),
                             "Error":float(curr_errors[idx].item())})
    return torch.cat(errors).numpy(), rows

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chk,norms,img_size=load_checkpoint(best_check_point,device)
    model=load_model()
    model=model.to(device)
    model.load_state_dict(chk["model_state"])
    transform=Swin_T_Weights.DEFAULT.transforms()
    dataset,loader=ds_loader(labels,split,norms,batch,img_size,workers,transform=transform)
    # dataset,loader=ds_loader(labels,split,norms,batch,img_size,workers)
    errs,rows=evaluate(model,dataset,loader,device)
    mean_err,median_err=float(errs.mean()),float(np.median(errs))
    print(f"mean error is{mean_err:.2f}, and median is {median_err:.2f}")
    #saving all results
    pd.DataFrame(rows).to_csv(output,index=False)
if __name__=="__main__":
    main()
