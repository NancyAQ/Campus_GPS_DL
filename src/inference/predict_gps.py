from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch
import sys
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision.models import swin_t,Swin_T_Weights #for our best performing model
from src.metrics.metric import haversine_distance
from torchvision import transforms
from src.data_procesing.dataset import Campus_GPS_Dataset
from torch.utils.data import DataLoader
#global vars needed
def load_model():   
    #we build a model identical to the one we trained(we just copiid it from SWIN_train.py)
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
##we mostly used code from our dataset code and evaluation
def denorm(pred, norms):
    if norms is None:
            return pred
    else:
            mean=pred.new_tensor([norms[0],norms[2]])
            standard_dev=pred.new_tensor([norms[1],norms[3]])
            return pred*standard_dev+mean
    
def load_checkpoint(best_check_point,device):
    best_ckpt=torch.load(best_check_point,map_location=device)
    norms=best_ckpt["norms"]
    img_size=best_ckpt.get("img_size",224)
    return best_ckpt,norms, img_size
##All the global vars we need
img_size=224
transform=Swin_T_Weights.DEFAULT.transforms()
best_check_point="CheckPoints/swin_t_CampGPS_best.pt"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_ckpt=torch.load(best_check_point,map_location=device)
norms=best_ckpt["norms"]
model=load_model().to(device)
model.load_state_dict(best_ckpt["model_state"])
model.eval() #since we want it in inference mode

def predict_gps(image: np.ndarray) -> np.ndarray:
     img=Image.fromarray(image.astype(np.uint8))
     #aplying same transform to image as in ds and addong 4th dim
     img=transform(img).unsqueeze(0).to(device)
     #actual inference
     with torch.no_grad():
          pred=denorm((model(img).squeeze(0)),norms)
          return pred.cpu().numpy()
def main():
    if len(sys.argv)!=2:
        print('Please pride a path to the input rgb img')
        sys.exit(1)
    img_path=Path(sys.argv[1])
    if not img_path.exists():
        raise FileNotFoundError("img not found")
    img=np.array(Image.open(img_path).convert("RGB"))
    print("The GPS coordinates are: ",predict_gps(img))
    
    
if __name__=="__main__":
        main()