# we chose to visualize with opencv sicne we have prir exp
import cv2 
import numpy as np
from src.metrics.metric import haversine_distance
import itertools
import pandas as pd
import csv
from pathlib import Path
import torch
#error function in meters

def visualize_results(img_path,preds,gt):
    img=cv2.imread(str(img_path))
    img=cv2.resize(img,(1024,1024))
    x,y=5,20
    lat,lon=gt.tolist()
    pred_lat,pred_lon=preds.tolist()
    error=haversine_distance(preds.unsqueeze(0),gt.unsqueeze(0)).item()
    text=f"Error: {error:.1f}m|True({lat:.4f},{lon:.4f})|Pred({pred_lat:.4f},{pred_lon:.4f})"
    cv2.putText(img,text,(x,y),color=(0,255,0),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.8,thickness=2) #green 
    # cv2.imshow("result image",img)
    return img


def main():
    out_dir=Path("outputs/Swin_T/visualized")
    test_set=Path("dataset/splits/test.txt")
    test_res_path="outputs/Swin_T/test_res.csv"
    img_dir=Path("dataset/images")
    imgs_paths=[img_dir/l.strip() for l in test_set.read_text().splitlines() if l.strip()]
    data_frame=pd.read_csv(test_res_path)
    for idx in range(len(data_frame)):
        row=data_frame.iloc[idx]
        preds=torch.tensor([row["lat"],row["lon"]],dtype=torch.float32)
        gt=torch.tensor([row["True_Lat"],row["True_Lon"]],dtype=torch.float32)
        img_path=imgs_paths[idx]
        img=visualize_results(img_path,preds,gt)
        cv2.imwrite(str(out_dir/f"{idx:04d}_{img_path.stem}_vis.jpg"),img)

if __name__=="__main__":
    main()