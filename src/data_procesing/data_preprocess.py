import json
import math
import pandas as pd
import pillow_heif
import subprocess
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Tuple, Any
from PIL import Image
pillow_heif.register_heif_opener()
extensions={".jpeg",".jpg",".png",".heic",".heif"}

"""we use thos fucntion to unionize the type of our ds imgs"""
def type2jpg(out_path,img,quality=95):
    out_path.parent.mkdir(parents=True,exist_ok=True)
    img.save(out_path,format="JPEG", quality=quality)
    
"""load image from path"""
def load_img(path):
    img=Image.open(path)
    img=img.convert("RGB")
    return img

"""extracting gps cords using exiftool(basically running the comand)"""
def extract_cords(path):
    command=[r"C:\Tools\exiftool.exe", "-n","-j", "-GPSLatitude", "-GPSLongitude",str(path)]
    out=subprocess.run(command,capture_output=True,text=True,check=False)
    data=json.loads(out.stdout)[0]
    lat=data.get("GPSLatitude")
    lon=data.get("GPSLongitude")
    return lat,lon

def main():
    raw_path=Path("data/raw")
    processed_dir=Path("dataset/images")
    csv_out=Path("dataset/gt.csv")
    paths=[p for p in raw_path.rglob("*")if p.suffix.lower() in extensions]
    rows=[]
    for p in tqdm(sorted(paths),desc="Preprocessing imgs"):
        gps=extract_cords(p)
        lat,lon=gps
        img=load_img(p)
        out_name=f"{p.stem}.jpg"
        out_path=processed_dir/out_name
        type2jpg(out_path,img)
        rows.append({"name":out_name,
                      "Latitude":lat,
                      "Longitude":lon,})
    csv_out.parent.mkdir(parents=True,exist_ok=True)
    data_frame=pd.DataFrame(rows)
    data_frame.to_csv(csv_out,index=False)

if __name__=="__main__":
    main()
