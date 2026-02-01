import numpy as np
import pandas as pd
from pathlib import Path
def main():
    labels=Path("dataset/gt.csv")
    out_path=Path("dataset/splits")
    out_path.mkdir(parents=True, exist_ok=True)
    data_frame=pd.read_csv(labels)
    test_per=0.2
    train_per=0.6
    val_per=0.2
    ind=(np.random.default_rng(0)).permutation(len(data_frame))
    train_lim=int(train_per*(len(data_frame)))
    val_lim=train_lim+int(val_per*(len(data_frame)))
    train_idx=ind[:train_lim]
    val_idx=ind[train_lim:val_lim]
    test_idx=ind[val_lim:]
    paths=data_frame.iloc[train_idx]["name"].tolist()
    (out_path/"train.txt").write_text("\n".join(paths)+"\n")
    paths=data_frame.iloc[val_idx]["name"].tolist()
    (out_path/"val.txt").write_text("\n".join(paths)+"\n")
    paths=data_frame.iloc[test_idx]["name"].tolist()
    (out_path/"test.txt").write_text("\n".join(paths)+"\n")

if __name__=="__main__":
    main()