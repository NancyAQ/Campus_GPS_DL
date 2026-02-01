import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
csv_path="outputs/Swin_T/test_res.csv"
out_path="outputs/Swin_T/error_cdf.png"
def main():
    plt.figure(figsize=(7,4))
    Max=60
    model="Swin_T"
    data_frame=pd.read_csv(csv_path)
    error=np.sort(pd.to_numeric(data_frame["Error"],errors="coerce").dropna().to_numpy())
    y_axis=np.arange(1, len(error)+1)/len(error)
    plt.plot(error,y_axis, label=model)
    plt.xlim(0,Max)
    plt.ylim(0,1)
    plt.xlabel("Error(m)")
    plt.ylabel("Part of test set")
    plt.title("CDF of errors")
    plt.grid(True,alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path,dpi=200)
    plt.close()
if __name__=="__main__":
    main()