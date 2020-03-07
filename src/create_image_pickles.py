import pandas as pd
import numpy as np
import joblib
import glob
from tqdm import tqdm

if __name__=="__main__":
    files=glob.glob("../input/train_*.parquet")
    for f in files:
        df=pd.read_parquet(f)
        # print(df.columns)
        imageids=df.image_id.values
        df=df.drop('image_id',axis=1)
        image_arrays=df.values

        for j, img_ids in tqdm(enumerate(imageids),total=len(imageids)):
            joblib.dump(image_arrays[j,:],f"../input/image_pickles/{img_ids}.pkl")