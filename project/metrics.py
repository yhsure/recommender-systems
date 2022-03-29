import pandas as pd
import numpy as np

## Utility file for commonly used methods
def relevant_column(preds, df_test, k): 
    topKpreds = preds.groupby(['uid']).apply(lambda x: x.nlargest(k,['score'])).reset_index(drop=True)[["uid", "iid", "score"]]
    merged = topKpreds.merge(df_test[["uid", "iid", "overall"]], how="left", on=["uid", "iid"])
    merged["relevant"] = (merged["overall"] >= 4) * 1 
    return merged

def PatK(preds, df_test, k):
    merged = relevant_column(preds, df_test, k)
    score  = merged[["uid", "iid", "relevant"]].groupby(by="uid")["relevant"].mean().mean()
    return score

#helper function for MRRatK: inverse of rank position of first relevant item
def first(x):
    for i in range(len(x)):
        if x.iloc[i].relevant == 1:
            return 1/(i+1)
    return 0 

def MRRatK(preds, df_test, k):
    merged = relevant_column(preds, df_test, k)
    score  = merged[["uid", "iid", "relevant"]].groupby(by="uid").apply(first).mean()
    return score

def MAPatK(preds, df_test, k):
    merged = relevant_column(preds, df_test, k)
    score  = merged[["uid", "iid", "relevant"]].groupby(by="uid")["relevant"].apply(lambda x: 1./np.arange(1,k+1) @ x).mean()
    return score

def HRatK(preds, df_test, k):
    merged = relevant_column(preds, df_test, k)
    score = merged[["uid", "iid", "relevant"]].groupby(by="uid")["relevant"].apply(lambda x: x.any()*1).mean()
    return score