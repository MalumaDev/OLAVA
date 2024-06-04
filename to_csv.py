import json
from pathlib import Path

import pandas as pd

with open("llava1.5_13b_test.json", "r") as f:
    data = json.load(f)

df = pd.read_csv("dummy.csv")

tmp = {}

for key in data:
    key2 = key.split("_")[:-1]
    if "myinfographic" in key or "mydoc" in key or "mychart" in key:
        key2 = key2[1:]
    key2 = "_".join(key2)
    tmp[key2] = data[key].split("\n")[0]


for row in df.iterrows():
    index, row = row
    if row["id"] in tmp:
        df.at[index, "pred"] = tmp[row["id"]]

df.to_csv("results.csv", index=False)
