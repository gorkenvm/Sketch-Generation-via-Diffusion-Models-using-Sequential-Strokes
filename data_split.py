import os
import ndjson
import json
import random

def create_split(ndjson_file, split_json_file, train_ratio=0.8, seed=42):
    with open(ndjson_file, "r") as f:
        data = ndjson.load(f)
    N = len(data)
    indices = list(range(N))
    random.seed(seed)
    random.shuffle(indices)
    N_train = int(N * train_ratio)
    split = {
        "train": indices[:N_train],
        "test": indices[N_train:]
    }
    with open(split_json_file, "w") as f:
        json.dump(split, f)
    print(f"Split saved to {split_json_file}")
    print(f"{N_train} train, {N-N_train} test samples.")


categories = ["bus", "cat", "rabbit"]
for cat in categories:
    ndjson_fp = f"subset/{cat}/{cat}.ndjson"
    split_fp = f"subset/{cat}/indices.json"
    create_split(ndjson_fp, split_fp)