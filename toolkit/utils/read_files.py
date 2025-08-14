import os
import json
import math
import random
import numpy as np
import pandas as pd


# 功能3：从csv中读取特定的key对应的值
def func_read_key_from_csv(csv_path, key):
    values = []
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        if key not in row:
            values.append("")
        else:
            value = row[key]
            if pd.isna(value):
                value = ""
            values.append(value)
    return values
