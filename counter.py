import pandas as pd
import os
import numpy as np

df = pd.read_csv("results.csv")

grouped = df.groupby(['Model','Task']).size()

print(grouped)
