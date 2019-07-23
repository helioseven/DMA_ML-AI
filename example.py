import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load a new data set
dataset = pd.read_csv("./happiness.csv")

# data pre-processing
for col in dataset.columns:
	dataset[col].replace('', np.nan, inplace=True)
dataset.dropna(inplace=True)

# save the processed data set
dataset.to_csv("./modified_happiness.csv")