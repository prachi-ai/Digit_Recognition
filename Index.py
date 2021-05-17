import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
# !unzip '/content/Train_UQcUa52 (1).zip'
df = pd.read_csv("/content/train.csv")
df.head()