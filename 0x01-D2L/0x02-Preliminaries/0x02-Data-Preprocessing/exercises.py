""" Exercises from section 2.1 - Data preprocessing in d2l.ai """
import pandas as pd
import torch as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

## 1 - Try Try loading datasets, e.g., Abalone from the UCI Machine Learning Repository and inspect their properties. What fraction of them has missing values? What fraction of the variables is numerical, categorical, or text?

wine_data = pd.read_csv("https://archive.ics.uci.edu/static/public/109/data.csv")
X = wine_data.to_numpy()

## Try indexing and selecting data columns by name rather than by column number. The pandas documentation on indexing has further details on how to do this.

alcohol = tf.tensor(wine_data["Alcohol"].to_numpy(), dtype=tf.float64)

## How large a dataset do you think you could load this way? What might be the limitations? Hint: consider the time to read the data, representation, processing, and memory footprint. Try this out on your laptop. What happens if you try it out on a server?

# - The dataset can be as big as the RAM on the computer that is running the program.

## How would you deal with data that has a very large number of categories? What if the category labels are all unique? Should you include the latter?

# - Store them on the hard drive

## What alternatives to pandas can you think of? How about loading NumPy tensors from a file? Check out Pillow, the Python Imaging Library.

# - Sql Databases, Python dictionaries, Polars ... 
