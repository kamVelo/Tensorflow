#this is a linear regression tutorial using tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

dfTrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv") #trianing data
dfEval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv") #evalutation data
yTrain = dfTrain.pop("survived")
yTest = dfEval.pop("survived")
