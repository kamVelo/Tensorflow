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

print(dfTrain.loc[0]) #this is row zero
print(dfTrain.head()) #this prints the first five rows
print(dfTrain.describe()) #this prints some generalised statistical observations
print(dfTrain.shape) # this prints out the shape of the dataframe.
dfTrain.age.hist(bins = 20)#this gives us a histogram of age onboard
dfTrain.sex. value_counts().plot(kind="barh") #this gives a hor. bar chart of sex
dfTrain['class'].value_counts().plot(kind="barh")
pd.concat([dfTrain,yTrain],axis=1).groupby("sex").survived.mean().plot(kind="barh").set_xlabel("% survived")

"""
    Observations:
    - most passengers are 20-30
    - most passengers are men
    - most passengers are 3rd class
    - females were much more likely to survive.
"""
""""""

CATEGORICAL_COLUMNS = ["sex", "n_siblings_spouses","parch","class","deck",
                       "embark_town", "alone"]
NUMERICAL_COLUMNS = ["age", "fare"]
featureColumns = []

for name in CATEGORICAL_COLUMNS:
    vocabulary = dfTrain[name].unique() # gets a list of the possible values
    featureColumns.append(tf.feature_column.categorical_column_with_vocabulary_list(name,vocabulary))
    
for name in NUMERICAL_COLUMNS:
    featureColumns.append(tf.feature_column.numeric_column(name, dtype=tf.float32))

""" 
the way a linear regression model is trained is by feeding the model data from 
the dataset. The way data is loaded is through batches. In this model data is fed
in batches of 32. The reason we don't load it one by one is because it would take
too long. The reason we don't load it all at once is that whilst with our dataset
it might be possible to load all 627 training records into RAM, with other models
where there could be terabytes of data it's simply not feasible to load all the 
data at once. Thus batches of 32 records provide a good middle ground between,
one by one and all at once.

we will feed the batches to our model multiple times according to the number of
EPOCHS specified. 

An epoch is simply one stream of our entire dataset. Meaning, if the model has
gone through our entire dataset (in prespecified batches of 32) it has completed
one epoch.  

Since we need to feed our data in batches, multiple times (according to the no. of 
epochs) we need to create something called the INPUT FUNCTION. this function 
decides how our dataset will be made into batches at each epoch.

if we don't do enough epochs what will happen is simply that our model will not
be accurate enough. 
however, if we do too many epochs we may overfit the model to our specific data.
  
in this tutorial we will make an input_function which will encode our pandas
dataframe object into a tf.data.Dataset object.
this function can be slightly complicated.
"""
def makeInputFN(dataDf, labelDf, numEpochs=10, shuffle=True, batchSize=32):
    def inputFunction(): # inner function which will be returned
        ds = tf.data.Dataset.from_tensor_slices((dict(dataDf),labelDf))
        if shuffle:
            ds = ds.shuffle(1000)#randomize order of data
        ds = ds.batch(batchSize).repeat(numEpochs)# split dataset into batches of 32
        # and repeat process for number of epochs
        return ds #return a batch of the dataset
    return inputFunction #returns the function object for use.

trainInputFn = makeInputFN(dfTrain, yTrain)
evalInputFn = makeInputFN(dfEval, yTest)
linear_est = tf.estimator.LinearClassifier(featureColumns)



linear_est.train(trainInputFn)
result = linear_est.evaluate(evalInputFn)
clear_output()

#tensorflow models are very good at predicting for large batches of data.

pred = list(linear_est.predict(evalInputFn))
print(dfEval.loc[0])
print(yTest.loc[0])
print(pred[0]["probabilities"][1])


