import tensorflow as tf
import pandas as pd
""" 
classification is the process of classifying a data point by generating the 
probability that it is within a given class.

We will be using the iris flower dataset.
the species are:
    - setosa
    - versicolor
    - virginica
    
the information:
    - sepal length
    - sepal width
    - petal length
    - petal width
"""
csvColumnNames = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth","Species"]
SPECIES = ["Setosa", "Versicolor", "Virginica"]
# defining some constants:

trainPath = tf.keras.utils.get_file("irisTraining.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
testPath = tf.keras.utils.get_file("irisTest.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
train = pd.read_csv(trainPath, names=csvColumnNames, header=-0)
test = pd.read_csv(testPath,names= csvColumnNames, header=-0)
print(train.head())
# the species are pre-encoded 0 - setosa, 1- versicolor, 2 - virginica

yTrain = train.pop("Species")
yTest = test.pop("Species")


def inputFn (features, labels, training=True, batchSize=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batchSize)

featureColumns = []
for key in train.keys():
    featureColumns.append(tf.feature_column.numeric_column(key=key))


"""
some Classifiers:
    - DNNClassifier - Deep Neural Network
    - LinearClassifier - similar to linear regressor however for classification
"""

classifier = tf.estimator.DNNClassifier(
    feature_columns=featureColumns,
    hidden_units=[30,10],
    n_classes=3
)

classifier.train(input_fn= lambda: inputFn(train, yTrain, training=True), steps=5000)
# 5000 steps means go thru the dataset till 5000 records have been viewed.

eval_result = classifier.evaluate(input_fn=lambda:inputFn(test, yTest,False))
print('\n test accuracy: {accuracy:0.3f}\n'.format(**eval_result))

def inputFn(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices((dict(features))).batch(batch_size)
features = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
predict = {}
print("Please type numeric values as prompted.")
for feature in features:
    valid = True
    while valid:
        val = input(feature + ":")
        if not val.isdigit(): valid = False
    predict[feature] = [float(val)]

prediction = classifier.predict(input_fn=lambda: inputFn(predict))
for pred_dict in prediction:
    classId = pred_dict["class_ids"][0]
    probability = pred_dict["probabilities"][classId]
    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[classId],100*probability
    ))

