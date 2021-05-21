import requests
import random
import tensorflow as tf
from tensorflow.keras import datasets

dataset = datasets.mnist
(trainX, trainy), (testX, testy) = dataset.load_data()
trainX = trainX[..., tf.newaxis]
trainy = trainy[..., tf.newaxis]
idx=random.randint(0, len(trainX)-1)

transformer_addr="hostname:port"
server_addr="hostname:port"
model_name="example-sklearn-mnist-svm"

res = requests.post('http://'+transformer_addr, 
        json={"instances": trainX[idx].tolist(), "server_addr": server_addr, "model_name": model_name})
print("Expected Result:", trainy[idx][0])
print("Inference Output: ", res.json())
