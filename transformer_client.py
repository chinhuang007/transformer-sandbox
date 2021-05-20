import requests
import random
import tensorflow as tf
from tensorflow.keras import datasets

dataset = datasets.mnist
(trainX, trainy), (testX, testy) = dataset.load_data()
trainX = trainX[..., tf.newaxis]
trainy = trainy[..., tf.newaxis]
idx=random.randint(0, len(trainX)-1)

transformer_addr="169.44.151.84:6000"
server_addr="169.62.82.164:8033"

res = requests.post('http://'+transformer_addr, json={"instances": trainX[idx].tolist(), "server_addr": server_addr})
print("Expected Result:", trainy[idx][0])
print("Inference Output: ", res.json())

'''
for idx in range(20):
  res = requests.post('http://'+transformer_addr, json={"instances": trainX[idx].tolist(), "server_addr": server_addr})
  print("expected:", trainy[idx])
  print(res.json())
'''
