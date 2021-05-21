# Predict with transformer 
Sample transformer to transform a Tensorflow image, 28x28, to SKLearn mnist model input, 8x8

The following variables need to be updated in transformer_client.py
transformer_addr: the transformer service address
server_addr: the gRPC server address
model_name: the model name for predict
