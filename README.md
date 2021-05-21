# Predict with transformer 
Sample transformer to transform a Tensorflow image, 28x28, to SKLearn mnist model input, 8x8

First start the predictor as described [here](https://github.com/pvaneck/model-serving-sandbox/tree/main/grpc-predict#readme)

Next apply the transformer.yaml to your kube cluster.
```sh
kubedtl apply -f transformer.yaml
```

Update the following variables in transformer_client.py

* transformer_addr: the transformer service address
* server_addr: the gRPC server address
* model_name: the model name of the predictor

Run the client.
```sh
python transformer_client.py
```

Note the inference results will not match all the times due to overly simplistic data transformation. The idea here is to provide a sample case for data transformer.
