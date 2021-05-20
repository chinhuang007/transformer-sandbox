from flask import Flask
import grpc_predict_v2_pb2 as pb
import grpc_predict_v2_pb2_grpc as pb_grpc
import grpc

import tensorflow as tf
import flask
import json

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def pre_process():
    input_data = flask.request.json

    # input transformation for model predict
    data = tf.image.resize(input_data['instances'], [8,8])
    data = tf.reshape(data/16, [1,64]).numpy()

    # setup gRPC channel, client, and input
    server_addr = input_data['server_addr']
    channel = grpc.insecure_channel(server_addr)
    #channel = grpc.insecure_channel('169.62.82.164:8033')
    infer_client = pb_grpc.GRPCInferenceServiceStub(channel)

    tensor_contents = pb.InferTensorContents(fp32_contents=data[0])

    infer_input = pb.ModelInferRequest().InferInputTensor(
        name="input-0",
        shape=[1,64],
        datatype="FP32",
        contents=tensor_contents
    )
    metadata = (('mm-vmodel-id','example-sklearn-mnist-svm'),)
    inputs = [infer_input]
    request = pb.ModelInferRequest(model_name="example-sklearn-mnist-svm", inputs=inputs)

    # send inference request
    results, call = infer_client.ModelInfer.with_call(request=request, metadata=metadata)
    print(results)
    return json.dumps({"prediction": int(results.outputs[0].contents.fp32_contents[0])})


if __name__ == "__main__":
    app.run(host='0.0.0.0')
