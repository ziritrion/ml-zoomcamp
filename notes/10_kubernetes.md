> Previous: [Serverless Deep Learning](09_serverless.md)

> [Back to Index](README.md)

> Next: _(Coming soon)_

# Kubernetes and TensoFlow-Serving overview

![overview](images/10_01.png)

***Kubernetes*** is a _container orchestration system_ used to automatically deploy, scale and operate containers.

In this lesson we will use Kubernetes along with ***TensorFlow Serving***, a component of the TensorFlow Extended (TFX) family of technologies used to deploy models in production environments.

In the image above you can see the main architecture we will create during the lesson. The _gateway_ will be created with Flask and will take care of downloading images, resizing them, preparing the input for the models and post-process the model outputs. The _model server_ will be used for inference only and will receive for input an image in the form of a numpy array and will return a prediction using `gRPC` to the gateway.

The gateway only uses CPU computing and needs less resources than the model server, which makes use of GPU computing. By decoupling these 2 components, we can run them in separate containers and in different numbers (for example, 5 gateway containers and 2 model server containers, as shown in the image), allowing us to optimize resources and lower deployment costs.

# TensorFlow-Serving

## Model format and signature definition

TensorFlow Serving (`tf-serving`) requires models to be in a specific format.

    import tensorflow as tf
    from tensorflow import keras

    model = keras.models.load_model('original_model.h5')

    tf.saved_model.save(model, 'converted_model')

This code loads a Keras model and simply saves it to Tensorflow format rather than Keras.

Tensorflow-formatted models are not single files; they are directories containing a number of files. Here's an example model directory structure:

    converted_model
    ┣╸ assets
    ┣╸ saved_model.pb
    ┗╸ variables
        ┣╸ variables.data-00000-of-00001
        ┗╸ variables.index

The `saved_model_cli` utility allows us to inspect the contents of the tf model:
* `saved_model_cli show --dir converted_model --all`

We're specifically interested in the ***signature definition*** of the model. The signature definition describes both the inputs and the outputs of the model. Here's an example of signature definition:

    signature_def['serving_default']:
        The given SavedModel SignatureDef contains the following input(s):
            inputs['input_8'] tensor_info:
                dtype: DT_FLOAT
                shape: (-1, 299, 299, 3)
                name: serving_default_input_8:0
        The given SavedModel SignatureDef contains the following output(s):
            outputs['dense_7'] tensor_info:
                dtype: DT_FLOAT
                shape: (-1, 10)
                name: StatefulPartitionedCall:0
        Method name is: tensorflow/serving/predict

> WARNING: the output of `saved_model_cli` contains many more things than the signature which can be easily mixed. Make sure that you're looking at `signature_def['serving_default']` rather than `signature_def['__saved_model_init_op']`

In the signature above we can see that our example model has 1 input called `input_8` and 1 output called `dense_7`. Both of these names as well as the name of the signature definition (in our example, `serving_default`) are necessary for the next step.

## Running a container with a tf-serving model

TensorFlow-Serving has an official Docker image ready for deployment. By using ***volumes*** (folders in the host machine that can be mounted to containers in a manner similar to external storage) we can deploy our model in a volume and attach it to a container without the need to rebuild a new image, thus reducing the size of the Docker image and therefore reducing costs.

We can run the official tf-serving Docker image with our model mounted in a volume with the following command:

    docker run -it --rm \
        -p 8500:8500 \
        -v "$(pwd)/converted_model:/models/converted_model/1" \
        -e MODEL_NAME="converted_model" \
        tensorflow/serving:2.7.0

* `-it`: short for `-i -t`
    * `-i`: interactive mode
    * `-t`: open a terminal
* `--rm`: when the interaction with the container ends, using this flag means that the container will be deleted.
* `\`: allows to escape the Enter key and create multi-line shell commands.
* `-p 8500:8500`: maps port 8500 in the host to 8500 in the container. We use the port 8500 because the official tf-serving image uses it.
* `-v "$(pwd)/converted_model:/models/converted_model/1"`
    * `-v`: volume. Maps a folder in the host (first half of the path) to a folder within the container (second half).
    * Both folder paths need to be absolute, so relative paths like `"./converted_model"` won't work. We can use a convenient trick with the `pwd` command to print our current folder: `$(pwd)` will be replaced with the current folder of the host machine.
    * The path to the container folder ends with a version number. If no version number is provided the `docker run` command will fail. You can just put `/1` all the time if you have other means of controlling model versions.
* `-e MODEL_NAME="converted_model"`: environment variable. The container makes use of an environment variable called `MODEL_NAME` which must match the name of our model.
* `tensorflow/serving:2.7.0` is the Docker image we will we using.

## Testing the container with Jupyter Notebook

With the Docker container running, we can now test it from a Jupyter Notebook.

    !pip install grpcio==1.42.0 tensorflow-serving-api==2.7.0
    !pip install keras-image-helper

    import grpc
    import tensorflow as tf
    from tensorflow_serving.apis import predict_pb2
    from tensorflow_serving.apis import prediction_service_pb2_grpc

    host = ' localhost:8500'
    channel = grpc.insecure_channel(host)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

tf-serving makes use of gRPC, a framework for connecting services in and across datacenters. gRPC uses _Protocol Buffers_ (AKA ***Protobuf***) for formatting data, a kind of binary encoding which is faster and more efficient than JSON.

gRPC can establish both insecure and secure communication channels. Secure channels can make use of authentication and other advanced security features, but since our model server won't be accessible from the outside, insecure channels are adequate for our use case.

The _stub_ is our interface with the tf-serving in order to make inference with our model. It needs a channel as a parameter to establish the communication. 

    from keras_image_helper import create_preprocessor

    preprocessor = create_preprocessor('xception', target_size=(299, 299))
    url = 'http://bit.ly/mlbookcamp-pants'
    X = preprocessor.from_url(url)

This is the same kind of code seen in [lesson 9](09_serverless.md) for transforming the image into a numpy array that our model can process.

    def np_to_protobuf(data):
        return tf.make_tensor_proto(data, shape=data.shape)

    pb_request = predict_pb2.PredictRequest()
    pb_request.model_spec.name = 'converted_model'
    pb_request.model_spec.signature_name = 'serving_default'
    pb_request.inputs['input_8'].CopyFrom(np_to_protobuf(X))

We can now set up our request to our model by instancing a Protobuf request object and defining its model name, the model's signature name that we saw before and its input.

For the input, note that we make use of the input name that we found in the signature. We also convert the numpy array of our image to Protobuf format and copy it to the request object.

    pb_response = stub.Predict(pb_request, timeout=20.0)

    pb_response.outputs['dense_7'].float_val

Inference is done with the stub's `Predict()` method. We pass out request object as a parameter and define a timeout as well.

The `Predict()` method returns a Protobuf response object. We can access our predictions with the name of the output that we found in the signature definition. `float_val` returns the predictions as a regular Python list, so there is no need to do additional conversions.

    classes = [
        'dress',
        'hat',
        #...
        'pants'
    ]

    dict(zip(classes, preds))

This code is just a convenience to tag each prediction value to the class they belong to.

Incidentally, this Jupyter Notebook code will be the basis for our gateway Flask app.

# Creating a pre-processing service

Remember that you can export a Jupyter Notebook to a Python script with the following command:

    jupyter nbconvert --to script notebook.ipynb

This command will output a `.py` file with the same name as the notebook.

We can rename the script to `gateway.py`, clean it up, organize the code in methods and add a `if __name__ == '__main__'` statement in order to convert the script to a Flask app.

>Note: a Flask cheatsheet is available [in this link](https://gist.github.com/ziritrion/9b80e47956adc0f20ecce209d494cd0a)

There is one issue: in the notebook we defined the following function:

    def np_to_protobuf(data):
        return tf.make_tensor_proto(data, shape=data.shape)

The `make_tensor_proto()` method is a TensorFlow method and TensorFlow is a huge library about 2GB in size. A smaller `tensorflow-cpu` library exists but it still is over 400MB in size.

Since we only need to use that particular method, we can instead make use of a separate [`tensorflow-protobuf`](https://github.com/alexeygrigorev/tensorflow-protobuf) package which is available on pip.

    !pip install tensorflow-protobuf==2.7.0

The [GitHub page for `tensorflow-protobuf`](https://github.com/alexeygrigorev/tensorflow-protobuf) contains info on how to replace the `make_tensor_proto()` method.

Since the additional code is wordy, it would be convenient to define the `np_to_protobuf()` method on a separate `proto.py` script and then import it to the gateway app with `from proto import np_to_protobuf`.

# Running everything locally with Docker-compose

# Intro to Kubernetes

# Deploying a simple service to Kubernetes

# Deploying TensorFlow models to Kubernetes

# Deploying to EKS

> Previous: [Serverless Deep Learning](09_serverless.md)

> [Back to Index](README.md)

> Next: _(Coming soon)_