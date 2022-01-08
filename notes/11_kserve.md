# Introduction to KServe

***[KServe](https://github.com/kserve/kserve)*** provides a Kubernetes _Custom Resource Definition_ for serving ML models on arbitrary frameworks.

A Kubernetes _Resource Definition_ is a YAML file that defines a _resource_ such as a deployment or a service, as seen in [lesson 10](10_kubernetes.md). Kubernetes allows creating new types of resources for additional abstraction.

In other words: KServe simplifies the deployment of Kubernetes apps by using shorter custom YAML files that take care of much of the work for us: with plain Kubernetes we needed 4 separate YAML files for our 2-component app, but with KServe we will only need 1. KServe works with many ML frameworks such as TensorFlow, PyTorch, XGboost, etc.

KServe used to be part of a bigger toolkit called [Kubeflow](https://www.kubeflow.org/) but it's become an independent project. Kubeflow is a toolkit for managing the complete ML lifecycle from development to deployment on top of Kubernetes, but KServe only focuses on serving models.

KServe is structured around apps being designed with a ***two tier architecture***. In other words; apps served with KServe must have 2 main components: ***transformers*** and ***predictors***, which fulfill similar roles to our gateway and model server from the previous lesson.

![two tier architecture](images/11_01.png)

# Running KServe locally

## Installing Kserve locally

In order to run Kserve locally you will need to [install Kind](https://kind.sigs.k8s.io/docs/user/quick-start/) as well as [install kubectl](https://kubernetes.io/docs/tasks/tools/). If you followed [lesson 10](10_kubernetes.md) you should already have both.

You can install the Kserve Quickstart environment by [following the instructions in this link](https://kserve.github.io/website/0.7/get_started/). Make sure that a local Kind cluster is running before installing Kserve. The installation process will take several minutes.

>IMPORTANT: Kserve is installed on top of a cluster. Whenever you create a new one you will have to install Kserve on top of it.

## Deploying an example model

Once Kserve has been successfully installed, you may follow [these steps](https://kserve.github.io/website/0.7/get_started/first_isvc/) to install and test an example service.

Here's the test `iris-example.yaml` file:

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "sklearn-iris"
spec:
  predictor:
    sklearn:
      storageUri: "gs://kfserving-samples/models/sklearn/iris"

```

* The `kind` field is `InferenceService`, a custom resource that Kserve provides.
* `metadata.name` contains the name of the `InferenceService` (_`isvc`_ for short).
* `spec` contains the actual content of the file:
    * `predictor` is the Kserve component defined in the file, which will serve our model and output the predictions.
    * `sklearn` is the runtime we use for this particular model. We specify it here in order to let Kserve know how to handle the model internally. A list of available runtimes is available [here](https://kserve.github.io/website/0.7/modelserving/v1beta1/serving_runtime/).
    * `storageUri` links to the model (not a Docker image, but the actual model file!). The example file links to an example model hosted on Google Cloud, thus the `gs` scheme (_Google Storage_).

We can now apply the isvc resource:

```sh
kubectl apply -f iris-example.yaml
```

Let's now list all the available instances of `InferenceService`:

```sh
kubectl get inferenceservice
# or alternatively,
kubectl get isvc
```

If the isvc was successfully applied, it should display the state `READY = True` and display a public URL.

The public URL always follows the following pattern:

```sh
http://${SERVICE_NAME}.${NAMESPACE}.${DOMAIN}
```

In our example, since we didn't specify a namespace, the URL will be `http://sklearn-iris.default.example.com`

Kserve makes use of [Istio](https://istio.io/), a [_service mesh_](https://www.redhat.com/en/topics/microservices/what-is-a-service-mesh) that adds additional networking and monitoring capabilities to Kubernetes. When creating the InferenceService, Kserve (by means of Istio) created an _ingress service_, a public-facing load balancer for the app. You should see a `istio-ingressgateway` LoadBalancer service when typing the following command:

```sh
kubectl get service -n istio-system
# The istio-ingressgateway should be listed along with additional services
```

We have to specify the namespace `istio-system` in order to list the ingress; thus the `-n istio-system` part of the command.

The external IP will most likely be `<pending>` because we haven't configured Kubernetes to provide an external load balancer for the ingress gateway. However, we can make use of port forwarding to check whether the app is working or not:

```sh
kubectl port-forward -n istio-system service/istio-ingressgateway 8080:80
```

Now let's prepare a `iris-input.json` test input file:

```json
{
  "instances": [
    [6.8,  2.8,  4.8,  1.4],
    [6.0,  3.4,  4.5,  1.6]
  ]
}
```

This request has 2 observations, so we expect 2 predictions.

Since we do not have DNS, we will curl with the ingress gateway external IP using the HOST Header:

```sh
curl -v -H "Host: ${HOST}" ${URL} -d @./iris-input.json
```
* `${HOST}` is the `sklearn-iris.default.example.com` URL we got from using `kubectl get isvc`.
    * We don't have access to `example.com` but Kserve still needs to know it because it needs to be able to know which InferenceService we want to send the request to (imagine we had several isvc's in our cluster; when the ingress gateway receives the request it needs to know where to send it!).
* `${URL}` is the actual external URL of the InferenceService in our hosted Kubernetes which we need to send the predict request to. If you're running Kubernetes on localhost, it will have the following format:
    * `http://localhost:8080/v1/models/${SERVICE_NAME}:predict`
    * Remember that ${SERVICE_NAME} in our particular case is `sklearn-iris`.
    * `localhost` is our `INGRESS_HOST` and `8080` is our `INGRESS_PORT`.
    * `${SERVICE_NAME}` is `sklearn-iris`.
    * In Kserve, the convention is to use a colon (`:`) to separate the path from the URL, which is why we have 
* In `curl`, we can specify an HTTP header with the `-h` option. We use it to send the Host info for the Ingress Gateway to know which isvc needs to receive the user's request.
* `-v` is the verbose option. You may omit it if you want.
* `-d` specifies that the data we want to send is inside a file.

You should receive something like this:

```sh
{"predictions": [1, 1]}
```

Alternatively, you can also use a python file to send a request instead of using `curl`:

```python
import requests

service_name = 'sklearn-iris'
host = f'{service_name}.default.example.com'
actual_domain = 'http://localhost:8080'
url = f'{actual_domain}/v1/models/{service_name}:predict'

headers = {
    'Host': host
}

request = {
    "instances": [
      [6.8,  2.8,  4.8,  1.4],
      [6.0,  3.4,  4.5,  1.6]
    ]
  }

response = requests.post(url, json=request, headers=headers)

print(response.json())
```

# Deploying a Scikit-Learn model with KServe

In the previous block we already deployed an example Scikit-Learn model, but in this block we will train a new model and deploy it to explore the specifics.

## Training the churn model with a specific Scikit-Learn version

We will make use of the [churn model we saw in lesson 5](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/05-deployment/code/05-train-churn-model.ipynb).

Kserve includes a _Scikit-Learn Server_ component that requires a specific Scikit-Learn library and Python versions. [In this folder of the Kserve repo](https://github.com/kserve/kserve/tree/master/python) you will find the [`sklearn.Dockerfile`](https://github.com/kserve/kserve/blob/master/python/sklearn.Dockerfile) that contains the Python version we need (Docker image `python:3.7-slim` at the time of writing these notes), and [in the folder for the `sklearnserver`implementation](https://github.com/kserve/kserve/tree/master/python/sklearnserver) you will wind the [`setup.py`](https://github.com/kserve/kserve/blob/master/python/sklearnserver/setup.py) that contains the dependencies we need, which at the time of writing are:
* `kserve` 0.7.0
* `scikit-learn` 0.20.3
* `joblib` 0.13.0

We will use Conda to create a virtual Python environment with these library dependencies. Pipenv is good for managing library versions but trickier for Python versions, so we will use Conda in this instance:

```sh
conda create -n py37-sklearn-0.20.3 python=3.7 scikit-learn==0.20.3 pandas joblib
```
* We don't bother specifiying library versions for the other dependencies because they're not important in this case.

We will now train the churn model again within this new environment based on the [05-train-churn-model.ipynb](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/course-zoomcamp/05-deployment/code/05-train-churn-model.ipynb) Jupyter Notebook. We will export the notebook to a [`churn-train.py`](../11_kserve/churn/churn-train.py) file and make a few small changes:
* No need to import `train_test_split` nor `KFold`.
* We will read the data from a URL rather than a local file.
* No need to split the dataset; we will train on the whole dataset but we will only use the numerical features `tenure` and `monthlycharges`, and the categorical `contract`.
* We will use `joblib` rather than `pickle`. Import it and save the model with `joblib.dump()`.
    * The sklearn-server needs a specific name for the model. Save it as `model.joblib`.
*  The sklearn-server will only call the `predict()` method on our model but we need to fit a `DictVectorizer` as well as the model. We can use [sklearn's pipelines](07_misc.md#scikit-learn-pipelines) to fit both things within a single pipeline object that we can later dump as a file; that way sklearn-serve will call `predict()` on our pipeline and fit the vectorizer as well as the model.

Activate the environment and run the script to save the model. You may now deactivate the environment since we won't be using it anymore.

## Deploying the churn prediction model with KServe

We now need to create a `churn-service.yaml` InferenceService file:

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "churn"
spec:
  predictor:
    sklearn:
      storageUri: "http://172.31.13.90:8000/churn/model.joblib"
```

* We will call the isvc `churn` in `metadata.name`.
* In `spec.predictor.sklearn.storageUri` we need to provide a link to our model. You can upload it to a GitHub repo for convenience. In this example we use Python's built in web server to locally host the model and put the URL there.
    * In order to run Python's built-in web server, go to your model's directory on a terminal and run:
    * `python -m http.server`
    * The contents of the folder will be available on `http://localhost:8000`.
    * Kserve won't be able to make use of the `localhost` name to retrieve the model, so you will need to substituite it with the IP address of your machine. You can find it with `ifconfig`/`ipconfig`.
    * Find out more info about using storage URI's [in this link](https://github.com/kserve/kserve/tree/master/docs/samples/storage/uri).

Deploy the isvc:

```ssh
kubectl apply -f churn-service.yaml
```

Now let's test if the isvc deployed successfully with our model:
1. Get the pod name with `kubectl get pods` .
1. Check the logs of the pod with `kubectl logs <pod_name>` . At the end of the logs the line `Registering model: churn` should appear.
    * If the pod contains multiple containers, Kubernetes will complain and offer you a list of available containers. In this case, use `kubectl logs <pod_name> <container_name>` .
1. Now login to the pod:
    * `kubectl exec -it <pod_name> <container_name> --bash`
1. Navigate to the model directory with `cd mnt/models/` and check the contents with `ls` . The `model.joblib` file should be there.
1. Open a python interactive terminal with `python` and run the following lines:
```python
import joblib

model = joblib.load('model.joblib')

# get a few examples to run from the original dataset on a separate Notebook or whatever. You can also use these:
X = = [{'contract': 'one_year', 'tenure': 34, 'monthlycharges': 56.95}]

model.predict(X)
# you should get a binary prediction

model.predict_proba(X)
# you should get a prediction probability

exit()
```
Alternatively, you can also create a [`churn-test.py` file](../11_kserve/churn/churn-test.py) in a similar fashion to the script we created in the previous block as an alternative to `curl` and run it.

In the next block we'll see how to use custom Python and Scikit-Learn versions.

# Deploying custom Scikit-Learn images with KServe

## Customizing the Scikit-Learn image
## Running KServe service locally

# Serving TensorFlow models with KServe

## Converting the Keras model to saved_model format
## Deploying the model
## Preparing the input

# KServe transformers

## Why do we need transformers
## Creating a service for pre and post processing
## Using existing transformers

# Deploying with KServe and EKS

## Creating an EKS cluster
## Installing KServe on EKS
## Configuring the domain
## Setting up S3 access
## Deploying the clothing model