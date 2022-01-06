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
    * `storageUri` links to the model. The example file links to an example model hosted on Google Cloud, thus the `gs` scheme (_Google Storage_).

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



# Deploying a Scikit-Learn model with KServe

## Training the churn model with a specific Scikit-Learn version
## Deploying the churn prediction model with KServe

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