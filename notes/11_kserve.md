# Introduction to KServe

***[KServe](https://github.com/kserve/kserve)*** provides a Kubernetes _Custom Resource Definition_ for serving ML models on arbitrary frameworks.

A Kubernetes _Resource Definition_ is a YAML file that defines a _resource_ such as a deployment or a service, as seen in [lesson 10](10_kubernetes.md). Kubernetes allows creating new types of resources for additional abstraction.

In other words: KServe simplifies the deployment of Kubernetes apps by using shorter custom YAML files that take care of much of the work for us: with plain Kubernetes we needed 4 separate YAML files for our 2-component app, but with KServe we will only need 1. KServe works with many ML frameworks such as TensorFlow, PyTorch, XGboost, etc.

KServe used to be part of a bigger toolkit called [Kubeflow](https://www.kubeflow.org/) but it's become an independent project. Kubeflow is a toolkit for managing the complete ML lifecycle from development to deployment on top of Kubernetes, but KServe only focuses on serving models.

KServe is structured around apps being designed with a ***two tier architecture***. In other words; apps served with KServe must have 2 main components: ***transformers*** and ***predictors***, which fulfill similar roles to our gateway and model server from the previous lesson.

![two tier architecture](images/11_01.png)

# Running KServe locally

## Installing Kserve locally with Kind
## Deploying an example model

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