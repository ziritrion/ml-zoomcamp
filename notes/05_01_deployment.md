# Deploying Machine Learning Models

## Overview

Jupyter Notebook is useful for prototyping and developing but it's not meant for production environments, because the notebook may contain things such as plots and prints that are useful for understanding our models but are unnecessary in production.

For production environments, we need to ***extract*** the model in a way that can be used by other components such as ***web services***.

Using the _Churn_ exercise from weeks 2 and 3, a production environment for our Churn model could be the following:

![Production example](/notes/05_d01.png)

* `model.bin` file, containing the model extracted from the notebook.
* `Churn service` which exposes the model and allows other components to access it and make predictions.
* `Marketing service` is used by users who want to make predictions. The users input customer data into the service and the service communicates with Churn to request a prediction. Once the prediction is received, the service can execute whichever task is deemed appropiate, suchn as sending emails with offers to potentially churning customers.

## Extracting the model

1. Save whichever variables you may need (such as the model or the one-hot encoding vectorizer) as files. Include the ability to save to files and open files in your code.
1. Export the `i.pynb` file as a `.py` file. Clean it up by moving the imports to the top and removing unnecessary steps such as plots and unnecessary prints.
1. Separate the _train_ and _predict_ methods by creating 2 different `.py` files, each containing each method.