This folder contains all the required code for the [capstone project of Alexey Grigorev's ML Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/12-capstone).

# Dataset Description

_Original dataset [found here](https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition)_

This dataset contains images of the following food items:

* Fruits: banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.
* Vegetables: cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.

This dataset contains three folders:

* train (100 images each)
* test (10 images each)
* validation (10 images each)

Each of the above folders contains subfolders for different fruits and vegetables wherein the images for respective food items are present

# Project Description

For the capstone project I decided to work on a project involving Deep Learning. I opted for a simple ***image recognition task*** due to time constraints and inexperience.

This is a simple dataset containing 36 different classes of fruits and vegetables. The dataset is very small, which makes training a network from scratch unfeasible. Thus, the project relies exclusively on Transfer Learning for image feature extraction.

3 different convolutional architectures were used:

* `Xception`: one of the most popular architectures available on Keras.
* `MobileNet_v2`: a very lean and efficient architecture with excellent performance to model size ratio.
* `NasNetLarge`: One of the largest models available on Keras besides `VGG`. Chosen for its large size but excellent accuracy as listed on the [Keras Applications website](https://keras.io/api/applications/).

# Files

# Environment setup

The Jupyter Notebook and Python scripts were all tested on Ubuntu 20.04 Intel x64.

## Pip

You may install all dependencies with `pip` with the following command:

`pip install -r pip-requirements.txt`

> Note: `pip-requirements.txt` was created with the following command:

> `pip list --format=freeze > pip-requirements.txt`

## Conda

2 alternative methods are provided:

1. Create an environment named `tf` with the provided `conda-environment.yml` file:

    * `conda env create -f conda-environment.yml`
    * You can change the name of the environment by modifying the first line of the `conda-environment.yml` file. Modify and save the file before creating the environment!

1. Create an environment with a name of your choosing with all the dependencies listed in `conda-requirements.txt`:
    * `conda create --name <env_name> --file conda-requirements.txt`
    * `<env_name>` may be any name you choose.

> Note: `conda-environment.yml` was created with the following command:

> `conda env export --from-history > conda-environment.yml`

> `conda-requirements.txt` was created with the following command:

> `conda list -e > conda-requirements.txt`



# Docker

# Deployment