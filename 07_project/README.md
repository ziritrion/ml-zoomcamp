This folder contains all the required code for the midterm project of Alexey Grigorev's ML Zoomcamp.

# Dataset Description

_Original dataset [found here](https://www.openml.org/d/40536)_

This data was gathered from participants in experimental speed dating events from 2002-2004. During the events, the attendees would have a four-minute "first date" with every other participant of the opposite sex. At the end of their four minutes, participants were asked if they would like to see their date again. They were also asked to rate their date on six attributes: Attractiveness, Sincerity, Intelligence, Fun, Ambition, and Shared Interests. The dataset also includes questionnaire data gathered from participants at different points in the process. These fields include: demographics, dating habits, self-perception across key attributes, beliefs on what others find valuable in a mate, and lifestyle information.

### Attribute Information
```
* gender: Gender of self
* age: Age of self
* age_o: Age of partner
* d_age: Difference in age
* race: Race of self
* race_o: Race of partner
* samerace: Whether the two persons have the same race or not.
* importance_same_race: How important is it that partner is of same race?
* importance_same_religion: How important is it that partner has same religion?
* field: Field of study
* pref_o_attractive: How important does partner rate attractiveness
* pref_o_sinsere: How important does partner rate sincerity
* pref_o_intelligence: How important does partner rate intelligence
* pref_o_funny: How important does partner rate being funny
* pref_o_ambitious: How important does partner rate ambition
* pref_o_shared_interests: How important does partner rate having shared interests
* attractive_o: Rating by partner (about me) at night of event on attractiveness
* sincere_o: Rating by partner (about me) at night of event on sincerity
* intelligence_o: Rating by partner (about me) at night of event on intelligence
* funny_o: Rating by partner (about me) at night of event on being funny
* ambitous_o: Rating by partner (about me) at night of event on being ambitious
* shared_interests_o: Rating by partner (about me) at night of event on shared interest
* attractive_important: What do you look for in a partner - attractiveness
* sincere_important: What do you look for in a partner - sincerity
* intellicence_important: What do you look for in a partner - intelligence
* funny_important: What do you look for in a partner - being funny
* ambtition_important: What do you look for in a partner - ambition
* shared_interests_important: What do you look for in a partner - shared interests
* attractive: Rate yourself - attractiveness
* sincere: Rate yourself - sincerity
* intelligence: Rate yourself - intelligence
* funny: Rate yourself - being funny
* ambition: Rate yourself - ambition
* attractive_partner: Rate your partner - attractiveness
* sincere_partner: Rate your partner - sincerity
* intelligence_partner: Rate your partner - intelligence
* funny_partner: Rate your partner - being funny
* ambition_partner: Rate your partner - ambition
* shared_interests_partner: Rate your partner - shared interests
* sports: Your own interests [1-10]
* tvsports
* exercise
* dining
* museums
* art
* hiking
* gaming
* clubbing
* reading
* tv
* theater
* movies
* concerts
* music
* shopping
* yoga
* interests_correlate: Correlation between participant’s and partner’s ratings of interests.
* expected_happy_with_sd_people: How happy do you expect to be with the people you meet during the speed-dating event?
* expected_num_interested_in_me: Out of the 20 people you will meet, how many do you expect will be interested in dating you?
* expected_num_matches: How many matches do you expect to get?
* like: Did you like your partner?
* guess_prob_liked: How likely do you think it is that your partner likes you?
* met: Have you met your partner before?
* decision: Decision at night of event.
* decision_o: Decision of partner at night of event.
* match: Match (yes/no)
```

### Relevant paper

Raymond Fisman; Sheena S. Iyengar; Emir Kamenica; Itamar Simonson.
Gender Differences in Mate Selection: Evidence From a Speed Dating Experiment.
The Quarterly Journal of Economics, Volume 121, Issue 2, 1 May 2006, Pages 673–697,
[https://doi.org/10.1162/qjec.2006.121.2.673](https://doi.org/10.1162/qjec.2006.121.2.673)

# Project Description

For this midterm project, a binary classification model was trained on the Speed Dating dataset in order to predict the `match` feature.

3 models were trained: a Decision Tree, a Random Forest and a Gradient Boosting model. Out of the 3, Gradient Boosting was the model with the better performance (using the XGBoost library). A pretrained model is provided in the file `model.bin`, which can be loaded with `pickle`.

The exploratory data analysis and model selection was done with the help of a Jupyter Notebook, `notebook.ipynb`.

The model training script was exported to `train.py`.

A Flask app was created in `predict.py`, which can be deployed with any WSGI server. This project has been developed and tested with Gunicorn, but any other WSGI server should work as well.

Additional requirement files are provided for local virtual environment set up and deployment.

# Files

* `README.md`: the file you're reading right now.
* `speeddating.csv`: the CSV file containing the Speed Dating dataset.
* `notebook.ipynb`: a Jupyter Notebook containing all of the Exploratory Data Analysis and model building.
* `train.py`: a training script. It will train the best model found on `notebook.ipynb` and store 2 files: `dv.bin` and `model.bin`. These 2 files are already provided in this repo; running `train.py` should overwrite these 2 files with new but identical ones, due to the code defining a seed for its random state.
* `predict.py`: Flask app that receives a query and outputs a prediction.
* `model.py`: pre-trained gradient boosting model.
* `dv.bin`: a pre-made DictVectorizer object, necessary for creating the data structures needed for prediction.
* `requirements.txt`: requirements file for `predict.py` deployment.
* `requirements-all.txt`: requirements file for running everything in this folder.
* `environment.yml`: environment file for creating a new Conda virtual environment with the packages listed in `requirements.txt`.
* `environment-all.yml`: environment file for creating a new Conda virtual environment with the packages listed in `requirements-all.txt`.
* `Dockerfile`: a dockerfile for containerizing the Flask app.
* `Dockerfile-old`: a second dockerfile provided for convenience.
* `scratch.ipynb`: a simple scratch notebook with code for testing. Useful to send requests to `predict.py`.

# Run the code

Conda is used for package dependency and virtual environment management. The code has been tested on MacOS (Intel) and Linux.

Please follow these steps to run the code.

1. Instal Conda on your platform. Follow [these steps](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#) if you have not installed it before.
1. Create a new Conda virtual environment. 2 methods are provided for 2 different environments:
    1. Method 1: using YML files. The virtual environment names are defined by these files (make sure they don't conflict with existing virtual environment names).
        * Create a new `match` virtual environment with all the necessary dependencies to run all the code contained here.
            * `conda env create -f environment-all.yml`
        * Create a new `matchdeploy` virtual environment with only the necessary dependencies for `predict.py`.
            * `conda env create -f environment.yml`
        * Activate your environment before going to the next step (choose one of the following two).
            * `conda actiave match`
            * `conda activate matchdeploy`
    1. Method 2: using the requirement TXT files. This method allows for custom environment names.
        1. Create a new virtual environment with Python 3.8. The example code will use the name `match` for the virtual environment; feel free to replace it with any other name.
            * `conda create --name match python=3.8`
        1. Activate the environment.
            * `conda activate match`
        1. Install the dependencies. 2 dependency files are provided: `requirements-all.txt` contains all the needed requirements to run all of the code locally, and `requirement.txt` contains only the dependencies needed to run and deploy `predict.py` on Docker or any virtual environment.
            * Install all the requirements.
                * `conda install --file requirements-all.txt -c conda-forge`
            * Install only the necessary requirements for deployment.
                * `conda install --file requirements.txt -c conda-forge`
1. Run `predict.py` either by itself (for testing) or with a WSGI server for deployment. The code and these instructions asume that Gunicorn will be used for deployment.
    * Run the code for quick testing:
        * `python predict.py`
    * Run the code on a WSGI server (Gunicorn):
        * `gunicorn --bind 0.0.0.0:9696 predict:app`
    * Stop running either of these by typing `CTRL + C` on your keyboard.
1. Use `scratch.ipynb` for testing `predict.py`. You may need to change the `url` variable to make it work locally; check the output of `python predict.py` to know your local IP address.
1. After you're done, you may deactivate your virtual environment with:
    * `conda deactivate`.

# Docker

A Dockerfile is provided for building a Docker image for deployment of the Flask app. Because Conda has such a huge overhead, a multi-stage build is used to reduce the final image size. A second Dockerfile is provided as well in case the multi-stage build causes any problems.

Please follow these steps to build and run the containerized `predict.py` app.

1. [Install Docker](https://docs.docker.com/get-docker/).
2. Build the image. You may change the name `matchdeploy` to anything else, but don't remove the period at the end.
    * `docker build -f Dockerfile -t matchdeploy .`
3. Once the image is built, run a container with mapped ports 9696:9696 (or change the external port to anything you may prefer)
    * `docker run -it --rm -p 9696:9696 matchdeploy:latest`
4. You may stop the container by typing `CTRL + C` on your keyboard.

A second dockerfile, `Dockerfile-old`, is provided for building an image in a regular single-stage build. The resulting image will be over 1GB in size, so this method is not recommended unless absolutely necessary.

# Acknowledgments

[Multi-stage Docker Builds](https://pythonspeed.com/articles/conda-docker-image-size/)