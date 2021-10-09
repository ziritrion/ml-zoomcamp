# Question 1
## Steps

`pipenv --version`

## Answer
2021.5.29

# Question 2
## Steps
Check the hash inside `Pipfile.lock`
## Answer
sha256:121f78d6564000dc5e968394f45aac87981fcaaf2be40cfcd8f07b2baa1e1829

# Question 3
## Steps
Run `q3.py`
## Answer
0.115

# Question 4
## Steps
You'll need to run pipenv successfully. Locally, I had to run pipenv inside a Multipass VM.

With pipenv available, inside the `05_deploument` directory, activate the virtual environment:

`pipenv shell`

Run the Flask script (debug):

`python q4.py`

Alternatively, run the Flask script with a WSGI server:

`gunicorn --bind=0.0.0.0:9696 q4:app`

Finally, on another local console, score the customer with a request to the Flask script:

`python q4_response.py`

Or just open the `p4.ipynb` file and run all the cells.

***WARNING***: both `q4.ipynb` and `q4_response.py` might not work in all machines due to using Multipass on my side for the `q4.py` Flask script. Change the IP accordingly.

## Answer

0.998

# Question 5

## Steps
Make sure to have `Dockerfile` in the same directory and run the following:

`docker build -t q5 .`

Check the logs.

On MacOS the output is different, so the Digest is shown instead of the Image ID.
## Answer
Digest:

sha256:1ee036b365452f8a1da0dbc3bf5e7dd0557cfd33f0e56b28054d1dbb9c852023

Image ID:

f0f43f7bc6e0

# Question 6

## Steps
Make sure the docker image is created with `docker images` and check if a container is running with `docker ps`. If not, run the following:

`docker run -it --rm -p 9696:9696 q5`

Once the model is running, on another local terminal run the following:

`python q6_request.py`

Or just open `q6.ipynb` and run all the cells.
## Answer

0.728