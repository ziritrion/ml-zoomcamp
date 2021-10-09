FROM agrigorev/zoomcamp-model:3.8.12-slim
# add your stuff here

# (pipenv) install pipenv
RUN pip install pipenv

# set the working directory in the container
# It's already specified by the base image, so this is unnecessary
# WORKDIR /app

# (pipenv) copy dependencies to the working directory
COPY ["Pipfile", "Pipfile.lock", "./"]

# (pipenv) Install dependencies
# (pipenv) We don't need a virtualenv in Docker, so we can install dependencies to the system
RUN pipenv install --system --deploy

# Copy the files we need
# WARNING! The base image already contains a model and a vectorizer!
# We only need the script!
COPY ["q5.py", "./"]

# Expose a port on the container
# Remember to map the port to a port in the host when running the container!
EXPOSE 9696

# Specify entrypoint
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "q5:app"]

