# The build-stage image:
FROM continuumio/miniconda3 AS build

# Install the package as normal:
COPY environment.yml .
RUN conda env create -f environment.yml

# Install conda-pack:
RUN conda install -c conda-forge conda-pack

# Use conda-pack to create a standalone enviornment
# in /venv:
RUN conda-pack -n matchdeploy -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

# We've put venv in same path it'll be in final image,
# so now fix up paths:
RUN /venv/bin/conda-unpack

# The runtime-stage image; we can use Debian as the
# base image since the Conda env also includes Python
# for us.
FROM debian:buster AS runtime

# Copy /venv from the previous stage:
COPY --from=build /venv /venv

# set the working directory in the container
WORKDIR /app

 # Copy the model
COPY ["predict.py", "model.bin", "dv.bin", "./"]

# Expose a port on the container
# Remember to map the port to a port in the host when running the container!
EXPOSE 9696

# When image is run, run the code with the environment
# activated:
SHELL ["/bin/bash", "-c"]
# Specify entrypoint
ENTRYPOINT source /venv/bin/activate && \
    gunicorn --bind 0.0.0.0:9696 predict:app