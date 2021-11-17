# The build-stage image:
FROM continuumio/miniconda3

# Install the package as normal:
COPY environment.yml .
RUN conda env create -f environment.yml

# set the working directory in the container
WORKDIR /app

 # Copy the model
COPY ["predict.py", "model.bin", "dv.bin", "./"]

# Expose a port on the container
# Remember to map the port to a port in the host when running the container!
EXPOSE 9696

# Specify entrypoint
ENTRYPOINT ["conda", "run", "-n", "matchdeploy", \
            "gunicorn", "--bind=0.0.0.0:9696", "predict:app"]