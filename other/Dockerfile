# Use a base image with Conda installed (like miniconda3)
FROM continuumio/miniconda3

# Copy the environment.yml to the container
COPY environment.yml /tmp/environment.yml

# Create the Conda environment
RUN conda env create -f /tmp/environment.yml || tail -n 10 /tmp/conda_build.log

# Activate the environment in the shell
SHELL ["conda", "activate", "-n", "llm", "/bin/bash", "-c"]

# Set environment variables to ensure the environment is active by default
ENV PATH=/opt/conda/envs/llm/bin:$PATH

# Install any additional dependencies required in the app (optional)
# RUN conda install -n <env_name> <additional_packages>

# Copy your application code into the container (customize as needed)
COPY . /app
WORKDIR /app

# Command to run your application (modify as needed)
CMD ["python", "local_llm.py"]
