FROM continuumio/miniconda3

# Copy environment and requirements files
COPY environment.yml /tmp/environment.yml
COPY requirements.txt /tmp/requirements.txt

# Install bash.
RUN apt update && apt install -y bash

# Create Conda environment
RUN conda env create -f /tmp/environment.yml

# Activate environment and install pip packages
RUN /bin/bash -c "source activate minigrok && pip install -r /tmp/requirements.txt"

# Set environment path
ENV PATH /opt/conda/envs/minigrok/bin:$PATH

# Set working directory and copy app code
WORKDIR /app
COPY . /app

# Expose port and run FastAPI app
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
