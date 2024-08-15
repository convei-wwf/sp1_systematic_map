# docker build . -t pytorch && docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v .:/workspace nvcr.io/pytorch

# Use the NVIDIA PyTorch container as the base image
FROM nvcr.io/nvidia/pytorch:24.04-py3

# Set the working directory inside the container
WORKDIR /workspace

# Install necessary dependencies
RUN pip install --upgrade pip \
    && pip install packaging ninja

# Clone the Flash Attention repository and install it
RUN git clone https://github.com/Dao-AILab/flash-attention.git \
    && cd flash-attention \
    && python setup.py install

# # Ensure all installations are complete and cleanup unnecessary files
# RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install transformers[torch] datasets evaluate scikit-learn chardet

RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('microsoft/deberta-base'); AutoTokenizer.from_pretrained('microsoft/deberta-base')"
RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('microsoft/deberta-large'); AutoTokenizer.from_pretrained('microsoft/deberta-large')"


# Set the default command to run a shell when the container starts
CMD ["bash"]
