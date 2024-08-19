# docker build . -t pytorch && docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v .:/workspace nvcr.io/pytorch
FROM nvcr.io/nvidia/pytorch:24.04-py3
WORKDIR /workspace

COPY requirements.txt /workspace
RUN pip install -r requirements.txt

# Flash Attention repository for faster CUDA processing
RUN git clone https://github.com/Dao-AILab/flash-attention.git \
    && cd flash-attention \
    && python setup.py install

RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('microsoft/deberta-large'); AutoTokenizer.from_pretrained('xlnet/xlnet-base-cased')"
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /workspace/requirements.txt

CMD ["bash"]
