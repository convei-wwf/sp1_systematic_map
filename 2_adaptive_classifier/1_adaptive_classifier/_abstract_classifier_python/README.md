# Abstract Classifier

This repository contains tools to classify abstracts using a pre-trained machine learning model. The workflow involves launching a Docker container with the appropriate environment and running a Python script within the container.

## Requirements

1) Docker must be installed and it must be configured to use the GPU(s) on the system.
2) Download and extract the abstract classifier model somewhere you can access it at or below the directory this file is in: https://drive.google.com/file/d/1ep1ZC5sC5LoKzlLPHr4JoCKpRnDiouIy/view?usp=drive_link. In the example below this model is extracted to a subdirectory called `models`

### Step 1: Launch the Docker Container

The Docker container has a preconfigured environment with all appropriate NLP models and supporting libraries installed. The specific configuration can be found in `Dockerfile` in this directory, but we have already pre-built the container at `therealspring/convei_abstract_classifier:latest`. We did this because the `flash-attention` library takes quite a long time to compile (>30m).

Example command to launch the Docker container

`bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v .:/workspace therealspring/convei_abstract_classifier:latest`

Argument descriptions

* `--gpus all`: Enables access to all available GPUs in the container. Ensure your machine has CUDA-enabled GPUs. This was developed and tested on a workstation with one GPU so I'm not sure what happens if you have more than one.
* `--ipc=host`: Necessary for good function of the `flash-attention` library, otherwise allows the docker container to access the shared memory space on the OS.
* `--ulimit memlock=-1`: Found during development that this flag was necessary to avoid memory errors, it's otherwise a flag to the Linux kernel to removes the memory lock limit but the default is supposed to be -1 already, so there may be a parent Docker container that messes with that somehow... This makes it specific.
* `--ulimit` stack=67108864: This sets the stack size limit to 64 MB and in development we found that some library, perhaps the flash-attention library or the NLP libraries could make deep stack calls, eventually doing stack overflow errors. This fixes that.
* `-it`: Allocates an interactive terminal for the container necessary for the user to interact with the shell in the Docker container environment.
* `--rm`: Automatically removes the container after it stops, ensuring a clean environment.
* `-v .:/workspace`: Mounts the current working directory on your host machine (.) to /workspace inside the container. `workspace` is the internal Docker container's working directory so this effectively maps the contents of the directory you launch this command from to the internal Docker container workspace. From there you can launch any of the python scripts in that directory, and have any access to data or models in that directory.
* `therealspring/convei_abstract_classifier:latest`: The precompiled Docker container specified in `Dockerfile`.

### Step 2: Run the Abstract Classifier Script

Within the container shell, run the abstract classifier with the following command:

`python abstract_classifer.py ./models/convei_abstract_classifier_2024_10_22_19_07_91p classifier_round3_set.csv`

Where,

* `./models/convei_abstract_classifier_2024_10_22_19_07_91p` is the pre-trained NLP model used in this call (change if you have a different model)
* `classifier_round3_set.csv` is a CSV table with at least the field `abstract`.

The script will run which, after ~30s of self-organizing` will classify approximately two abstracts per second (tested on an RTX 4080 GPU).

The result will be in the same working directory and be named the same name as the input table with "predicted" prefixed to it; in the case above `predicted_classifier_round3_set.csv`.
