# Finetune UniVLA using Manipulation Data

The current official implementation (Manipulation-Challenge branch) is difficult to set up in local environment and thus difficult to reproduce.
This file is to to provide a reproducible way to finetune UniVLA using the manipulation data.


## 1. Setup Development Environment

The development environment is self-contained in a [Docker image file](./.devcontainer/train.dockerfile) with all dependencies installed in a ubuntu OS.
To set up the environment, make sure the following tools are installed on your local machine:
- [Docker](https://www.docker.com/get-started)
- [Git](https://git-scm.com/downloads)
- [Visual Studio Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).


## 2. Clone the Repository
Open a terminal and run the following command to clone the repository:
```bash
git clone --recurse-submodules -b basecode https://github.com/robot-squat/AgiBot-World.git
```

Here we use the `basecode` branch as our base code. The `Manipulation-Challenge` branch is the official implementation, and will be regularly merged into the `basecode` branch.

## 3. Finetune UniVLA

1. Download a sample dataset processed by robot-squad team from [here](https://huggingface.co/datasets/yangyangfu/agibot) and place it in the `dataset` folder.

2. Download checkpoints by running the following command in the terminal:
```bash
cd ckpts && \
bash download_baseline.sh
```

3. Open the repository in Visual Studio Code, and open the command palette (Ctrl+Shift+P) and select "Dev Containers: Reopen in Container". This will build the Docker image and open the repository in the container.

This will take ~10 minutes to build the docker image. Once the image is built, you will see a green icon in the bottom left corner of the window indicating that you are in a dev container.

4. Open a terminal in the dev container and run the following command to finetune UniVLA:
```bash
bash scripts/finetue.sh
```

**Note: this finetuning will not be successful for now. Need to understand how the dataset is setup for the finetuning.**