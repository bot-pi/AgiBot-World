# Finetune UniVLA using Manipulation Data

The current official implementation (Manipulation-Challenge branch) is difficult to set up in local environment and thus difficult to reproduce.
This file is to to provide a reproducible way to finetune UniVLA using the manipulation data.

## 1. Setup Development Environment

### 1.1 Development in Local Computer
The development environment is self-contained in a [Docker image file](./.devcontainer/train.dockerfile) with all dependencies installed in a ubuntu OS.
To set up the environment, make sure the following tools are installed on your local machine:
- [Docker](https://www.docker.com/get-started)
- [Git](https://git-scm.com/downloads)
- [Visual Studio Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

### 1.2 Development in Server Mode

***Enable SSH remote connection***

If you want to develop in a server, and access the server via SSH from local machine, you can use the following steps:
- On the server, install 
  - [Docker](https://www.docker.com/get-started)    
  - [Git](https://git-scm.com/downloads)
  - If not installed, [SSH server](https://www.ssh.com/ssh/server/)
- On the local machine, install 
  - [Visual Studio Code](https://code.visualstudio.com/) with the [Remote - SSH extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh).

Make sure you can SSH into the server from the local machine. You can test this by running the following command in the terminal:
```bash
ssh username@server_ip_address
```

***Add Docker Context***

Follow this [guide](https://code.visualstudio.com/docs/containers/ssh) to add the server as a Docker context, so that you can run Docker commands on the server from your local machine.


## 2. Clone the Repository
Open a terminal in your local machine or server depends on your development environment, and navigate to the directory where you want to clone the repository.

Run the following command to clone the repository:
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

**Local Development**

3. Open the repository in Visual Studio Code, and open the command palette (Ctrl+Shift+P) and select "Dev Containers: Reopen in Container". This will build the Docker image and open the repository in the container.

This will take ~10 minutes to build the docker image. Once the image is built, you will see a green icon in the bottom left corner of the window indicating that you are in a dev container.

**Server Development**

3. general steps are: 1) to activate a dev container in server, 2) remote connect to the server via Visual Studio Code.
  - Open a terminal and connect to server via SSH:
    ```bash
    ssh username@server_ip_address
    ```
  - Navigate to the cloned repository if not
  - Activate dev container by running the following command:
  
    ```bash
    cd .devcontainer && make build # build the dev container if not built yet
    cd ../ && bash .devcontainer/run_devcontainer.sh
    ```
  - Open Visual Studio code on your local machine, and in the Docker extension, find the running devcontainer and attach a visual studio code to it. This will allow you enter into the dev container and edit the code in the container.

4. Open a terminal in the dev container and run the following command to finetune UniVLA:
```bash
bash scripts/finetune.sh
```

**Note: this finetuning will not be successful for now. Need to understand how the dataset is setup for the finetuning.**

