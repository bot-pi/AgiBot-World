# download baseline checkpoints from huggingface
#!/bin/bash

# Check if the Hugging Face CLI is installed
if [ ! -z "$(command -v huggingface-cli)" ]; then
    echo "Hugging Face CLI is installed."
else
    echo "Hugging Face CLI is not installed. Please install it first."
    exit 1
fi

# Download the checkpoints to current directory
# univla latent action model with two stages
huggingface-cli download qwbu/univla-latent-action-model \
                --repo-type model \
                --local-dir ./univla-latent-action-model \
                --max-workers 16

# prism language model
huggingface-cli download TRI-ML/prismatic-vlms \
                prism-dinosiglip-224px+7b/checkpoints/latest-checkpoint.pt \
                --repo-type model \
                --local-dir . \
                --max-workers 16

# univla model 
huggingface-cli download qwbu/univla-7b \
                --repo-type model \
                --local-dir ./univla-7b \
                --max-workers 16 