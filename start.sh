#!/bin/bash

# This script is run when the Docker container starts

# Display a message to indicate that the pod has started
echo "pod started"

# If a public SSH key is provided, set up the SSH authorized_keys
if [[ $PUBLIC_KEY ]]
then
    # Create the .ssh directory in the user's home directory if it doesn't exist
    mkdir -p ~/.ssh
    # Set the directory permissions to be read, write, and execute only by the user
    chmod 700 ~/.ssh
    # Navigate to the .ssh directory
    cd ~/.ssh
    # Append the provided public key to the authorized_keys file
    echo $PUBLIC_KEY >> authorized_keys
    # Set the permissions of the authorized_keys file to be read and write only by the user
    chmod 600 authorized_keys
    # Return to the previous directory
    cd /
    # Start the SSH service
    service ssh start
fi

# Navigate to the /workspace directory where the repositories will be cloned
cd /workspace
# Clone the deepspeed-testing repository from GitHub
git clone https://github.com/trevorWieland/deepspeed-testing.git
# Clone the minGPT repository from GitHub
git clone https://github.com/karpathy/minGPT.git
# Navigate into the minGPT directory
cd minGPT
# Install the minGPT package using pip
pip install -e .
# Return to the parent directory
python -c mingpt/train.py
# Keep the container running indefinitely
sleep infinity
