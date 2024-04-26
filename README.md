docker build -f dockerfile -t your_image_name .
docker build --build-arg PYTORCH_IMAGE=your_custom_image:tag -t your_image_name .
The your_image_name can be peartree::train

Mount a Volume: When you run the Docker container, you can mount a host directory as a data volume inside the container. This way, your training data stored on the host machine can be accessed within the container. Replace /path/to/local/training/data with the path to the training data on your host machine, and /path/inside/container with the path where you want the data to be accessible inside the Docker container.

docker run -v /path/to/local/training/data:/path/inside/container your-image-name
dl-model-docker
