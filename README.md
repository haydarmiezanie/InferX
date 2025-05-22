# InferX - Machine Learning Inference System

InferX is a high-performance inference system designed for deploying machine learning models at scale. This repository contains the necessary components to deploy models in both local and production environments.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Cloning the Repository](#cloning-the-repository)
- [Local Testing](#local-testing)
- [Production Deployment](#production-deployment)
- [Configuration Container During Training](#configuration-container-during-training)
- [Configuration Container During Hosting](#configuration-container-during-hosting)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before using InferX, ensure you have the following installed:

- Python 3.8 or higher
- Docker (for containerized deployment)
- AWS CLI (for AWS deployment)
- The following Python packages:
  ```bash
  pip install numpy pandas scikit-learn flask gunicorn boto3
  ```
## Clone Repository

To get started with InferX, clone the repository:
```bash
git clone https://github.com/haydarmiezanie/InferX.git
cd InferX
```

## Local Testing

1. Build the docker image
   ```bash
   docker build -t inferx-image .
   ```

2. Training
    - Local Training
        ```bash
        docker run -v $(pwd)/test_dir:/opt/ml --rm ${image} train
        ```
    - Local Serving
        ```bash
        docker run -v $(pwd)/test_dir:/opt/ml -p 8080:8080 --rm ${image} serve
        ```
    
3. Predict
    ```bash
    curl --data-binary @${prediction_data} -H "Content-Type: text/csv" -v http://localhost:8080/invocations
    ```

## Production Deployment

1. Geth auth token
    ```bash
    aws ecr get-login-password --region ap-southeast-3 | docker login --username AWS --password-stdin 650663850722.dkr.ecr.ap-southeast-3.amazonaws.com
    ```

2. Change tag
    ```bash
    docker tag inferx-image:latest {ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{YOUR_REPOSITORY}
    ```

3. Push Image
    ```bash
    docker push {ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{YOUR_REPOSITORY}
    ```

## Configuration container during training

When Amazon SageMaker runs training, your `train` script is run just like a regular Python program. A number of files are laid out for your use, under the `/opt/ml` directory:

    /opt/ml
    |-- input
    |   |-- config
    |   |   |-- hyperparameters.json
    |   |   `-- resourceConfig.json
    |   `-- data
    |       `-- <channel_name>
    |           `-- <input data>
    |-- model
    |   `-- <model files>
    `-- output
        `-- failure

#### The input

* `/opt/ml/input/config` contains information to control how your program runs. `hyperparameters.json` is a JSON-formatted dictionary of hyperparameter names to values. These values will always be strings, so you may need to convert them. `resourceConfig.json` is a JSON-formatted file that describes the network layout used for distributed training. Since scikit-learn doesn't support distributed training, we'll ignore it here.
* `/opt/ml/input/data/<channel_name>/` (for File mode) contains the input data for that channel. The channels are created based on the call to CreateTrainingJob but it's generally important that channels match what the algorithm expects. The files for each channel will be copied from S3 to this directory, preserving the tree structure indicated by the S3 key structure. 
* `/opt/ml/input/data/<channel_name>_<epoch_number>` (for Pipe mode) is the pipe for a given epoch. Epochs start at zero and go up by one each time you read them. There is no limit to the number of epochs that you can run, but you must close each pipe before reading the next epoch.

#### The output

* `/opt/ml/model/` is the directory where you write the model that your algorithm generates. Your model can be in any format that you want. It can be a single file or a whole directory tree. SageMaker will package any files in this directory into a compressed tar archive file. This file will be available at the S3 location returned in the `DescribeTrainingJob` result.
* `/opt/ml/output` is a directory where the algorithm can write a file `failure` that describes why the job failed. The contents of this file will be returned in the `FailureReason` field of the `DescribeTrainingJob` result. For jobs that succeed, there is no reason to write this file as it will be ignored.

## Configuration container during hosting

Hosting has a very different model than training because hosting is responding to inference requests that come in via HTTP. In this example, we use our recommended Python serving stack to provide robust and scalable serving of inference requests:

![Request serving stack](stack.png)

This stack is implemented in the sample code here and you can mostly just leave it alone. 

Amazon SageMaker uses two URLs in the container:

* `/ping` will receive `GET` requests from the infrastructure. Your program returns 200 if the container is up and accepting requests.
* `/invocations` is the endpoint that receives client inference `POST` requests. The format of the request and the response is up to the algorithm. If the client supplied `ContentType` and `Accept` headers, these will be passed in as well. 

The container will have the model files in the same place they were written during training:

    /opt/ml
    `-- model
        `-- <model files>

### The parts of the sample container

In the `container` directory are all the components you need to package the sample algorithm for Amazon SageMager:

    .
    |-- Dockerfile
    |-- build_and_push.sh
    `-- decision_trees
        |-- nginx.conf
        |-- predictor.py
        |-- serve
        |-- train
        `-- wsgi.py

Let's discuss each of these in turn:

* __`Dockerfile`__ describes how to build your Docker container image. More details below.
* __`build_and_push.sh`__ is a script that uses the Dockerfile to build your container images and then pushes it to ECR. We'll invoke the commands directly later in this notebook, but you can just copy and run the script for your own algorithms.
* __`decision_trees`__ is the directory which contains the files that will be installed in the container.
* __`local_test`__ is a directory that shows how to test your new container on any computer that can run Docker, including an Amazon SageMaker notebook instance. Using this method, you can quickly iterate using small datasets to eliminate any structural bugs before you use the container with Amazon SageMaker. We'll walk through local testing later in this notebook.

In this simple application, we only install five files in the container. You may only need that many or, if you have many supporting routines, you may wish to install more. These five show the standard structure of our Python containers, although you are free to choose a different toolset and therefore could have a different layout. If you're writing in a different programming language, you'll certainly have a different layout depending on the frameworks and tools you choose.

The files that we'll put in the container are:

* __`nginx.conf`__ is the configuration file for the nginx front-end. Generally, you should be able to take this file as-is.
* __`predictor.py`__ is the program that actually implements the Flask web server and the decision tree predictions for this app. You'll want to customize the actual prediction parts to your application. Since this algorithm is simple, we do all the processing here in this file, but you may choose to have separate files for implementing your custom logic.
* __`serve`__ is the program started when the container is started for hosting. It simply launches the gunicorn server which runs multiple instances of the Flask app defined in `predictor.py`. You should be able to take this file as-is.
* __`train`__ is the program that is invoked when the container is run for training. You will modify this program to implement your training algorithm.
* __`wsgi.py`__ is a small wrapper used to invoke the Flask app. You should be able to take this file as-is.

In summary, the two files you will probably want to change for your application are `train` and `predictor.py`.

# Reference

[deployment reference](https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.ipynb)
