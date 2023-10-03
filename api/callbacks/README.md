# API Callbacks

## Overview

The AIDocker set of machine learning models can be configured in different API configurations. Using the `container.yaml` file, under keys `input` and `output`, the user can specify what kind of input and output the model expects and returns. The models from this repository are designed to work in horizontally scalable environments with different queue types:
- a simple, file-queue based API that tries to take advantage of running multiple model instances in parallel from within a single container, using multiple forked processes, described in the `callbacks.file_queue` module;
- a zeromq based API that uses a zeromq queue to distribute the load between multiple model instances running either inside a single container or across multiple container, described in the `callbacks.zero_queue` module;
- a rabbitmq based API that uses a rabbitmq queue to distribute the load between multiple model instances running each inside its own container, described in the `callbacks.rabbit_queue` module;
- a kafka based API that uses a kafka queue to distribute the load between multiple model instances running each inside its own container, suitable , described in the `callbacks.kafka_queue` module;

Each callbacks submodule contains inputs and outputs callbacks for most common input and output types.
