# API Callbacks #

## Overview ##

The AIDocker set of machine learning models can be configured in different API configurations. Using the `container.yaml` file, under keys `input` and `output`, the user can specify what kind of input and output the model expects and returns. The models from this repository are designed to work in horizontally scalable environments with different queue types:
- a simple file-queue based on the filesystem, which uses 3 folders from which files are moved around during processing. `/tmp/ai/staged` is the folder where files are place when waiting to be processed, `/tmp/ai/sourced` is the folder where files are placed while they are under processing and `/tmp/ai/processed` is the folder where files are placed after they have been processed. The file-queue is configured with the `file` key.
- a zeromq queue, which uses a zeromq socket to send and receive messages. The zeromq queue is configured with the `zero` key.
Each callbacks submodule contains inputs and outputs callbacks for most common input and output types.

## File Queue ##

Usually a model is called via the APIs `/put/<something>` endpoints, where `<something>` can be a text, an image, a document, a csv file etc. The file/data is hashed using a configurable hash (defaults to SHA256), metadata is extracted from it and two files are created under `/tmp/ai/staged`:
- `<hash>.json` which contains the metadata
- `<hash>.<extension>` which contains the data itself, where `<extension>` is the extension of the original file normalized (eg. `jpeg` becomes `jpg`)
The API relinquishes control over the file at this point and the role is passed to the [queue daemon](../../daemon/aibatchdaemon.py) which selects files using a FIFO policy and packs them into small batches which are moved to `/tmp/ai/sourced` and passes control to the model. The model takes the batch, processes it and produces an output in `/tmp/ai/prepared`, while removing the `/tmp/ai/sourced` files. Also, the metadata file gets updated at every point with relevant information. The control is lastly passed to the [cleaner daemon](../../daemon/queuecleaner.py) which deletes the `/tmp/ai/prepared` files together with the metadata file and any other files that might not have been deleted by the model daemon.

## ZeroMQ Queue ##

The ZeroMQ is implemented with two components: a [ZMQ proxy](../../daemon/zmqdaemon.py) which ensures that the whole system is able to use a [router-to-dealer pattern](https://zguide.zeromq.org/docs/chapter3/) and a [worker daemon](../../daemon/aizerodaemon.py) which spins up workers to process the messages. ZeroMQ queue is suitable for real-time inference due to its low latency so at the moment of writing this there's no file support for it.
