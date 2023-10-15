# Daemons and Services #

The module contains services and daemons that can be used to implement the machine learning pipeline.

## Overview ##

The main class define in this modules is the [Daemon](./daemon.py) class. This is the basis for a service or daemon that can be run in the background. It extends the [ModelMixin](./modelmixin.py) which provides the basic methods for loading the model configuration, and provides several methods that are intended to be extend to provide functionality:
- `run()` - this method is called when the daemon is started; it doesn't return and should be an infinite loop
- `load()` - this method is called when the daemon is started, after the call to `daemonize()`; it should be used to load the model and any other resources that are needed
- `get_input_batch(batch_size: int) -> Iterable[Any]` - this method is called to get a batch of input data; it should return an iterable of input data
Otherwise, the `Daemon` class provides basic functionality for start, stop, debug and daemonize the daemon.

Next in the daemon object hierarchy are the `Ai*Daemon` classes. They are initialized with the same parameters as `Daemon` while adding an `input_type` parameter that specifies the type of the model input, an `Ai*Input` descendent. The `input_type` is used to instantiate an object with the current batch of input data, and always has `prepare()` and `serve()` methods for converting the input batch into something a machine learning can handle (usually a numpy array or a tensor) and respectively, getting the model output (again, numpy array or tensor typed) and converting it into the expected format. This makes the `Ai*Input` class the perfect place to store feature engineering and post-processing code. On the other hand, the `Ai*Daemon` has an `ai(input)` method that takes as input the output of `Ai*Input.prepare()` and produces and output that can be directly passed to `Ai*Input.serve()`. The `ai(input)` must be implemented specifically for each model type and can make use of `load()` method inherited from `Daemon` to load the model and any other resources needed. The `Ai*Daemon` has a `queue()` method that's dependent on the type of queue chosen, that selects a batch of input data, creates the `Ai*Input` object and feeds it to the `ai(input)` method and then passes it to the `serve()` method. The `queue()` method is run in an infinite loop by the `run()` method overriden from `Daemon`. Here's how the whole process works in python inspired pseudo-code:

```python
class AiInput:
    def __init__(self, input_batch):
        # here we can acknowledge the input batch and store it in the object, removing it from the queue
        queue.acknowledge(input_batch)
        self.input_batch = input_batch

    def prepare(self) -> np.ndarray:
        # here we can do feature engineering and other pre-processing
        return np.concat([file.read() for file in input_batch])
    
    def serve(self, model_output: np.ndarray):
        # here we can do post-processing and serve the model output
        for file, output in zip(input_batch, model_output):
            file.write(output)

# when initialized, the AiInput type is passed to the AiDaemon
AiDaemon.input_type = AiInput

def AiDaemon::load():
    # here we can load the model and any other resources needed
    self.model = load_model()

def AiDaemon::ai(model_input: np.ndarray) -> np.ndarray:
    # here we do the actual machine learning
    return self.model.predict(model_input)

# the Daemon.run infinite loop
while True:
    # the Ai*Daemon.queue() method
    while input_batch := AiDaemon.input_type.get_input_batch():
        model_input = AiDaemon.input_type(input_batch) # an Ai*Input object
        model_output = AiDaemon.ai(model_input.prepare())
        Ai*Output.serve(model_output)
    # if there are no input batches, the Daemon.run() method sleeps for a while before trying again
    time.sleep(os.getenv("QUEUE_LATENCY"))
```

The `daemon` module contains several templates for `Ai*Daemon`s and their matchin `Ai*Input` classes.

`AiForkDaemon` is designed for multi-core machines that will run single-threaded models. It uses the `fork()` system call to create a new process for each input batch, and then waits for the process to finish before moving on to the next batch. This daemon works well with tabular models that can be restricted to single-threaded operation or with highly optimized image models likes the ones from openCV. It is not suitable for pytorch or tensorflow models that are designed to take advantage of multiple cores, mostly due to how memory is shared between processes. You can however use it in these situations as well, making sure that the model weights are copied to each process and that the model is loaded in each process. 
