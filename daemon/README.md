# Daemons and Services #

The module contains services and daemons that can be used to implement the machine learning pipeline.

## Overview ##

The main class define in this modules is the [Daemon](./daemon.py) class. This is the basis for a service or daemon that can be run in the background. It extends the [ModelMixin](./modelmixin.py) which provides the basic methods for loading the model configuration, and provides two methods that are intended to be extend to provide functionality:
- `run()` - this method is called when the daemon is started; it doesn't return and should be an infinite loop
- `load()` - this method is called when the daemon is started, after the call to `daemonize()`; it should be used to load the model and any other resources that are needed
Otherwise, the `Daemon` class provides basic functionality for start, stop, debug and daemonize the daemon.
