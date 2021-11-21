# Instalation

To install build the modnet docker image run the following command inside the modnet folder (the one containing the Dockerfile file).

```sh
docker build -t modnet .
```

The above command uses the following switches:
- `-t container_name` : specifies a container name;

# Running the image

To run the Docker image as a daemon, run the following command.

```sh
docker run -d -p 127.0.0.1:5000:5000/tcp modnet
```

The above command uses the following switches:
- `-d` : run the Docker container in daemon mode;
- `-p host:host_port:container_port/proto` : binds `container_port` to the `host_port` on the host address using the protocol `proto`;

To check if the Docker image is running, run the following command.

```sh
docker ps
```

And the output should look like this:
```
CONTAINER ID   IMAGE           COMMAND                  CREATED         STATUS         PORTS                                       NAMES
132763ced3ab   modnet          "/opt/app/app.py"        2 seconds ago   Up 2 seconds   0.0.0.0:5000->5000/tcp, :::5000->5000/tcp   musing_proskuriakova
```

To stop a container, run the following command:

```sh
docker stop {container_id}
```

Where `{container_id}` is the Container ID that appears in `docker ps` output. To stop the example container, run `docker stop 132763ced3ab`.

# Using the container

To extract the background from an image, do the following:

```sh
curl -F 'image=@/path/to/image.jpg' http://localhost:5000/put/image
```

This will place the `image.jpg` in the processing queue. The supported image types are `image/jpeg` and `image/png`. More can be easily added. The output of the command will produce the following:

```
{"token": "1a4d524ad2c21a7f50dc64ce4ee3a345e28972961c16513465d5161a8c0a3d1b", "mime": "image/jpeg"}
```

Here, the `token` can be used to retrieve the result. To retrieve the result, use the output of the first command and run the following:

```sh
curl -X POST http://localhost:5000/get/image -H 'Content-Type: application/json' -d '{"token": "1a4d524ad2c21a7f50dc64ce4ee3a345e28972961c16513465d5161a8c0a3d1b", "mime": "image/jpeg"}' --output /path/to/output.png
```

If the container is busy processing other files, the `output.png` file will contain the following string `{"wait":"true"}`. Otherwise, it will be a normal PNG image in which the background was removed.

# Error Handling

- if the token has expired, you will receive a 400 HTTP status code and the response body will be '{"error":"unknown token"}';
- if the image type is not supported (png or jpg) a 400 HTTP status code is returned and the response body will be '{"error":"unknown image format"}';

# Debug

To start the container in the debug mode, run the following command:

```sh
docker run -p 127.0.0.1:5000:5000/tcp -it --entrypoint=/bin/bash modnet
```

This will get you a bash to the inside of the container, but will not run the Python app. The switches used are:
- `-it` : this is actually `-i -t`, `-i` starting the container in interactive mode and `-t` granting it a tty for starting a console;
- `--entrypoint=/bin/bash` : when a container starts, it executes by default a command (in this case `/opt/app/app.py`); this will override the command and will actually run the `/bin/bash`;

To run the Python app inside the container, run:

```sh
/opt/app/app.py
```