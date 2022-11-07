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

## Image Segmentation ##

To extract the background from an image, do the following:

```sh
curl -F 'image=@/path/to/image.jpg' http://localhost:5000/put/image
```

This will place the `image.jpg` in the processing queue. The supported image types are `image/jpeg` and `image/png`. More can be easily added. The output of the command will produce the following (the token is an example token, expect different tokens for different images):

```json
{"token": "1a4d524ad2c21a7f50dc64ce4ee3a345e28972961c16513465d5161a8c0a3d1b"}
```

Here, the `token` can be used to retrieve the result. To retrieve the result, use the output of the first command and run the following:

```sh
curl -X POST http://localhost:5000/get/json -H 'Content-Type: application/json' -d '{"token": "1a4d524ad2c21a7f50dc64ce4ee3a345e28972961c16513465d5161a8c0a3d1b"}'
```

No need to pass the mimetype. This is stored internally by the container and the output will always follow the same mimetype as the input image.
This will produce a response depending on the container. For the segmentation containers, the output json will look like this:

```json
{"url": "/get/image/1a4d524ad2c21a7f50dc64ce4ee3a345e28972961c16513465d5161a8c0a3d1b.jpg", "status": "success"}
```

If the container is busy processing other files, the `output.png` file will contain the following string `{"wait":"true"}`. Otherwise, it will be a normal PNG image in which the background was removed.

In order to get the image, just do the `curl` command on the output URL (don't forget to append the absolute path):

```sh
curl http://localhost:5000/get/image/1a4d524ad2c21a7f50dc64ce4ee3a345e28972961c16513465d5161a8c0a3d1b.jpg --output out.jpg
```

Remember that the localhost and the 5000 port are the ones outside from the container. This is because the container is not aware of what runs outside.

## Custom Background ##

You can customize the background via the `background` parameter:

```sh
curl -F 'background=#ffffff' -F 'image=@/path/to/image.jpg' http://localhost:5000/put/image
```

Will put a white background image behind the detected object. Any 6-hex digit color code is supported. Default is black.

```sh
curl -F 'background=https://picsum.photos/200' -F 'image=@/path/to/image.jpg' http://localhost:5000/put/image
```

Will put a lorem ipsum image behind the detected object.

## Object Detection ##

To detect the objects inside an image, do the following:

```sh
curl -F 'image=@/path/to/image.jpg' http://localhost:5000/put/image
```

This will place the `image.jpg` in the processing queue. The supported image types are `image/jpeg` and `image/png`. More can be easily added. The output of the command will produce the following:

```json
{"token": "1a4d524ad2c21a7f50dc64ce4ee3a345e28972961c16513465d5161a8c0a3d1b", "mime": "image/jpeg"}
```

Here, the `token` can be used to retrieve the result by calling the `get/json` endpoint. To retrieve the result, use the output of the first command and run the following:

```sh
curl -X POST http://localhost:5000/get/json -H 'Content-Type: application/json' -d '{"token": "1a4d524ad2c21a7f50dc64ce4ee3a345e28972961c16513465d5161a8c0a3d1b", "mime": "image/jpeg"}' --output /path/to/output.json
```

The result is a JSON response that contains the following keys:
* results: this is an array of detected objects, sorted by the percentage of the image that the image covers; an object has the following keys:
    * class: this is the name of the detected object; can be: person bicycle car motorcycle airplane bus train truck boat trafficlight firehydrant stopsign parkingmeter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard sportsball kite baseballbat baseballglove skateboard surfboard tennisracket bottle wineglass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hotdog pizza donut cake chair couch pottedplant bed diningtable toilet tv laptop mouse remote keyboard cellphone microwave oven toaster sink refrigerator book clock vase scissors teddybear hairdrier toothbrush; the list is available in yolov4/coco.names file;
    * conf: this is the confidence level of the object's detection - a number between 0 and 1, 0 = no confidence (actually the threshold is set to 0.5) and 1.0 = 100% sure;
    * x: the center of the object's bounding box on the horizontal, starting from the left, in pixels;
    * y: the center of the object's bounding box on the vertical, starting from the top, in pixels;
    * w: the width of the object's bounding box in pixels;
    * h: the height of the object's bounding box in pixels;
    * area: the percentage of the image covered by the object - 0 = 0%, 1.0 = 100%
* status: can be one of the following:
    * "not queued": the image is not yet placed in the processing queue; in this case, in the JSON output you'll find "wait": "true"
    * "processing": the image is in the processing queue, but still being processed; in this case, in the JSON output you'll find "wait": "true"
    * "success": the image was successfully processed and the result key is present in the JSON output;
* error: present if an error has happened and it will contain an error message
    * "unknown image format": if the image is not PNG or JPG
    * "invalid model output": the model has not produced correctly formated JSON data
    * "unknown token": the token used to get the output most likely expired (or didn't existed in the first place) and the image requires resubmission;

For ease of testing, you can use the old `get/image` endpoint to get an image that has the object bounding boxes drawn on top of the original image, to check that the detection works.

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