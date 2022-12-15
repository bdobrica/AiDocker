# YOLOv4 Model #

## References ##

- [https://github.com/kiyoshiiriemon/yolov4_darknet](https://github.com/kiyoshiiriemon/yolov4_darknet)

## Use Case ##

Given an image, detect the objects in the image and return the bounding boxes, the probability and the class name for each object.
## API EndPoints ##

- endpoint: `/put/image`
    - accepted method: POST
    - encoding: multipart/form-data
    - parameters:
        - `image`: image file
    - response: `{"token": <token>, "version": <version>}`
        - `token`: hash of the image file; can be configured via the `API_IMAGE_HASHER` environment variable; default is `SHA256`
        - `version`: version of the model
    - CURL example: `curl -s -F "image=@<path/to/image/file>" <SERVER>/put/image`
- endpoint: `/get/json`
    - accepted method: POST
    - encoding: application/json
    - parameters:
        - `token`: token returned by `/put/image`
    - responses:
        - `{"error": "unknown token", "version": <version>}`: if there were no images uploaded with the given token
        - `{"error": "missing file metadata", "version": <version>}`: if file metadata could not be extracted
        - `{"error": "corrupted image metadata", "version": <version>}`: if file metadata was extracted, but it got corrupted
        - `{"error": "token expired", "version": <version>}`: if the token has expired; the expiration time of a token can be set via the `API_CLEANER_FILE_LIFETIME` environment variable; default is 30 minutes
        - `{"error": "invalid image extension", "version": <version>}`: if the image extension is not supported; while the model supports whatever OpenCV is allowed to read (that's almost anything), the allowed mimetimes are listed in the `mimetypes.json` file; if you need a new mimetype, please add it yourself to the `mimetypes.json` file and open a pull request;
        - `{"error": "invalid model output", "version": <version>}`: the model run, but did not produce any useful output
        - `{"wait": "true", "status": "processing", "version": <version>}`: the model is still processing the image; you can retry the query after a while
        - `{"wait": "true", "status": "not queued", "version": <version>}`: the model hasn't started processing the image yet; you can retry the query after a while
        - `{"results": [{"class": <class>, "conf": <conf>, "x": <x>, "y": <y>, "w": <w>, "h": <h>, "area": <area>}, ...], "token": <token>, "status": "success", "version": <version>}`: the model has finished processing the image:
            - `token`: token of the image
            - `version`: version of the model
            - `results`: an array that contains an object for each face with the following properties:
                - `class`: class name from the list available here [coco.names](https://raw.githubusercontent.com/kiyoshiiriemon/yolov4_darknet/master/data/coco.names);
                - `conf`: confidence of the detection, between 0 and 1;
                - `x`: the center of the object on the horizontal axis;
                - `y`: the center of the object on the vertical axis;
                - `w`: the width of the object;
                - `h`: the height of the object;
                - `area`: the area of the object relative to the size of the image; it is a value between 0 and 1; results are sorted by this parameter descendingly.
    - CURL example: `curl -s -X POST -H "Content-Type: application/json" -d '{"token": <token>}' <SERVER>/get/json`

## Example Usage ##

The following script will send the image for which the path is passed as parameter to the model for processing and will wait for the result, checking every second if the model has finished processing the image:

```bash
#!/bin/bash
SERVER="http://localhost:5000" # change this to the address of your server
IMAGE_PATH=$1 # path to the image file

echo "uploading image"
TOKEN=$(curl -s -F "image=@${IMAGE_PATH}" $SERVER/put/image | jq '.token' | sed -e 's/"//g')

RESULTS="null"
ERROR="null"
while [ "$RESULTS" == "null" ] && [ "$ERROR" == "null" ] ; do
    echo "try ..."

    RESPONSE=$(curl -s -X POST $SERVER/get/json -H 'Content-Type: application/json' -d "{\"token\": \"${TOKEN}\"}")

    ERROR=$(echo "${RESPONSE}" | jq '.error' | sed -e 's/"//g')
    RESULTS=$(echo "${RESPONSE}" | jq '.results' | sed -e 's/"//g')

    if [ "$RESULTS" != "null" ]; then
        FILENAME=$(basename -- $1)
        echo "results: ${RESULTS}"
        echo "${RESPONSE}" > "out/${FILENAME%.*}.json"
        echo "done."
        break
    fi
    if [ "$ERROR" != "null" ]; then
        echo "file: $1 error: $ERROR"
        echo "${RESPONSE}" > "out/${FILENAME%.*}-error.json"
        echo "done."
        break
    fi

    sleep 1 # wait a second before trying again
done
```
