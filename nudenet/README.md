# NudeNet Model #

## References ##

- [https://github.com/notAI-tech/NudeNet](https://github.com/notAI-tech/NudeNet)

## Use Case ##

Given an image, detect and extract body parts that are considered indecent. It can either generate a JSON with the coordinates of the detected body parts or an image with the detected body parts masked out.

## API EndPoints ##

- endpoint: `/put/image`
    - accepted method: POST
    - encoding: multipart/form-data
    - parameters:
        - `image`: image file
        - `fast`: (optional) can be either `yes` or `no`; defaults to `no`; it affects the speed and the quality of the detection
        - `censor`: (optional) can be either `yes` or `no`; defaults to `no`; if set to `yes`, it will produce an image as a result with the body parts censored; the censoring algorithm can be set via the `API_NUDENET_CENSOR_TYPE` environment variable and it defaults to `blackbox`; it can be either `blackbox` or `blur`
    - response: `{"token": <token>, "version": <version>}`
        - `token`: hash of the image file; can be configured via the `API_FILE_HASHER` environment variable; default is `SHA256`
        - `version`: version of the model
    - CURL example: `curl -s -F "censor=yes" -F "image=@<path/to/image/file>" <SERVER>/put/image`
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
        - `{"results": [{"score": <score>, "label": <label>, "box": [<x_left>, <y_top>, <x_right>, <y_bottom>]}, ...], "token": <token>, "status": "success", "version": <version>}`: the model has finished processing the image and the parameter `censor` was set to `no`:
            - `token`: token of the image
            - `version`: version of the model
            - `results`: an array that contains an object for each detected body part with the following properties:
                - `score`: confidence score of the detection;
                - `label`: label of the detected body part; class names are available here [NudeNet](https://github.com/notAI-tech/NudeNet)
                - `box`: the coordinates of the top-left corner and the bottom-right corner of the body part
        - `{"url": "/get/image/<token>.<extension>, "status": "success", "version": <version>}`: the model has finished processing the image and the parameter `censor` was set to `yes`:
            - `token`: token of the image
            - `version`: version of the model
            - `url`: URL to the processed image relative to `<SERVER>`; the image will have any body parts considered sensitive censored; you can restrict which body parts are included using the `API_NUDENET_KEEP_LABELS` (labels to keep censoring) and `API_NUDENET_DROP_LABELS` (labels that you don't want to censor) environment variables to alter the list of detected classes; the class names must be separated by commas
    - CURL example: `curl -s -X POST -H "Content-Type: application/json" -d '{"token": <token>}' <SERVER>/get/json`
- endpoint: `/get/image/<token>.<extension>`
    - accepted method: GET
    - parameters:
        - `token`: token returned by `/put/image`
        - `extension`: extension of the original image; must be one of the supported mimetypes; see `/get/json` for more information
    - **NOTE**: this endpoint is intended to be called after a successful `/get/json` call; it is not meant to be called directly
    - CURL example: `curl -s -o <path/to/output/file> <SERVER>/get/image/<token>.<extension>`
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
