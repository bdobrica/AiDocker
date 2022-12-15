# GFM Model #

## Reference ##

- [https://github.com/JizhiziLi/GFM](https://github.com/JizhiziLi/GFM)

## Use Case ##

Given an image, extract the background of the image and keep only what is considered foregrownd.

## API EndPoints ##

- endpoint: `/put/image`
    - accepted method: POST
    - encoding: multipart/form-data
    - parameters:
        - `image`: image file
        - `background`: (optional) after the original background is removed, replace it with this; can be:
            - a color hexadecimal code (e.g. `#ff0000` for red)
            - an URL to an image; the URL must be accessible from within the container
    - response: `{"token": <token>, "version": <version>}`
        - `token`: hash of the image file; can be configured via the `API_IMAGE_HASHER` environment variable; default is `SHA256`
        - `version`: version of the model
    - CURL example: `curl -s -F 'background=#ff0000' -F "image=@<path/to/image/file>" <SERVER>/put/image`
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
        - `{"url": "/get/image/<token>.<extension>, "status": "success", "version": <version>}`: the model has finished processing the image:
            - `token`: token of the image
            - `version`: version of the model
            - `url`: URL to the processed image relative to `<SERVER>`; the extension is the same as the original image; to get transparent PNGs you need to pass a PNG image as input
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

URL="null"
ERROR="null"
while [ "$URL" == "null" ] && [ "$ERROR" == "null" ] ; do
    echo "try ..."

    RESPONSE=$(curl -s -X POST $SERVER/get/json -H 'Content-Type: application/json' -d "{\"token\": \"${TOKEN}\"}")
    
    URL=$(echo "${RESPONSE}" | jq '.url' | sed -e 's/"//g')
    ERROR=$(echo "${RESPONSE}" | jq '.error' | sed -e 's/"//g')
    
    if [ "$URL" != "null" ]; then
        FILENAME=$(basename -- $1)
        EXT="${FILENAME##*.}"
        curl -s "${SERVER}${URL}" --output "out/${FILENAME%.*}.${EXT}"
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
