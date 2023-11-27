# Stable Diffusion Model #

## References ##

- [https://github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)

## Use Case ##

Given a text prompt, generate an image.

## API EndPoints ##


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
