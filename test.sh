#!/bin/bash
SERVER="http://localhost:5002"

PWD=$(pwd)

echo "Testing ${SERVER} with files from ${PWD}/test. Output will be in ${PWD}/out"

func_upload(){
  echo "upload ...  $1"
  TOKEN=$(curl -s -F "image=@$1" $SERVER/put/image | jq '.token' | sed -e 's/"//g')

  URL="null"
  while [ "$URL" == "null" ]; do
    echo "try ..."
    RESPONSE=$(curl -s -X POST $SERVER/get/json -H 'Content-Type: application/json' -d "{\"token\": \"${TOKEN}\"}")
    STATUS=$(echo "${RESPONSE}" | jq '.status' | sed -e 's/"//g')
    ERROR=$(echo "${RESPONSE}" | jq '.error' | sed -e 's/"//g')
    URL=$(echo "${RESPONSE}" | jq '.url' | sed -e 's/"//g')
    echo "  url: ${URL}"
    echo "  status: ${STATUS}"
    echo "  error: ${ERROR}"
    if [ "$URL" != "null" ]; then
      FILENAME=$(basename -- $1)
      EXT="${URL##*.}"
      curl -s "${SERVER}${URL}" --output "out/${FILENAME%.*}.${EXT}"
      echo "done."
    fi
  done
}

rm -Rf out
mkdir -p out
find test/ -type f | while read image; do
  func_upload $image &
done