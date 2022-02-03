#!/bin/bash
SERVER="http://localhost:5000"

func_upload(){
  echo "upload ...  $1"
  TOKEN=$(curl -s -F "image=@$1" $SERVER/put/image | jq '.token' | sed -e 's/"//g')

  URL="null"
  RESULTS="null"
  ERROR="null"
  while [ "$URL" == "null" ] && [ "$RESULTS" == "null" ] && [ "$ERROR" == "null" ] ; do
    echo "try ..."
    RESPONSE=$(curl -s -X POST $SERVER/get/json -H 'Content-Type: application/json' -d "{\"token\": \"${TOKEN}\"}")
    STATUS=$(echo "${RESPONSE}" | jq '.status' | sed -e 's/"//g')
    ERROR=$(echo "${RESPONSE}" | jq '.error' | sed -e 's/"//g')
    URL=$(echo "${RESPONSE}" | jq '.url' | sed -e 's/"//g')
    RESULTS=$(echo "${RESPONSE}" | jq '.results' | sed -e 's/"//g')
    echo -e "\turl: ${URL}\tstatus: ${STATUS}\terror: ${ERROR}"
    if [ "$URL" != "null" ]; then
      FILENAME=$(basename -- $1)
      EXT="${FILENAME##*.}"
      curl -s "${SERVER}${URL}" --output "out/${FILENAME%.*}.${EXT}"
      echo "done."
    fi
    if [ "$RESULTS" != "null" ]; then
      FILENAME=$(basename -- $1)
      echo "results: ${RESULTS}"
      echo "${RESPONSE}" > "out/${FILENAME%.*}.json"
      echo "done."
    fi
    if [ "$ERROR" != "null" ]; then
      echo "file: $1 error: $ERROR"
      echo "${RESPONSE}" > "out/${FILENAME%.*}-error.json"
      echo "done."
    fi
  done
}

rm -Rf out
mkdir -p out
find test/ -type f | while read image; do
  func_upload $image &
done