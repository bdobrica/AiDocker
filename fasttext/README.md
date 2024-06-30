# FastText Language Identification Model #

## Reference ##

- [https://fasttext.cc/docs/en/language-identification.html](https://fasttext.cc/docs/en/language-identification.html)

## Use Case ##

Given a text snippet, detect and extract the language of the text.

## API EndPoints ##

- endpoint: `/put/text`
    - type: synchronous
    - accepted method: PUT
    - encoding: application/json
    - parameters:
        - `text`: UTF-8 encoded text
    - response: `{"worker_id": <worker_id>, "results": {"<language>": <probability>, ...}, "version": "<version>", "latency": <latency>}`
        - `language`: the 2 letter ISO language code of the detected language, in uppercase. The model returns the top 5 most probable languages in descending order of probability
        - `probability`: the probability with which the language was detected, a float between 0 and 1
        - `version`: the version of the model, a string
        - `latency`: the time it took to process the request, a float in seconds
    - CURL example: `curl -X PUT -H "Content-Type: application/json" -d '{"text":"Tiger! Tiger! burning bright in the forests of the night."}' <SERVER>/put/text`

## Example Usage ##

The following example shows how to use the FastText language identification model to detect the language of a text snippet.

```python
import requests

SERVER = "http://localhost:5000"

print("Detecting language of text snippet:")
text = input()

response = requests.put(
    f"{SERVER}/put/text",
    json={"text": text},
    headers={"Content-Type": "application/json"},
)
detected_languages = response.json().get("results", {})

language = list(detected_languages)[0]
probability = detected_languages[language]

print(f"Detected language: {language} with probability {probability}")
```
