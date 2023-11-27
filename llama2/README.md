# Facebook LLAMA2 7B Large Language Model #

## Reference ##

- [https://ai.meta.com/llama/](https://ai.meta.com/llama/)
- The code from this project is based on the LLama2 project available at [github.com/facebookresearch/llama](https://github.com/facebookresearch/llama). Changes were made to the original code to adapt it to the needs of this project.

## Use Case ##

Given a conversation, get the next prompt, similar to OpenAI's ChatGPT. The interface is mostly the same as the one from ChatGPT, but the model is different.
**Note:** The model requires a GPU with at least 16GB of memory to run properly. It can be run on a CPU, but it will be very slow.

## API EndPoints ##

- endpoint: `/put/text`
    - type: synchronous
    - accepted method: PUT
    - encoding: application/json
    - parameters:
        - `messages`: a list of messages, passed as a dictionary:
            - `role`: the role of the message, either `user`, `assistant` or `system`. The role `system` is used for messages that are not part of the conversation, such as the initial prompt.
            - `content`: the content of the message, a string
        - `max_gen_len`: the maximum length of the generated text, an integer. if not specified, the default value is the maximum length of the model
        - `temperature`: the temperature of the generated text, a float between 0 and 1. if not specified, the default value is 0.6
        - `top_p`: the top p of the generated text, a float between 0 and 1. if not specified, the default value is 0.9
    - response: `{"worker_id": <worker_id>, "choices": {"role": <role>, "content": <content>}, "version": "<version>", "latency": <latency>}`
        - `worker_id`: the id of the worker that processed the request, an integer
        - `choices`: the generated text, a dictionary
            - `role`: the role of the message, either `user`, `assistant` or `system`
            - `content`: the content of the message, a string
        - `version`: the version of the model, a string
        - `latency`: the time it took to process the request, a float in seconds
    - CURL example: `curl -X PUT -H "Content-Type: application/json" -d '[{"role": "system", "content": "Always answer with emojis"}, {"role": "user", "content": "How to go from Beijing to NY?"}]' <SERVER>/put/text`

## Example Usage ##

The following example shows how to use the LLama2 by having a conversation with it. The example uses the python library `requests` to make the requests to the server. It will ask the user for a prompt, send it to the server, and print the response. The conversation will end when the user types `exit`.

```python
import requests

SERVER = "http://localhost:5000"

messages = [{"role": "system", "content": "Always answer in haiku"}]

while True:
    print("Prompt:")
    content = input()

    if content.strip().lower() == "exit":
        break

    messages.append({"role": "user", "content": content})

    response = requests.put(
        f"{SERVER}/put/text",
        json=messages,
        headers={"Content-Type": "application/json"},
    )
    choices = response.json().get("choices", {})

    print("Response:", choices.get("content", ""))
    messages.append(choices)
```
