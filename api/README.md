# API Server #

The API server allows calls to the machine learning models moderated via a queue. As models can take different inputs and produce different outputs, the API server is configurable via the `container.yaml` file, under keys `input` and `output`. The models from this repository are designed to work in horizontally scalable environments with different queue types: file-based, zeromq, rabbitmq and kafka.

Here's an example of a `container.yaml` file that uses the file-based queue:

```yaml
...
input:
  - endpoint: /put/document
    queue: file
    required:
      document: document
  - endpoint: /delete/document
    queue: file
    required:
      token: token
  - endpoint: /put/text
    queue: zero
    required:
      text: text
output:
  - endpoint: /get/json
    queue: file
    type: json
...
```

For `input`, the endpoint provides the name of the callback which is built from the path's first to parts. Eg. `/<part[0]>/<part[1]>/...` translates into `<part[0]>_<part[1]>` callback. The `queue` key specifies from which module the callback should be imported. For now, the other keys are just descriptive with no actual effect.

For `output`, the endpoint and queue keys are used in a similar way. For now, the other keys are just descriptive with no actual effect.

If files are needed as input, then the recommended `enctype` is `application/x-www-form-urlencoded` and additional data is passed along as `multipart/form-data`.

If no files are needed as input, then both `multipart/form-data`, `application/x-www-form-urlencoded` and `application/json` are supported.
