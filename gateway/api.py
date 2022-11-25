import json
import os

from datamodels import AvailableModels, ImageRequest, RootResponse, TextRequest
from fastapi import Depends, FastAPI
from initializers import cassandra_initialize, pika_initialize

__version__ = "0.8.6"

app = FastAPI(docs_url=None, redoc_url="/docs")

# PIKA initialization
pk_queue = os.environ.get("RABBITMQ_API_QUEUE", "ai_api")
pk_channel = pika_initialize(pk_queue)

# CASSANDRA initialization
cs_keyspace = os.environ.get("CASSANDRA_KEYSPACE", "ai")
cs_session = cassandra_initialize(cs_keyspace)


@app.get("/", response_model=RootResponse)
def get_root():
    return {"version": __version__}


@app.put("/{model_id}/image")
async def put_image(
    model_id: AvailableModels,
    image: ImageRequest = Depends(ImageRequest.as_form),
):
    event = {
        "model_id": model_id.value,
    }
    pk_channel.basic_publish(
        exchange="",
        routing_key=pk_queue,
        body=json.dumps(event).encode("utf-8"),
    )


@app.put("/{model_id}/text")
async def put_text(
    model_id: AvailableModels, text: TextRequest = Depends(TextRequest.as_form)
):
    event = {
        "model_id": model_id.value,
    }
    pk_channel.basic_publish(
        exchange="",
        routing_key=pk_queue,
        body=json.dumps(event).encode("utf-8"),
    )
