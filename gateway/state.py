import json
import os
from functools import partial

import pika
from cassandra.cluster import Session
from initializers import cassandra_initialize, pika_initialize

__version__ = "0.8.6"


def callback(
    ch: pika.channel.Channel,
    method: pika.spec.Basic.Deliver,
    properties: pika.spec.BasicProperties,
    body: bytes,
    session: Session,
):
    """
    RabbitMQ consumer callback. Updates the state of the AI problem in Cassandra.
    :param ch: The RabbitMQ channel.
    :param method: The RabbitMQ method.
    :param properties: The RabbitMQ properties.
    :param body: The RabbitMQ body.
    :param session: The Cassandra session.
    """
    try:
        body = json.loads(body.decode("utf-8"))
        stm = session.prepare("INSERT INTO states (key, value) VALUES (?, ?)")
        session.execute(stm, (body, body))
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        ch.basic_nack(delivery_tag=method.delivery_tag)


if __name__ == "__main__":
    # PIKA initialization
    pk_queue = os.environ.get("RABBITMQ_STATE_QUEUE", "ai_state")
    pk_channel = pika_initialize(pk_queue)

    # CASSANDRA initialization
    cs_keyspace = os.environ.get("CASSANDRA_KEYSPACE", "ai")
    cs_session = cassandra_initialize(cs_keyspace)

    # Start consuming messages
    pk_channel.basic_consume(
        queue=pk_queue,
        on_message_callback=partial(callback, session=cs_session),
        auto_ack=False,
    )
    pk_channel.start_consuming()
