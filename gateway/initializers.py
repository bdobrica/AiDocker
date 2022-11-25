import os

import pika
from cassandra import InvalidRequest
from cassandra.cluster import Cluster, Session
from minio import Minio


def pika_initialize(queue) -> pika.channel.Channel:
    """Initialize a RabbitMQ connection."""
    params = pika.URLParameters(os.environ.get("RABBITMQ_URL"))
    conn = pika.BlockingConnection(params)
    channel = conn.channel()

    # Create queue if not exists
    channel.queue_declare(queue=queue)

    return channel, queue


def cassandra_initialize(keyspace: str) -> Session:
    """Initialize a Cassandra connection."""
    hosts = [
        host.strip()
        for host in os.environ.get("CASSANDRA_HOST", "localhost").split(",")
    ]
    cluster = Cluster(hosts)
    session = cluster.connect()

    # Create keyspace if not exists
    try:
        session.set_keyspace(keyspace)
    except InvalidRequest:
        session.execute(
            f"CREATE KEYSPACE {keyspace} WITH REPLICATION = "
            "{'class': 'SimpleStrategy', 'replication_factor': 1}"
        )
        session.set_keyspace(keyspace)

    return session


def minio_initialize() -> Minio:
    """Initialize a Minio connection."""
    host = os.environ.get("MINIO_HOST", "localhost")
    port = int(os.environ.get("MINIO_PORT", 9000))
    access_key = os.environ.get("MINIO_ACCESS_KEY", "minio")
    secret_key = os.environ.get("MINIO_SECRET_KEY", "minio123")
    bucket = os.environ.get("MINIO_BUCKET", "ai")
    client = Minio(
        "{HOST}:{PORT}".format(HOST=host, PORT=port),
        access_key=access_key,
        secret_key=secret_key,
    )

    return client
