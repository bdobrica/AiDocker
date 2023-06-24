class Config:
    DEBUG = True
    CLIENT_SOCKET = "unix:/tmp/client.sock"
    WORKER_SOCKET = "unix:/tmp/fastapi.sock"
