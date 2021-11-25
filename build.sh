#!/bin/bash
docker build -f modnet/Dockerfile -t modnet .
docker build -f u2net/Dockerfile -t u2net .