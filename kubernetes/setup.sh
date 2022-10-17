#!/bin/bash

kubectl create -f manifests/rabbitmq.yaml
kubectl create -f manifests/minio.yaml