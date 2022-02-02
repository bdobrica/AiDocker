#!/bin/bash
docker ps | grep yolov4 | awk '{print $1}' | xargs docker stop
docker ps | grep modnet | awk '{print $1}' | xargs docker stop
docker ps | grep u2net | awk '{print $1}' | xargs docker stop