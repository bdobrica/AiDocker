#!/bin/bash
docker ps | grep yolov4 | awk '{print $1}' | xargs docker stop > /dev/null 2>&1
docker ps | grep modnet | awk '{print $1}' | xargs docker stop > /dev/null 2>&1
docker ps | grep u2net | awk '{print $1}' | xargs docker stop > /dev/null 2>&1
docker ps | grep nudenet | awk '{print $1}' | xargs docker stop > /dev/null 2>&1
docker ps | grep agenet | awk '{print $1}' | xargs docker stop > /dev/null 2>&1
docker ps | grep gfm34b2tt | awk '{print $1}' | xargs docker stop > /dev/null 2>&1
docker ps | grep isnet | awk '{print $1}' | xargs docker stop > /dev/null 2>&1
docker ps | grep srfbnet | awk '{print $1}' | xargs docker stop > /dev/null 2>&1