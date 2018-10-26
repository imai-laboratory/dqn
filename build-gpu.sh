#!/bin/bash -eux

sed -e 's/{image}/tensorflow\/tensorflow:latest-gpu-py3/g' Dockerfile_template > Dockerfile
uid=$(id $(whoami) | awk '{print $1}' | sed -e 's/uid=//g' | sed -e 's/(.*$//g')
gid=$(id $(whoami) | awk '{print $2}' | sed -e 's/gid=//g' | sed -e 's/(.*$//g')
sudo docker build --build-arg uid=$uid --build-arg gid=$gid -t takuseno/dqn .
