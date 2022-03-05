#!/bin/bash

user=`id -n -u`
GID=`id -g`

docker build --build-arg USER_NAME=$user --build-arg UID=$UID --build-arg GID=$GID -t fastbatllnn-test:${user} .