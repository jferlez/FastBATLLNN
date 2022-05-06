#!/bin/bash

user=`id -n -u`
GID=`id -g`

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "$SCRIPT_DIR/DockerDeps"
docker build -t fastbatllnn:deps .
cd "$SCRIPT_DIR"

docker build --build-arg USER_NAME=$user --build-arg UID=$UID --build-arg GID=$GID -t fastbatllnn-run:${user} .