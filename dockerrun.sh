#!/bin/bash

user=`id -n -u`
uid=`id -u`
gid=`id -g`
mkdir container_results
docker run --privileged -it -p 3000:22 -v "$(pwd)"/container_results:/home/${user}/results fastbatllnn-test:${user} ${user}