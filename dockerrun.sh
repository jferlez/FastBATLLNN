#!/bin/bash

user=`id -n -u`
uid=`id -u`
gid=`id -g`
mkdir container_home
docker run --user=${uid}:${gid} --privileged -it -p 3000:22 -v "$(pwd)"/container_home:/home/${user} fastbatllnn-test:${user} ${user}