#!/bin/bash

user=`id -n -u`

docker run --privileged -it -p 3000:22 -v "$(pwd)"/container_home:/home/${user} fastbatllnn-test:${user} ${user}