#!/bin/bash

user=`id -n -u`
uid=`id -u`
gid=`id -g`
if [ ! -d container_results ]
then
    mkdir container_results
fi
cd container_results
if [ -e ~/.ssh/id_rsa.pub ]
then
    cat ~/.ssh/id_rsa.pub > authorized_keys
    echo "\n" >> authorized_keys
fi
if [ -e ~/.ssh/authorized_keys ]
then
    cat ~/.ssh/authorized_keys >> authorized_keys
fi
cd ..
docker run --privileged -it -p 3000:22 -v "$(pwd)"/container_results:/home/${user}/results fastbatllnn-test:${user} ${user}