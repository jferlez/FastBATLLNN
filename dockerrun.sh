#!/bin/bash

user=`id -n -u`
uid=`id -u`
gid=`id -g`

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

SHMSIZE=$(( `grep MemTotal /proc/meminfo | sed -e 's/[^0-9]//g'` / 2097152 ))
if [ ! -d "$SCRIPT_DIR/container_results" ]
then
    mkdir "$SCRIPT_DIR/container_results"
fi
cd "$SCRIPT_DIR/container_results"
if [ -e ~/.ssh/id_rsa.pub ]
then
    echo "Copying public key from ~/.ssh/id_rsa.pub to container authorized_keys"
    cat ~/.ssh/id_rsa.pub > authorized_keys
    echo "" >> authorized_keys
fi
if [ -e ~/.ssh/authorized_keys ]
then
    echo "Copying public keys from ~/.ssh/authorized_keys to container authorized_keys"
    cat ~/.ssh/authorized_keys >> authorized_keys
fi
cd ..
docker run --privileged --gpus all --shm-size=${SHMSIZE}gb -it -p 3000:22 -v "$(pwd)"/container_results:/home/${user}/results fastbatllnn-test:${user} ${user}