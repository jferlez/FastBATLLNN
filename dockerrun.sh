#!/bin/bash

user=`id -n -u`
uid=`id -u`
gid=`id -g`

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

SYSTEM_TYPE=$(uname)

for arg in "$@"; do
    case "$arg" in
        --gpu) GPUS="--gpus all";;
    esac
done

if [ "$SYSTEM_TYPE" = "Darwin" ]; then
    SHMSIZE=$(( `sysctl hw.memsize | sed -e 's/[^0-9]//g'` / 2097152 ))
    # Never enable GPUs on MacOS
    GPUS=""
else
    SHMSIZE=$(( `grep MemTotal /proc/meminfo | sed -e 's/[^0-9]//g'` / 2097152 ))
fi

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

CONTAINERS=`docker container ls -a | grep fastbatllnn-run:$user | sed -e "s/[ ].*//"`
EXISTING_CONTAINER=""
for CONT in $CONTAINERS; do
    EXISTING_CONTAINER=$CONT
    break
done

if [ "$EXISTING_CONTAINER" = "" ]; then
    docker run --privileged $GPUS --shm-size=${SHMSIZE}gb -it -p 3000:22 -v "$(pwd)"/container_results:/home/${user}/results fastbatllnn-run:${user} ${user}
else
    docker start $EXISTING_CONTAINER
fi