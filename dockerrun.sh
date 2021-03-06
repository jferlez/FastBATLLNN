#!/bin/bash

user=`id -n -u`
uid=`id -u`
gid=`id -g`

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

SYSTEM_TYPE=$(uname)
PORT=3000
HTTPPORT=8000
INTERACTIVE="-d"
SERVER="run"
ATTACH=""
for argwhole in "$@"; do
    IFS='=' read -r -a array <<< "$argwhole"
    arg="${array[0]}"
    val="${array[1]}"
    case "$arg" in
        --gpu) GPUS="--gpus all";;
        --ssh-port) PORT=`echo "$val" | sed -e 's/[^0-9]//g'`;;
        --http-port) HTTPPORT=`echo "$val" | sed -e 's/[^0-9]//g'`;;
        --interactive) INTERACTIVE="-it" && ATTACH="-ai";;
        --server) SERVER="server"
    esac
done

re='^[0-9]+$'
if ! [[ $PORT =~ $re ]] ; then
    echo "error: Invalid port specified" >&2; exit 1
fi
PORT="-p $PORT:22"

if ! [[ $HTTPPORT =~ $re ]] ; then
    echo "error: Invalid port specified" >&2; exit 1
fi

if [ "$SERVER" = "server" ]; then
    HTTPPORT="-p ${HTTPPORT}:8080"
else
    HTTPPORT=""
fi

if [ "$SYSTEM_TYPE" = "Darwin" ]; then
    SHMSIZE=$(( `sysctl hw.memsize | sed -e 's/[^0-9]//g'` / 2097152 ))
    # Never enable GPUs on MacOS
    GPUS=""
    CORES=$(( `sysctl -n hw.ncpu` / 2 ))
else
    SHMSIZE=$(( `grep MemTotal /proc/meminfo | sed -e 's/[^0-9]//g'` / 2097152 ))
    CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket:" | sed -e 's/[^0-9]//g'`
    SOCKETS=`lscpu | grep "Socket(s):" | sed -e 's/[^0-9]//g'`
    CORES=$(( $CORES_PER_SOCKET * $SOCKETS ))
    PYTHON=""
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
    if [ `docker inspect --format='{{.Config.Labels.server}}' $CONT` = "${SERVER}" ]; then
        EXISTING_CONTAINER=$CONT
        break
    fi
done

if [ "$EXISTING_CONTAINER" = "" ]; then
    docker run --privileged $GPUS --shm-size=${SHMSIZE}gb $INTERACTIVE $PORT $HTTPPORT --label server=${SERVER} -v "$(pwd)"/container_results:/home/${user}/results fastbatllnn-run:${user} ${user} $INTERACTIVE $SERVER $CORES
else
    echo "Restarting container $EXISTING_CONTAINER (command line options except \"--server\" ignored)..."
    docker start $ATTACH $EXISTING_CONTAINER
fi