#!/bin/bash
USER=$1
INTERACTIVE=$2
SERVER=$3
CORES=$4
PORTNUM=$5
MPIHOSTS=$6
MPIARGS=$7
if [ $MPIHOSTS == "127.0.0.1:localhost" ]; then
    MPIHOSTS=""
fi
/usr/sbin/sshd -D &> /root/sshd_log.out &
if [ ! -d /home/$USER/.ssh ]
then
    sudo -u $USER mkdir /home/$USER/.ssh
fi
for type in ed25519 rsa; do
    if [ -e "/etc/ssh/ssh_host_${type}_key.pub" ]; then
        echo "
    ****** SSH host key ******"
        cat /etc/ssh/ssh_host_${type}_key.pub
        echo "**************************
    "
        sudo -u $USER mkdir -p /home/$USER/results/ssh_keys
        sudo -u $USER chown -R $USER:$USER /home/$USER/results/ssh_keys
        sudo -u $USER cp /etc/ssh/ssh_host_${type}_key.pub /home/$USER/results/ssh_keys
        HOSTKEY=`cat "/etc/ssh/ssh_host_${type}_key.pub"`
        sudo -u $USER sh -c "echo \"*:$PORTNUM $HOSTKEY\" > /home/$USER/.ssh/known_hosts"
        break
    fi
done
for type in ed25519 rsa; do
    if [ -e "/home/$USER/.ssh/id_${type}.pub" ]; then
        echo "
    ****** SSH public key for user $USER ******"
        cat /home/$USER/.ssh/id_${type}.pub
        echo "***************************************
    "
        sudo -u $USER mkdir -p /home/$USER/results/ssh_keys
        sudo -u $USER chown -R $USER:$USER /home/$USER/results/ssh_keys
        sudo -u $USER cp /home/$USER/.ssh/id_${type}.pub /home/$USER/results/ssh_keys/id_${type}_${USER}.pub
        sudo -u $USER sh -c "cat /home/$USER/.ssh/id_${type}.pub > /home/$USER/.ssh/authorized_keys"
        break
    fi
done

# Setup authorized_keys/known_hosts
for fname in authorized_keys known_hosts; do
    if [ -e /home/$USER/results/$fname ]
    then
        sudo -u $USER sh -c "cat /home/$USER/results/$fname >> /home/$USER/.ssh/${fname}_base"
        rm /home/$USER/results/$fname
    fi
    if [ -e /home/$USER/.ssh/${fname}_base ]
    then
        sudo -u $USER sh -c "cat /home/$USER/.ssh/${fname}_base >> /home/$USER/.ssh/${fname}"
    fi
    if [ -e /home/$USER/.ssh/${fname} ]
    then
        chmod 600 /home/$USER/.ssh/${fname} && chown $USER:$USER /home/$USER/.ssh/${fname}
    fi
done

PYPATH="/home/$USER/tools/FastBATLLNN:/home/$USER/tools/FastBATLLNN/HyperplaneRegionEnum:/home/$USER/tools/FastBATLLNN/TLLnet:/home/$USER/tools/nnenum/src/nnenum"

if [ "$MPIHOSTS" != "" ]; then
    echo "$MPIHOSTS" | sed -e 's/,/\
/g' -e 's/:/    /g' >> /etc/hosts
    HOSTLIST=`echo "$MPIHOSTS" | sed -E -e 's/:[^:,]+/:-1/g'`
    echo "#!/bin/bash
source /etc/bash.bashrc
mpirun $MPIARGS -mca plm_rsh_args \"-p 3000\" -np $CORES -host $HOSTLIST -x PYTHONPATH=\"$PYPATH:\$PYTHONPATH\" -x TF_CPP_MIN_LOG_LEVEL=2 /usr/bin/python3.13 \"\$@\"" > /usr/local/bin/charming
else
    echo "#!/bin/bash
source /etc/bash.bashrc
mpirun $MPIARGS -np $CORES -x PYTHONPATH=\"$PYPATH:\$PYTHONPATH\" -x TF_CPP_MIN_LOG_LEVEL=2 /usr/bin/python3.13 \"\$@\"" > /usr/local/bin/charming
fi
chmod 755 /usr/local/bin/charming

if [ "$SERVER" = "server" ]; then
	sudo -u $USER /usr/local/bin/charming /home/$USER/tools/FastBATLLNN/FastBATLLNNServer.py &> "/home/$USER/results/FastBATLLNN_server_log.out" &
fi
if [ "$INTERACTIVE" = "-d" ]; then
	wait -n
else
	sudo -i -u $USER /bin/bash -l -i
	killall sshd
fi
