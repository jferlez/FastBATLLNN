#!/bin/bash
USER=$1
INTERACTIVE=$2
SERVER=$3
CORES=$4
/usr/sbin/sshd -D &> /root/sshd_log.out &
sudo -u $USER mkdir /home/$USER/.ssh
if [ -e /etc/ssh/ssh_host_rsa_key.pub ]; then
	echo "
****** SSH host key ******"
	cat /etc/ssh/ssh_host_rsa_key.pub
	echo "**************************
"
    mkdir -p /home/$USER/results/ssh_keys
    sudo -u $USER chown -R $USER:$USER /home/$USER/results/ssh_keys
    sudo -u $USER cp /etc/ssh/ssh_host_rsa_key.pub /home/$USER/results/ssh_keys
    sudo -u $USER sh -c "cat /etc/ssh/ssh_host_rsa_key.pub > /home/$USER/.ssh/known_hosts"
fi
if [ -e /home/$USER/.ssh/id_rsa.pub ]; then
	echo "
****** SSH public key for user $USER ******"
    cat /home/$USER/.ssh/id_rsa.pub
    echo "***************************************
"
    mkdir -p /home/$USER/results/ssh_keys
    sudo -u $USER chown -R $USER:$USER /home/$USER/results/ssh_keys
    sudo -u $USER cp /home/$USER/.ssh/id_rsa.pub /home/$USER/results/ssh_keys/id_rsa_${USER}.pub
    sudo -u $USER sh -c "cat /home/$USER/.ssh/id_rsa.pub > /home/$USER/.ssh/authorized_keys"
fi

# Setup authorized_keys/known_hosts
for fname in authorized_keys known_hosts; do
    if [ -e /home/$USER/results/$fname ]
    then
        sudo -u $USER sh -c "cat /home/$USER/results/$fname >> /home/$USER/.ssh/${fname}_base"
        rm /home/$USER/results/$fname
    fi
    if [ -e /home/$USER/.ssh/${fname}_base ]
    then
        sudo -u $USER sh -c "cat /home/$USER/results/${fname}_base >> /home/$USER/.ssh/${fname}"
    fi
    if [ -e /home/$USER/.ssh/${fname} ]
    then
        chmod 600 /home/$USER/.ssh/${fname} && chown $USER:$USER /home/$USER/.ssh/${fname}
    fi
done


if [ "$SERVER" = "server" ]; then
	sudo -u $USER PYTHONPATH="/home/$USER/tools/FastBATLLNN:/home/$USER/tools/FastBATLLNN/HyperplaneRegionEnum:/home/$USER/tools/FastBATLLNN/TLLnet:/home/$USER/tools/nnenum/src/nnenum" charmrun +p$CORES /home/$USER/tools/FastBATLLNN/FastBATLLNNServer.py &> "/home/$USER/results/FastBATLLNN_server_log.out" &
fi
if [ "$INTERACTIVE" = "-d" ]; then
	wait -n
else
	sudo -u $USER /bin/bash
	killall sshd
fi
