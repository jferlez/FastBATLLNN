#!/bin/bash
USER=$1
INTERACTIVE=$2
SERVER=$3
CORES=$4
/usr/sbin/sshd -D &> /root/sshd_log.out &
if [ -e /etc/ssh/ssh_host_rsa_key.pub ]; then
	echo "
****** SSH host key ******"
	cat /etc/ssh/ssh_host_rsa_key.pub
	echo "**************************
"
	sudo -u $USER cp /etc/ssh/ssh_host_rsa_key.pub /home/$USER/results
fi
if [ -e /home/$USER/results/authorized_keys ] && [ -d /home/$USER/.ssh ]
then
	cp /home/$USER/results/authorized_keys /home/$USER/.ssh
	chmod 600 /home/$USER/.ssh/authorized_keys && chown $USER:$USER /home/$USER/.ssh/authorized_keys && rm /home/$USER/results/authorized_keys
fi
if [ "$SERVER" = "server" ]; then
	sudo -u $USER PYTHONPATH="/home/$USER/tools/FastBATLLNN:/home/$USER/tools/FastBATLLNN/HyperplaneRegionEnum:/home/$USER/tools/FastBATLLNN/TLLnet:/home/$USER/tools/nnenum/src/nnenum" charmrun +p$CORES /home/$USER/tools/FastBATLLNN/FastBATLLNNServer.py &> "/home/$USER/results/FastBATLLNN_server_log.out" &
fi
if [ "$INTERACTIVE" = "-d" ]; then
	wait -n
else
	sudo -u $USER /bin/bash
	killall sshd
fi
