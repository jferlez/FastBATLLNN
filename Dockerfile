FROM fastbatllnn-deps:local

ARG USER_NAME
ARG UID
ARG GID
ARG CORES

RUN apt-get install psmisc

# Delete some groups that overlap with MacOS standard user groups
RUN delgroup --only-if-empty dialout
RUN delgroup --only-if-empty fax
RUN delgroup --only-if-empty voice

RUN addgroup --gid ${GID} ${USER_NAME}
RUN useradd -rm -d /home/${USER_NAME} -s /bin/bash -g ${USER_NAME} -G sudo -u ${UID} ${USER_NAME}
RUN echo "${USER_NAME}:${USER_NAME}" | chpasswd
RUN mkdir -p /home/${USER_NAME}/.ssh

RUN mkdir -p /home/${USER_NAME}/results

# switch to unpriviledged user, and configure remote access
WORKDIR /home/${USER_NAME}/tools
RUN chown -R ${UID}:${GID} /home/${USER_NAME}

USER ${USER_NAME}
RUN ssh-keygen -t rsa -q -f /home/${USER_NAME}/.ssh/id_rsa -N ""
# Now copy over code
RUN git clone --recursive https://github.com/jferlez/FastBATLLNN

# This installs VNNLIB support
RUN git clone https://github.com/stanleybak/nnenum

# WORKDIR /home/${USER_NAME}/tools/FastBATLLNN/HyperplaneRegionEnum
# RUN python3.9 posetFastCharm_numba.py

WORKDIR /home/${USER_NAME}
RUN git clone https://github.com/jferlez/FastBATLLNN_Experiments_HSCC2022
RUN echo "export PYTHONPATH=/home/${USER_NAME}/tools/FastBATLLNN:/home/${USER_NAME}/tools/FastBATLLNN/HyperplaneRegionEnum:/home/${USER_NAME}/tools/FastBATLLNN/TLLnet:/home/${USER_NAME}/tools/nnenum/src/nnenum" >> /home/${USER_NAME}/.bashrc
WORKDIR /home/${USER_NAME}/tools/FastBATLLNN

USER root
RUN chown -R ${UID}:${GID} /home/${USER_NAME}/

# To get nohup with mpirun... not needed, though https://stackoverflow.com/questions/48296285/nohup-does-not-work-mpirun
RUN echo "#!/bin/sh\n(\necho \"Date: \`date\`\"\necho \"Command: \$*\"\nPYTHONPATH=/home/${USER_NAME}/tools/FastBATLLNN:/home/${USER_NAME}/tools/FastBATLLNN/HyperplaneRegionEnum:/home/${USER_NAME}/tools/FastBATLLNN/TLLnet:/home/${USER_NAME}/tools/nnenum/src/nnenum nohup \"\$@\"\necho \"Completed: \`date\`\"\necho\n) >>\${LOGFILE:=log.out} 2>&1 &" > /usr/local/bin/bk
RUN chmod 755 /usr/local/bin/bk
#RUN echo "#!/bin/bash\n/usr/sbin/sshd -D &> /root/nohup.out &\nif [ -e /etc/ssh/ssh_host_rsa_key.pub ]\nthen\n echo \"\n****** SSH host key ******\"\ncat /etc/ssh/ssh_host_rsa_key.pub\necho \"**************************\n\"\nsudo -u \$1 cp /etc/ssh/ssh_host_rsa_key.pub /home/\$1/results\nfi\nif [ -e /home/\$1/results/authorized_keys ] && [ -d /home/\$1/.ssh ]\nthen\ncp /home/\$1/results/authorized_keys /home/\$1/.ssh\nchmod 600 /home/\$1/.ssh/authorized_keys && chown \$1:\$1 /home/\$1/.ssh/authorized_keys && rm /home/\$1/results/authorized_keys\nfi\nsudo -u \$1 bk charmrun +p${CORES} /home/\$1/tools/FastBATLLNN-VNNCOMP/FastBATLLNN/FastBATLLNNServer.py\nif [ \"\$2\" = \"-d\" ]; then\nwait -n\nelse\nsudo -u \$1 /bin/bash\nfi" > /usr/local/bin/startup.sh
RUN echo "#!/bin/bash\n/usr/sbin/sshd -D &> /root/sshd_log.out &\nif [ -e /etc/ssh/ssh_host_rsa_key.pub ]\nthen\n echo \"\n****** SSH host key ******\"\ncat /etc/ssh/ssh_host_rsa_key.pub\necho \"**************************\n\"\nsudo -u \$1 cp /etc/ssh/ssh_host_rsa_key.pub /home/\$1/results\nfi\nif [ -e /home/\$1/results/authorized_keys ] && [ -d /home/\$1/.ssh ]\nthen\ncp /home/\$1/results/authorized_keys /home/\$1/.ssh\nchmod 600 /home/\$1/.ssh/authorized_keys && chown \$1:\$1 /home/\$1/.ssh/authorized_keys && rm /home/\$1/results/authorized_keys\nfi\nif [ \"\$3\" = \"server\" ]; then\nsudo -u \$1 PYTHONPATH=/home/${USER_NAME}/tools/FastBATLLNN:/home/${USER_NAME}/tools/FastBATLLNN/HyperplaneRegionEnum:/home/${USER_NAME}/tools/FastBATLLNN/TLLnet:/home/${USER_NAME}/tools/nnenum/src/nnenum charmrun +p${CORES} /home/\$1/tools/FastBATLLNN/FastBATLLNNServer.py &> /home/\$1/FastBATLLNN/container_results/FastBATLLNN_server_log.out &\nfi\nif [ \"\$2\" = \"-d\" ]; then\nwait -n\nelse\nsudo -u \$1 /bin/bash\nkillall sshd\nfi" > /usr/local/bin/startup.sh
RUN chmod 755 /usr/local/bin/startup.sh

ENTRYPOINT [ "/usr/local/bin/startup.sh" ]