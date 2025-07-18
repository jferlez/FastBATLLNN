FROM fastbatllnn-deps:local

ARG USER_NAME
ARG UID
ARG GID
ARG CORES

RUN sed -i '16i Port 3000' /etc/ssh/sshd_config

# Delete some groups that overlap with MacOS standard user groups
RUN delgroup dialout
RUN delgroup fax
RUN delgroup voice

# Delete any existing user/group with the provided UID/GID
# (24.04 Ubuntu images have an 'ubuntu' user/group with uid/gid 1000 out of the box)
RUN bash -c "OLDUSER=\$(cat /etc/passwd | grep -E '^\w+\:x\:${UID}' | sed -E -e 's/^(\w+)\:.*/\1/' -); if [ \"\$OLDUSER\" != \"\" ]; then deluser \$OLDUSER; fi" && \
    bash -c "GRPNAME=\$(cat /etc/group | grep -E '^\w+\:x\:${GID}' | sed -E -e 's/^(\w+)\:.*/\1/' -); if [ \"\$GRPNAME\" != \"\" ]; then delgroup \$GRPNAME; fi"
RUN addgroup --gid ${GID} ${USER_NAME}
RUN useradd -rm -d /home/${USER_NAME} -s /bin/bash -g ${USER_NAME} -G sudo -u ${UID} ${USER_NAME}
RUN echo "${USER_NAME}:${USER_NAME}" | chpasswd
RUN mkdir -p /home/${USER_NAME}/.ssh

RUN mkdir -p /home/${USER_NAME}/results
RUN mkdir -p /media/azuredata
RUN mkdir -p /media/azuretmp
RUN chown -R ${UID}:${UID} /media/azuredata
RUN chown -R ${UID}:${UID} /media/azuretmp

# switch to unpriviledged user, and configure remote access
WORKDIR /home/${USER_NAME}/tools
RUN chown -R ${UID}:${GID} /home/${USER_NAME}

USER ${USER_NAME}
RUN ssh-keygen -t ed25519 -q -f /home/${USER_NAME}/.ssh/id_ed25519 -N ""
RUN cat /home/${USER_NAME}/.ssh/id_ed25519.pub >> /home/${USER_NAME}/.ssh/authorized_keys

# Install neovim stuff:
RUN mkdir -p /home/${USER_NAME}/.local/share/nvim/site/pack/packer/start
RUN git clone --depth=1 https://github.com/folke/tokyonight.nvim /home/${USER_NAME}/.local/share/nvim/site/pack/packer/start/tokyonight.nvim
RUN mkdir -p /home/${USER_NAME}/.local/nvim
RUN git clone --depth=1 https://github.com/jferlez/nvim-config.git /home/${USER_NAME}/.config/nvim
RUN TEMP="$(/usr/bin/nvim -c 'sleep 140' -c 'qa')"

# Now copy over FastBATLLNN code
RUN git clone --recursive https://github.com/jferlez/FastBATLLNN

# This installs VNNLIB support
#RUN git clone https://github.com/stanleybak/nnenum
RUN ln -s /usr/local/lib/python3.13/dist-packages/nnenum /home/${USER_NAME}/tools/nnenum

# WORKDIR /home/${USER_NAME}/tools/FastBATLLNN/HyperplaneRegionEnum
# RUN python3.9 posetFastCharm_numba.py

WORKDIR /home/${USER_NAME}
RUN git clone https://github.com/jferlez/FastBATLLNN_Experiments_HSCC2022
#RUN echo "export PYTHONPATH=/home/${USER_NAME}/tools/FastBATLLNN:/home/${USER_NAME}/tools/FastBATLLNN/HyperplaneRegionEnum:/home/${USER_NAME}/tools/FastBATLLNN/TLLnet:/home/${USER_NAME}/tools/nnenum/src/nnenum" >> /home/${USER_NAME}/.bashrc
RUN sed -i "4i export PYTHONPATH=/home/${USER_NAME}/tools/FastBATLLNN:/home/${USER_NAME}/tools/FastBATLLNN/HyperplaneRegionEnum:/home/${USER_NAME}/tools/FastBATLLNN/TLLnet:/home/${USER_NAME}/tools/nnenum/src" /home/${USER_NAME}/.bashrc
RUN sed -i "4i export TF_CPP_MIN_LOG_LEVEL=2" /home/${USER_NAME}/.bashrc
RUN echo "export TERM=xterm-256color" >> /home/${USER_NAME}/.bashrc
RUN echo "export COLORTERM=truecolor" >> /home/${USER_NAME}/.bashrc
RUN echo "export TERM_PROGRAM=iTerm2.app" >> /home/${USER_NAME}/.bashrc
RUN echo "set-option -gs default-terminal \"tmux-256color\" # Optional" >> /home/${USER_NAME}/.tmux.conf
RUN echo "set-option -gas terminal-overrides \"*:Tc\"" >> /home/${USER_NAME}/.tmux.conf
RUN echo "set-option -gas terminal-overrides \"*:RGB\"" >> /home/${USER_NAME}/.tmux.conf
RUN echo "set-window-option -g mode-keys vi" >> /home/${USER_NAME}/.tmux.conf
RUN echo "set-option -g history-limit 50000" >> /home/${USER_NAME}/.tmux.conf
WORKDIR /home/${USER_NAME}/tools/FastBATLLNN

USER root
RUN chown -R ${UID}:${GID} /home/${USER_NAME}/

COPY ./DockerConfig/startup.sh /usr/local/bin/startup.sh
RUN chmod 755 /usr/local/bin/startup.sh

ENTRYPOINT [ "/usr/local/bin/startup.sh" ]
