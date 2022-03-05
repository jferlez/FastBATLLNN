FROM jferlez/fastbatllnn:deps

ARG USER_NAME
ARG UID
ARG GID

RUN addgroup --gid ${GID} ${USER_NAME}
RUN useradd -rm -d /home/${USER_NAME} -s /bin/bash -g ${USER_NAME} -G sudo -u ${UID} ${USER_NAME}

# switch to unpriviledged user, and configure remote access
WORKDIR /home/${USER_NAME}/tools/FastBATLLNN
RUN chown -R ${UID}:${GID} /home/ubuntu/tools

USER ${USER_NAME}
# Now copy over code
COPY --chown=${UID}:${GID} . .

WORKDIR /home/${USER_NAME}/tools/FastBATLLNN/HyperplaneRegionEnum
RUN python3.9 posetFastCharm_numba.py

WORKDIR /home/${USER_NAME}
RUN echo "export PYTHONPATH=/home/${USER_NAME}/tools/FastBATLLNN:/home/${USER_NAME}/tools/FastBATLLNN/HyperplaneRegionEnum:/home/${USER_NAME}/tools/FastBATLLNN/Simple2xHRep" >> /home/${USER_NAME}/.bashrc
WORKDIR /home/${USER_NAME}/tools/FastBATLLNN

USER root
CMD /usr/local/bin/startup.sh ${USER_NAME}