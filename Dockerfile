FROM jferlez/fastbatllnn:deps
# switch to unpriviledged user, and configure remote access
WORKDIR /home/ubuntu/tools/FastBATLLNN
RUN chown -R ubuntu:root /home/ubuntu/tools

USER ubuntu
# Now copy over code
COPY --chown=ubuntu:root . .

WORKDIR /home/ubuntu/tools/FastBATLLNN/HyperplaneRegionEnum
RUN python3 posetFastCharm_numba.py

WORKDIR /home/ubuntu
RUN echo "export PYTHONPATH=/home/ubuntu/tools/FastBATLLNN/HyperplaneRegionEnum:/home/ubuntu/tools/FastBATLLNN/Simple2xHRep" >> /home/ubuntu/.profile
WORKDIR /home/ubuntu/tools/FastBATLLNN

USER root
CMD /usr/local/bin/startup.sh