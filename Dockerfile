FROM jferlez/fastbatllnn:deps
# switch to unpriviledged user, and configure remote access
WORKDIR /home/ubuntu/tools/FastBATLLNN
RUN chown -R ubuntu:root /home/ubuntu/tools

USER ubuntu
# Now copy over code
COPY --chown=ubuntu:root . .
RUN python3 posetFastCharm_numba.py
USER root
CMD /usr/local/bin/startup.sh