# FROM ubuntu:21.10
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
WORKDIR /tmp/docker_build
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get update && \
    apt-get -y upgrade && \
    apt -y install clang-12 python3-pip python3.9 python3.9-dev libopenmpi-dev cmake gcc-10 libgsl-dev libfftw3-3 libfftw3-dev libsuitesparse-dev git clang-11 libgmp-dev vim emacs nano screen ipython3 openssh-server sudo && \
    python3.9 -m pip install --upgrade pip && \
    python3.9 -m pip install tensorflow-gpu==2.8.0 scipy greenlet pycddlib Cython pyopencl pycuda cupy-cuda112
RUN git clone https://github.com/UIUC-PPL/charm4py && \
    cd charm4py && \
    sed -i -E -e "s/(\s+)(args\ \+\=)/\1args\.remove('\+\+interactive')\n\1\2/" ./charmrun/start.py && \
    git clone https://github.com/UIUC-PPL/charm charm_src/charm
WORKDIR /tmp/docker_build
# Build Fast BATLLNN dependencies
RUN wget https://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz && \
    tar -xf glpk-5.0.tar.gz && \
    cd glpk-5.0 && \
    CC=/usr/bin/clang-12 CXX=/usr/bin/clang++-12 CPP=/usr/bin/clang-cpp-12 ./configure --prefix /usr/local && make && make install && \
    cd .. && \
    git clone https://github.com/cvxopt/cvxopt.git && \
    cd cvxopt && \
    CVXOPT_BUILD_FFTW=1 CVXOPT_BUILD_GSL=1 CVXOPT_BUILD_GLPK=1 CVXOPT_GLPK_LIB_DIR=/usr/local/lib CVXOPT_GLPK_INC_DIR=/usr/local/include CC=/usr/bin/clang-12 CXX=/usr/bin/clang++-12 CPP=/usr/bin/clang-cpp-12 python3.9 setup.py install && \
    cd .. && \
    wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew && \
    chmod 755 coinbrew && \
    ./coinbrew --latest-release --no-prompt build Clp --prefix=/usr/local && \
    ./coinbrew --latest-release --no-prompt build Cbc --prefix=/usr/local && \
    COIN_INSTALL_DIR=/usr/local && \
    COIN_INSTALL_DIR=/usr/local python3.9 -m pip install cylp && \
    cd charm4py && \
    python3.9 setup.py install --mpi && \
    cd .. && \
    git clone https://github.com/numba/numba && \
    git clone https://github.com/numba/llvmlite && \
    cd llvmlite && \
    LLVM_CONFIG=llvm-config-11 python3.9 setup.py install && \
    cd ../numba && \
    python3.9 setup.py install && \
    cd .. && \
    rm -rf *
# Patch to enable profiling in charm4py
RUN sed -i -E -e 's/(assert not self.running and charm.runningEntryMethod is None)/\#\1/' -e 's/(assert self.running and charm.runningEntryMethod == self)/\#\1/'  /usr/local/lib/python3.9/dist-packages/charm4py/entry_method.py
# RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 ubuntu
# RUN echo 'ubuntu:ubuntu' | chpasswd
RUN /usr/bin/ssh-keygen -A
RUN echo "#!/bin/bash\nnohup /usr/sbin/sshd -D &> /root/nohup.out &\nsudo -u ${1} /bin/bash" > /usr/local/bin/startup.sh
RUN chmod 700 /usr/local/bin/startup.sh
RUN echo "export PATH=/usr/local/bin:$PATH\nexport LD_LIBRARY_PATH=/usr/local/lib\nexport COIN_INSTALL_DIR=/usr/local" >> /etc/bash.bashrc
# RUN chown ubuntu:root /home/ubuntu/.bashrc
# RUN rm /etc/ssh/ssh_host_rsa_key /etc/ssh/ssh_host_ecdsa_key && \
#     ssh-keygen -f /etc/ssh/ssh_host_rsa_key -N '' -t rsa && \
#     ssh-keygen -f /etc/ssh/ssh_host_dsa_key -N '' -t dsa && \
#     ssh-keygen -f /etc/ssh/ssh_host_ecdsa_key -N '' -t ecdsa -b 521
# RUN /usr/bin/ssh-keygen -A
RUN service ssh start
EXPOSE 22
# CMD ["/bin/bash"]
