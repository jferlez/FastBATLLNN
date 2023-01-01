# FROM ubuntu:21.10
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
WORKDIR /tmp/docker_build
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get update && apt-get -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/nightly
RUN apt-get update && apt-get -y install python3.10 python3.10-dev python3.10-distutils libpython3.10-dev curl
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3.10 -m pip install --upgrade pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
#RUN python3.10 -m pip install --upgrade pip && python3.10 -m pip install --upgrade html5lib
RUN apt-get update && \
    apt-get -y upgrade && \
    apt -y install clang-12 cmake gfortran gfortran-10 gcc-10 libgsl-dev libfftw3-3 libfftw3-dev libsuitesparse-dev git clang-11 libgmp-dev vim emacs nano screen tmux ipython3 openssh-server sudo curl universal-ctags ripgrep psmisc locales clangd util-linux git-lfs && \
    curl -sL https://deb.nodesource.com/setup_16.x | bash - && \
    apt -y install nodejs && \
    python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install tensorflow scipy greenlet pycddlib Cython pyopencl pycuda cupy-cuda112==10.5.0 mpmath matplotlib onnx onnxruntime tf2onnx torch torchvision torchaudio pylint flake8 vim-vint 'python-lsp-server[all]' pylsp-mypy pyls-isort pynvim cdifflib tree_sitter && \
    python3.10 -m pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN npm install -g vim-language-server node-gyp tree-sitter tree-sitter-cli
RUN git clone https://github.com/UIUC-PPL/charm4py && \
    cd charm4py && \
    git checkout 0aa892b7e09dc0c0b22b75f84103b1bfa7ed3ae3 && \
    sed -i -E -e "s/(\s+)(args\ \+\=)/\1args\.remove('\+\+interactive')\n\1\2/" ./charmrun/start.py && \
    git clone https://github.com/UIUC-PPL/charm charm_src/charm && \
    cd charm_src/charm && \
    git checkout c7483ea419f5319a40993681916d38e3a72bd6be && \
    cd ../..
WORKDIR /tmp/docker_build
# Install MS PowerShell
RUN wget -q "https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb"
RUN sudo dpkg -i packages-microsoft-prod.deb && rm packages-microsoft-prod.deb
RUN sudo apt-get update && sudo apt-get install -y powershell

# Install latest neovim
RUN wget https://github.com/neovim/neovim/releases/download/v0.8.0/nvim-linux64.deb 
RUN apt install ./nvim-linux64.deb
RUN rm ./nvim-linux64.deb
# Build Fast BATLLNN dependencies
RUN wget https://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz && \
    tar -xf glpk-5.0.tar.gz && \
    cd glpk-5.0 && \
    CC=/usr/bin/clang-12 CXX=/usr/bin/clang++-12 CPP=/usr/bin/clang-cpp-12 ./configure --prefix /usr/local && make && make install && \
    cd .. && \
    git clone https://github.com/cvxopt/cvxopt.git && \
    cd cvxopt && \
    git checkout f236615e5bd326daeb6cda511c31d86bcc354747 && \
    CVXOPT_BUILD_FFTW=1 CVXOPT_BUILD_GSL=1 CVXOPT_BUILD_GLPK=1 CVXOPT_GLPK_LIB_DIR=/usr/local/lib CVXOPT_GLPK_INC_DIR=/usr/local/include CC=/usr/bin/clang-12 CXX=/usr/bin/clang++-12 CPP=/usr/bin/clang-cpp-12 python3.10 setup.py install && \
    cd .. && \
    wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew && \
    chmod 755 coinbrew && \
    ./coinbrew --latest-release --no-prompt build Clp --prefix=/usr/local && \
    ./coinbrew --latest-release --no-prompt build Cbc --prefix=/usr/local && \
    COIN_INSTALL_DIR=/usr/local && \
    COIN_INSTALL_DIR=/usr/local python3.10 -m pip install cylp
# Installl OpenMPI with UCX support
RUN wget "https://content.mellanox.com/ofed/MLNX_OFED-5.8-1.1.2.1/MLNX_OFED_LINUX-5.8-1.1.2.1-ubuntu20.04-x86_64.tgz" && \
    tar -xf MLNX_OFED_LINUX-5.8-1.1.2.1-ubuntu20.04-x86_64.tgz && \
    cd MLNX_OFED_LINUX-5.8-1.1.2.1-ubuntu20.04-x86_64 && \
    sudo apt -y install autoconf libgfortran5 gfortran libnuma-dev debhelper kmod ethtool libpci3 flex libfuse2 m4 libelf1 lsof libnl-route-3-dev autotools-dev libnuma1 libnl-route-3-200 libusb-1.0-0 tk libnl-3-dev swig tcl libnl-3-200 pciutils udev libltdl-dev libmnl0 bison graphviz automake pkg-config chrpath dpatch && \
    ./mlnxofedinstall --user-space-only --force && \
    cd ..
RUN apt-get -y install zlib1g zlib1g-dev
RUN cd charm4py && \
    PATH="/usr/mpi/gcc/openmpi-4.1.5a1/bin:$PATH" && \
    python3.10 setup.py install --mpi && \
    cd .. && \
    git clone https://github.com/numba/numba && \
    git clone https://github.com/numba/llvmlite && \
    cd llvmlite && \
    git checkout b9c748c6dc786857e8ba09382cfdce03533b7c03 && \
    LLVM_CONFIG=llvm-config-11 python3.10 setup.py install && \
    cd ../numba && \
    git checkout 63d0c471410063b716797425aeaf244b9fa8b2b7 && \
    python3.10 setup.py install && \
    cd .. && \
    rm -rf *

# Install nnenum globally:
WORKDIR /usr/local/lib/python3.10/dist-packages
RUN git clone https://github.com/stanleybak/nnenum
WORKDIR /tmp/docker_build

# Configuration to enable UCX/InfiniBand Networking:
RUN echo "* soft memlock unlimited\n* hard memlock unlimited" >> /etc/security/limits.conf
# Patch to enable profiling in charm4py
RUN sed -i -E -e 's/(assert not self.running and charm.runningEntryMethod is None)/\#\1/' -e 's/(assert self.running and charm.runningEntryMethod == self)/\#\1/'  /usr/local/lib/python3.10/dist-packages/charm4py/entry_method.py
RUN sed -i -E -e 's/(\s*self\.options\.local_msg_optim\s*=\s*)False/\1True/' /usr/local/lib/python3.10/dist-packages/charm4py/charm.py
# RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 ubuntu
# RUN echo 'ubuntu:ubuntu' | chpasswd
RUN /usr/bin/ssh-keygen -A

# To get nohup with mpirun... not needed, though https://stackoverflow.com/questions/48296285/nohup-does-not-work-mpirun
#RUN echo "#!/bin/sh\n(\necho \"Date: \`date\`\"\necho \"Command: \$*\"\nPYTHONPATH=/home/${USER_NAME}/tools/FastBATLLNN:/home/${USER_NAME}/tools/FastBATLLNN/HyperplaneRegionEnum:/home/${USER_NAME}/tools/FastBATLLNN/TLLnet:/home/${USER_NAME}/tools/nnenum/src/nnenum nohup \"\$@\"\necho \"Completed: \`date\`\"\necho\n) >>\${LOGFILE:=log.out} 2>&1 &" > /usr/local/bin/bk
# Use bash for this script, since for all shells (login/non-login/interactive/non-interactive) it will use /etc/bash.bashrc
# See: https://unix.stackexchange.com/questions/170493/login-non-login-and-interactive-non-interactive-shells
RUN echo "#!/bin/bash" > /usr/local/bin/bk
RUN echo "(" >>  /usr/local/bin/bk
RUN echo 'echo "Date: `date`"' >>  /usr/local/bin/bk
RUN echo 'echo "Command: $*"' >>  /usr/local/bin/bk
RUN echo 'nohup "$@"' >>  /usr/local/bin/bk
RUN echo 'echo "Completed: `date`"' >>  /usr/local/bin/bk
RUN echo "" >>  /usr/local/bin/bk
RUN echo ") >>\${LOGFILE:=log.out} 2>&1 &" >>  /usr/local/bin/bk
RUN chmod 755 /usr/local/bin/bk

RUN echo "#!/bin/bash\n/usr/sbin/sshd -D &> /root/sshd_log.out &\nif [ -e /etc/ssh/ssh_host_rsa_key.pub ]\nthen\n echo \"\n****** SSH host key ******\"\ncat /etc/ssh/ssh_host_rsa_key.pub\necho \"**************************\n\"\nsudo -u \$1 cp /etc/ssh/ssh_host_rsa_key.pub /home/\$1/results\nfi\nif [ -e /home/\$1/results/authorized_keys ] && [ -d /home/\$1/.ssh ]\nthen\ncp /home/\$1/results/authorized_keys /home/\$1/.ssh\nchmod 600 /home/\$1/.ssh/authorized_keys && chown \$1:\$1 /home/\$1/.ssh/authorized_keys && rm /home/\$1/results/authorized_keys\nfi\nif [ \"\$2\" = \"-d\" ]; then\nwait -n\nelse\nsudo -u \$1 /bin/bash\nkillall sshd\nfi" > /usr/local/bin/startup.sh
RUN chmod 755 /usr/local/bin/startup.sh

#RUN echo "export PATH=/usr/local/bin:\$PATH\nexport LD_LIBRARY_PATH=/usr/local/lib\nexport COIN_INSTALL_DIR=/usr/local\nexport PATH=\"/usr/mpi/gcc/openmpi-4.1.2a1/bin:$PATH\"" >> /etc/bash.bashrc
RUN sed -i "2i export PATH=/usr/local/bin:/usr/mpi/gcc/openmpi-4.1.5a1/bin:\$PATH" /etc/bash.bashrc
RUN sed -i "3i export LD_LIBRARY_PATH=/usr/local/lib" /etc/bash.bashrc
RUN sed -i "4i export COIN_INSTALL_DIR=/usr/local" /etc/bash.bashrc

# RUN chown ubuntu:root /home/ubuntu/.bashrc
# RUN rm /etc/ssh/ssh_host_rsa_key /etc/ssh/ssh_host_ecdsa_key && \
#     ssh-keygen -f /etc/ssh/ssh_host_rsa_key -N '' -t rsa && \
#     ssh-keygen -f /etc/ssh/ssh_host_dsa_key -N '' -t dsa && \
#     ssh-keygen -f /etc/ssh/ssh_host_ecdsa_key -N '' -t ecdsa -b 521
# RUN /usr/bin/ssh-keygen -A
RUN sed -i -E -e 's/\s*#\s*PasswordAuthentication\s+(yes|no)/PasswordAuthentication no/' /etc/ssh/sshd_config
RUN service ssh start
EXPOSE 3000
# CMD ["/bin/bash"]

# Thsese are related to neovim; in future builds, they will be incorporated into earlier layers
RUN locale-gen en_US.UTF-8

