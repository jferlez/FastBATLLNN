# FROM ubuntu:21.10
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
WORKDIR /tmp/docker_build
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get update && apt-get -y install software-properties-common && \
    add-apt-repository ppa:neovim-ppa/unstable && \
    add-apt-repository ppa:deadsnakes/nightly && \
    apt-get update && apt-get -y install python3.13 python3.13-dev libpython3.13-dev curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13 && \
    python3.13 -m pip install --upgrade pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1 && \
    # New install mechanism for Mellanox drivers \
    export DOCA_URL="https://linux.mellanox.com/public/repo/doca/3.0.0/ubuntu24.04/x86_64/" && \
    curl https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | gpg --dearmor > /etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub && \
    echo "deb [signed-by=/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub] $DOCA_URL ./" > /etc/apt/sources.list.d/doca.list && \
    apt-get update && \
    apt-get -y upgrade && \
    apt -y install clang-14 clang-15 clang-18 cmake gfortran gfortran-14 gcc-14 g++-14 libstdc++-14-dev libgsl-dev libcdd-dev libfftw3-dev libsuitesparse-dev git libgmp-dev vim emacs nano screen tmux ipython3 openssh-server sudo curl universal-ctags ripgrep psmisc locales clangd util-linux git-lfs htop zlib1g zlib1g-dev && \
    apt -y install autoconf libnuma-dev debhelper kmod ethtool libpci3 flex libfuse2 m4 libelf1 lsof libnl-route-3-dev autotools-dev libnuma1 libnl-route-3-200 libusb-1.0-0 tk libnl-3-dev swig tcl libnl-3-200 pciutils udev libltdl-dev libmnl0 bison graphviz automake pkg-config chrpath && \
    curl -sL https://deb.nodesource.com/setup_22.x | bash - && \
    apt -y install nodejs && \
    npm install -g vim-language-server node-gyp tree-sitter tree-sitter-cli && \
    python3.13 -m pip install --upgrade pip
RUN sudo apt -y remove python3-traitlets python3-numpy python3-kiwisolver && python3.13 -m pip install --use-pep517 build==1.2.2.post1 protobuf==6.31.1 numpy==2.1.3 scipy==1.16.0 greenlet==3.2.3 tf-nightly==2.20.0.dev20250711 pycddlib==3.0.2 Cython==3.0.12 mpmath==1.3.0 matplotlib==3.10.3 onnxruntime==1.22.1 pylint==3.3.7 flake8==7.3.0 vim-vint==0.3.21 swiglpk==5.0.12 threadpoolctl==3.6.0 skl2onnx==1.19.1 python-lsp-server==1.12.2 pylsp-mypy pyls-isort pynvim==0.5.2 cdifflib==1.2.6 tree_sitter jupyterlab ipympl portalocker Pillow==11.3.0 kiwisolver==1.4.8 jedi==0.19.2 parso==0.8.4 py-cpuinfo==9.0.0 psutil==7.0.0 traitlets==5.14.3 && \
    python3.13 -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 && \
    python3.13 -m pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#  cupy-cuda12x==13.5.1  scipy==1.17.0
# RUN git clone https://github.com/google/jax && \
#     cd jax && \
#     python3.11 build/build.py --enable_cuda --cuda_version='11.8' --cudnn_version='8.9.0' && \
#     python3.11 -m pip install dist/*.whl && \
#     python3.11 -m pip install -e . && \
#     cd .. && \
#     rm -rf jax/
    # Non-frozen version install command:
    # python3.11 -m pip install numpy tensorflow scipy greenlet pycddlib Cython pyopencl pycuda cupy-cuda112==10.5.0 mpmath matplotlib onnx onnxruntime tf2onnx torch torchvision torchaudio pylint flake8 vim-vint 'python-lsp-server[all]' pylsp-mypy pyls-isort pynvim cdifflib tree_sitter && \
RUN git clone https://github.com/mckib2/scikit-glpk && \
    sed -i -E -e 's/24/23/' ./scikit-glpk/requirements.txt && \
    sed -i -E -e 's#xprintf#//xprintf#' ./scikit-glpk/glpk-5.0/src/api/advbas.c && \
    sed -i -E -e 's#xprintf#//xprintf#' ./scikit-glpk/glpk-5.0/src/api/cpxbas.c && \
    sed -i -E -e 's#xprintf#//xprintf#' ./scikit-glpk/glpk-5.0/src/draft/glpscl.c && \
    sed -i -E -e 's/warn\(/#warn\(/' ./scikit-glpk/glpk/_glpk.py && \
    sed -i -E -e 's/GLPK\.RET_CODES\[ret_code\],/GLPK\.RET_CODES\[ret_code\],'"'"'ret_code'"'"':ret_code,/' ./scikit-glpk/glpk/_glpk.py && \
    cd scikit-glpk && \
    CC=/usr/bin/clang-14 CXX=/usr/bin/clang++-14 CPP=/usr/bin/clang-cpp-14 python3.13 -m pip install --use-pep517 . && \
    cd .. && rm -rf scikit-glpk
# RUN git clone https://github.com/onnx/tensorflow-onnx && \
#     cd tensorflow-onnx && \
#     CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 CPP=/usr/bin/cpp-14 python3.13 -m pip install --use-pep517 . && \
#     cd .. && rm -rf tensorflow-onnx
    # cd tensorflow-onnx && git checkout 554d90ad8b8e0bf089ed10550c4c4a253ac1f717 && \
    # sed -i -E -e 's/concrete_func\.graph\.\_captures/concrete_func\.graph\.function_captures\.by_val_internal/' ./tf2onnx/convert.py && \
    # sed -i -E -e 's/t_name\.name\ for\ t_val\,\ t_name/t\.name\ for\ t/' ./tf2onnx/convert.py && \
    # sed -i -E -e 's/<3\.0\,//' setup.py && \

WORKDIR /tmp/docker_build
# Install MS PowerShell
RUN wget -q "https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb" && \
    sudo dpkg -i packages-microsoft-prod.deb && rm packages-microsoft-prod.deb && \
    sudo apt-get update && sudo apt-get install -y powershell

# Install latest neovim
RUN sudo apt-get -y install neovim
# Build Fast BATLLNN dependencies
RUN wget https://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz && \
    tar -xf glpk-5.0.tar.gz && \
    cd glpk-5.0 && \
    CC=/usr/bin/clang-14 CXX=/usr/bin/clang++-14 CPP=/usr/bin/clang-cpp-14 ./configure --prefix /usr/local && make && make install && \
    cd .. && rm -rf glpk-5.0 && \
    git clone https://github.com/cvxopt/cvxopt.git && \
    cd cvxopt && \
    git checkout f236615e5bd326daeb6cda511c31d86bcc354747 && \
    CVXOPT_BUILD_FFTW=1 CVXOPT_BUILD_GSL=1 CVXOPT_BUILD_GLPK=1 CVXOPT_GLPK_LIB_DIR=/usr/local/lib CVXOPT_GLPK_INC_DIR=/usr/local/include CC=/usr/bin/clang-14 CXX=/usr/bin/clang++-14 CPP=/usr/bin/clang-cpp-14 python3.13 -m pip install --use-pep517 . && \
    cd .. && rm -rf cvxopt && \
    wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew && \
    chmod 755 coinbrew && \
    CC=/usr/bin/clang-14 CXX=/usr/bin/clang++-14 CPP=/usr/bin/clang-cpp-14 ./coinbrew --latest-release --no-prompt build Clp --prefix=/usr/local && \
    CC=/usr/bin/clang-14 CXX=/usr/bin/clang++-14 CPP=/usr/bin/clang-cpp-14 ./coinbrew --latest-release --no-prompt build Cbc --prefix=/usr/local && \
    COIN_INSTALL_DIR=/usr/local && \
    COIN_INSTALL_DIR=/usr/local python3.13 -m pip install cylp && \
    rm -rf *
# Installl OpenMPI with UCX support
# NB: dependencies are
# apt -y install autoconf libgfortran5 gfortran libnuma-dev debhelper kmod ethtool libpci3 flex libfuse2 m4 libelf1 lsof libnl-route-3-dev autotools-dev libnuma1 libnl-route-3-200 libusb-1.0-0 tk libnl-3-dev swig tcl libnl-3-200 pciutils udev libltdl-dev libmnl0 bison graphviz automake pkg-config chrpath dpatch
# RUN wget "https://content.mellanox.com/ofed/MLNX_OFED-24.10-3.5.5.0/MLNX_OFED_LINUX-24.10-3.2.5.0-ubuntu24.04-x86_64.tgz" && \
#     tar -xf MLNX_OFED_LINUX-24.10-3.2.5.0-ubuntu24.04-x86_64.tgz && \
#     cd MLNX_OFED_LINUX-23.10-0.5.5.0-ubuntu22.04-x86_64 && \
#     ./mlnxofedinstall --user-space-only --force --without-ucx-cuda && \
#     cd .. && \
#     rm -rf *
RUN apt -y install doca-ofed
# Install charm4py
RUN git clone https://github.com/UIUC-PPL/charm4py && \
    cd charm4py && \
    git checkout a067434616f23ef726e6076ebfe7a34dc02a317a && \
    sed -i -E -e "s/(\s+)(args\ \+\=)/\1args\.remove('\+\+interactive')\n\1\2/" ./charmrun/start.py && \
    git clone https://github.com/UIUC-PPL/charm charm_src/charm && \
    cd charm_src/charm && \
    git checkout 3cf60d56fd9db5b399dc4569d7cc7f77e73a8ea1 && \
    cd ../.. && \
    PATH="/usr/mpi/gcc/openmpi-4.1.7rc1/bin:$PATH" && \
    python3.13 -m pip install --use-pep517 --config-settings="--mpi=True" . && \
    cd .. && \
    git clone https://github.com/numba/numba && \
    git clone https://github.com/numba/llvmlite && \
    cd llvmlite && \
    git checkout 2c67a6b8deaaa21cf23e6cb59cf66905b63281ba && \
    LLVM_CONFIG=llvm-config-15 python3.13 -m pip install --use-pep517 .  && \
    cd ../numba && git fetch && git pull && \
    git checkout 1e70d8ceba56a135e046e32e1e7ad2fcd22fd8ab && \
    python3.13 -m pip install --use-pep517 . && \
    cd .. && \
    rm -rf *

# Install nnenum globally:
WORKDIR /usr/local/lib/python3.13/dist-packages
RUN git clone https://github.com/stanleybak/nnenum
WORKDIR /tmp/docker_build

# Configuration to enable UCX/InfiniBand Networking:
RUN echo "* soft memlock unlimited\n* hard memlock unlimited" >> /etc/security/limits.conf
# Patch to enable profiling in charm4py
RUN sed -i -E -e 's/(assert not self.running and charm.runningEntryMethod is None)/\#\1/' -e 's/(assert self.running and charm.runningEntryMethod == self)/\#\1/'  /usr/local/lib/python3.13/dist-packages/charm4py/entry_method.py
RUN sed -i -E -e 's/(\s*self\.options\.local_msg_optim\s*=\s*)False/\1True/' /usr/local/lib/python3.13/dist-packages/charm4py/charm.py && \
    sed -i -E -e 's/(\s*)super\(em\.C(\,\ obj\)\.__init__\(\)\s*\#\s*call\s*Chare\s*class\s*__init__\s*first)/\1Chare\.__init__\(obj\)/' /usr/local/lib/python3.13/dist-packages/charm4py/charm.py
# RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 ubuntu
# RUN echo 'ubuntu:ubuntu' | chpasswd
RUN /usr/bin/ssh-keygen -A

# To get nohup with mpirun... not needed, though https://stackoverflow.com/questions/48296285/nohup-does-not-work-mpirun
#RUN echo "#!/bin/sh\n(\necho \"Date: \`date\`\"\necho \"Command: \$*\"\nPYTHONPATH=/home/${USER_NAME}/tools/FastBATLLNN:/home/${USER_NAME}/tools/FastBATLLNN/HyperplaneRegionEnum:/home/${USER_NAME}/tools/FastBATLLNN/TLLnet:/home/${USER_NAME}/tools/nnenum/src/nnenum nohup \"\$@\"\necho \"Completed: \`date\`\"\necho\n) >>\${LOGFILE:=log.out} 2>&1 &" > /usr/local/bin/bk
# Use bash for this script, since for all shells (login/non-login/interactive/non-interactive) it will use /etc/bash.bashrc
# See: https://unix.stackexchange.com/questions/170493/login-non-login-and-interactive-non-interactive-shells
RUN echo "#!/bin/bash" > /usr/local/bin/bk && \
    echo "(" >>  /usr/local/bin/bk && \
    echo 'echo "Date: `date`"' >>  /usr/local/bin/bk && \
    echo 'echo "Command: $*"' >>  /usr/local/bin/bk && \
    echo 'nohup "$@"' >>  /usr/local/bin/bk && \
    echo 'echo "Completed: `date`"' >>  /usr/local/bin/bk && \
    echo "" >>  /usr/local/bin/bk && \
    echo ") >>\${LOGFILE:=log.out} 2>&1 &" >>  /usr/local/bin/bk && \
    chmod 755 /usr/local/bin/bk && \
    echo "#!/bin/bash\n/usr/sbin/sshd -D &> /root/sshd_log.out &\nif [ -e /etc/ssh/ssh_host_ed25519_key.pub ]\nthen\n echo \"\n****** SSH host key ******\"\ncat /etc/ssh/ssh_host_ed25519_key.pub\necho \"**************************\n\"\nsudo -u \$1 cp /etc/ssh/ssh_host_ed25519_key.pub /home/\$1/results\nfi\nif [ -e /home/\$1/results/authorized_keys ] && [ -d /home/\$1/.ssh ]\nthen\ncp /home/\$1/results/authorized_keys /home/\$1/.ssh\nchmod 600 /home/\$1/.ssh/authorized_keys && chown \$1:\$1 /home/\$1/.ssh/authorized_keys && rm /home/\$1/results/authorized_keys\nfi\nif [ \"\$2\" = \"-d\" ]; then\nwait -n\nelse\nsudo -u \$1 /bin/bash\nkillall sshd\nfi" > /usr/local/bin/startup.sh && \
    chmod 755 /usr/local/bin/startup.sh && \
    sed -i "2i export PATH=/usr/local/bin:/usr/mpi/gcc/openmpi-4.1.7rc1/bin:\$PATH" /etc/bash.bashrc && \
    sed -i "3i export LD_LIBRARY_PATH=/usr/local/lib" /etc/bash.bashrc && \
    sed -i "4i export COIN_INSTALL_DIR=/usr/local" /etc/bash.bashrc

#RUN echo "export PATH=/usr/local/bin:\$PATH\nexport LD_LIBRARY_PATH=/usr/local/lib\nexport COIN_INSTALL_DIR=/usr/local\nexport PATH=\"/usr/mpi/gcc/openmpi-4.1.2a1/bin:$PATH\"" >> /etc/bash.bashrc

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

