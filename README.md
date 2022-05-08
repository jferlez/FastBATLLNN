# FastBATLLNN: Fast Box Analysis of Two-Level Lattice Neural Networks

FastBATLLNN is a fast verifier of box-like (hyper-rectangle) output properties for Two-Level Lattice (TLL) Neural Networks (NN). In particular, FastBATLLNN can formally verify whether the output of a TLL NN is confined to a specified hyper-rectangle whenever its inputs are confined to a specified closed, convex polytope.

This work appeared in [HSCC 2022](https://hscc.acm.org/2022/). Please refer to/cite the following publication:

>_Fast BATLLNN: Fast Box Analysis of Two-Level Lattice Neural Networks._  
>James Ferlez, Haitham Khedr and Yasser Shoukry. HSCC '22: 25th ACM International Conference on Hybrid Systems: Computation and Control, May 2022. Article No.: 23. Pages 1–11. https://doi.org/10.1145/3501710.3519533

_Please contact [jferlez@uci.edu](mailto:jferlez@uci.edu) with any questions/bug reports._

## 1) Prerequisites

FastBATLLNN is written in Python, and it depends on the following packages/libraries (and their dependencies):

* [charm4py](https://charm4py.readthedocs.io/en/latest/) (Python/Library)
* [glpk](https://www.gnu.org/software/glpk/) (Library)
* [cylp](https://github.com/coin-or/CyLP) (Python) / [Clp](https://github.com/coin-or/Clp) (Library)
* [cvxopt](https://cvxopt.org) (Python)
* [numba](https://numba.pydata.org) (Python)
* [pycddlib](https://pycddlib.readthedocs.io/en/latest/) (Python/Library)

These dependencies can be burdensome to install, so we have provided facilities for running FastBATLLNN in a [Docker](https://docker.com) container.

> **NOTE: A Docker container is the preferred means of running FastBATLLNN; the rest of this document assumes this run environment.**

### (Docker) Prerequisites:
1. A Linux or MacOS host (Windows hosts _may_ work via WSL, but this is untested.)
2. A recent version of [Docker](https://docker.com)
3. Approximately 25Gb of free disk space; however, the final Docker image only requires ~13Gb (MacOS users should allocate >=30Gb to the Docker VM to be safe)
4. The ability to run a Docker container with the [`--priviledged` switch](https://docs.docker.com/engine/reference/run/#runtime-privilege-and-linux-capabilities).


## 2) Quick Start

```Bash
LOCATION=/path/to/someplace/convenient
cd "$LOCATION"
git clone --recursive https://github.com/jferlez/FastBATLLNN # --recurisve is optional if using Docker
cd FastBATLLNN
./dockerbuild.sh
./dockerrun.sh
```
This should place you at a Bash shell inside a container with FastBATLLNN installed. If these scripts execute successfully, then you can skip to **4) Running FastBATLLNN**.

> **WARNING:** if you exit the container's Bash shell, then the container will stop. See Section **3)** for information about how to restart the container, and regain access.

> **NOTE:** The Docker images/container have a number of features that are documented in Section **3)**. **I ADVISE YOU NOT TO SKIP THAT SECTION!**

## 3) Creating the Docker Images/Container

### (i) Creating the Docker Images

The first step is to obtain the FastBATLLNN code by cloning this Git repository:

```Bash
LOCATION=/path/to/someplace/convenient
cd "$LOCATION"
git clone --recursive https://github.com/jferlez/FastBATLLNN # --recurisve is optional if using Docker
```

> **NOTE:** `$LOCATION` will refer to the FastBATLLNN install location henceforth.

Now, FastBATLLNN comes with the Bash script `dockerbuild.sh`, which should automatically build all of the necessary Docker images:

```Bash
cd FastBATLLNN
./dockerbuild.sh
```

>**WARNING:** The first run of `dockerbuild.sh` may take **AN HOUR OR MORE** to download and compile all of the dependencies.

`dockerbuild.sh` should create two docker images that appear in the output of `docker image ls` as follows:

    REPOSITORY                 TAG                               IMAGE ID       CREATED          SIZE
    fastbatllnn-run            USER                              123456789abc   31 minutes ago   12.5GB
    fastbatllnn                deps                              def123456789   32 minutes ago   12.5GB

Where `USER` is the **host** log-in name of the user who ran `dockerbuild.sh`; it will be automatically detected by `dockerbuild.sh`.

> **NOTE:** Henceforth `USER` will refer to the login name of the current host user.

These images are described as follows:

1. `fastbatllnn:deps` contains all of the runtime dependencies of FastBATLLNN; it is a separate image to facilitate re-use (originally I intended to host this image on DockerHub, but decided against it due to licensing issues.)
2. `fastbatllnn-run:USER` is the image that adds the FastBATLLNN code by cloning this repository; it is the one that will ultimately be run in a container. **It is configured to have a user with same user name, user id and group id as the creating host user** (placeholder `USER`).

> **WARNING:** `fastbatllnn-run:USER` is derived from `fastbatllnn:deps` with a `FROM` directive in its `Dockerfile`. Thus, **DO NOT DELETE `fastbatllnn:deps`**! Because of the way Docker works, its contents are not duplicated in `fastbatllnn-run:USER` anyway.

> **NOTE:** A subsequent call to `dockerbuild.sh` will use the **cached** version of `fastbatllnn:deps` but will **rebuild** `fastbatllnn-run:USER` from scratch -- _i.e., re-clone this repository_; `fastbatllnn:deps` is the only image that requires significant time to create.
### (ii) Running a Docker Container

Once a user has created the above Docker images (only `fastbatllnn-run:USER` is user specific), a suitable container can be started using the command:

```Bash
./dockerrun.sh
```

`dockerrun.sh` infers the current host user name (placeholder `USER`), and launches a relevant container in one of two ways:
1. If _there **EXISTS NO** container_ derived from `fastbatllnn-run:USER`, then `dockerrun.sh` starts a new container from that image, using the appropriate options; or
2. If _there **EXISTS** a container_ derived from `fastbatllnn-run:USER`, then `dockerrun.sh` simply attempts to (re-)start that container.

This should allow you to stop a container (say by restarting your computer), without having to figure out which container to restart.

You can interact with the resultant container in three ways:

1. When the container is first started by `dockerrun.sh`, you will be placed in a Bash shell in the container (i.e. option `-it` for [`docker run`](https://docs.docker.com/engine/reference/run/));
2. The container starts an SSH daemon listening on _localhost_, port 3000; it also attempts to copy the current user's **public** ssh keys for password-less login (i.e. it adds these keys to `./ssh/authorized_keys` in the container);
3. The host directory `$LOCATION/container_results` is bind-mounted to `/home/USER/results` in the container (the host directory will be created if it doesn't exist).

> **WARNING:** if you exit the container's Bash shell, then the container will stop. Restarting the container with `dockerrun.sh` (see above) will not return you to a Bash shell, but it will restart the SSH daemon: you will have to interact with the restarted container via SSH.

SSH is the best way to run code in the container:

* It allows the container to run in the background without occupying a terminal on the host;
* It allows SFTP to be used to transfer files to/from the container; and
* (Most importantly) It allows the use of [VS Code](https://code.visualstudio.com) to edit/run code directly in the container via its [Remote-SSH](https://code.visualstudio.com/docs/remote/ssh) functionality.

If you see either of the following lines output by `dockerrun.sh`:

    Copying public key from ~/.ssh/id_rsa.pub to container authorized_keys
    Copying public keys from ~/.ssh/authorized_keys to container authorized_keys

then you can log-in to the container as `USER` (i.e. the current host user) from the host or any computers that can log-in _to the host computer_, respectively.

That is you should be able to log-in to the container with the following command:

```Bash
ssh -p 3000 USER@localhost
```
with no password required. If the log-in is successful, you will be asked to accept the host key of the container using a prompt like:

    The authenticity of host '[localhost]:3000 ([127.0.0.1]:3000)' can't be established.
    ECDSA key fingerprint is SHA256:qwertyuiopasdfghjkl.
    Are you sure you want to continue connecting (yes/no/[fingerprint])?

This host key is automatically placed in `$LOCATION/container_results/ssh_host_rsa_key.pub` so you can verify that you are confirming the correct one.

> **WARNING:** if at any time in the past you logged in to the host via `localhost`, then you will be warned about a potential man-in-the-middle attack. In this case, simply disconnect, and remove or comment out the **host's** public key from `~/.ssh/known_hosts` _on the host_, and try reconnecting.

> **WARNING:** for security reasons, password log-in to the container is disabled. So you must have keys in either `~/.ssh/id_rsa.pub` (to log-in to the container _from the host_) or `~/.ssh/authorized_keys` (to log-in to the container from a computer that can log-in _to_ the host).

> **NOTE:** The container is configured with a user that has the same user name, user id and group id as the creating user on the host (`USER` placeholder). The password for that user is just the user name, should `sudo` be required in the container (i.e. `USER` for our placeholder).

> **NOTE:** The container has [`tmux`](https://github.com/tmux/tmux/wiki) and [`screen`](https://www.gnu.org/software/screen/) pre-installed to facilitate running tasks that persist after disconnecting an SSH session.

## 4) Running FastBATLLNN

This section assumes that you have successfully reached a prompt inside a `fastbatllnn-run:USER` container. For example:

```Bash
USER@0123456789ab:~$
```

where is a container id for a `fastbatllnn-run:USER` derived container. Please go back to Section **3)** if this is not the case.

> **WARNING:** in this section, we will assume that the host has at least 4 physical (i.e. non-hyperthreading) cores. See below for information about how to proceed if this is not the case.

FastBATLLNN is already configured to run inside the container, and there is a simple example to make sure everything is working. To test this, execute the following in the container (i.e. at the prompt above):

```Bash
cd ~/tools/FastBATLLNN
charmrun +p4 example.py
```

This command should produce the following output (abbreviated here):

    Running as 4 OS processes: /usr/bin/python3.9 example.py
    charmrun> /usr/bin/setarch x86_64 -R mpirun -np 4 /usr/bin/python3.9 example.py
    Charm++> Running on MPI library: Open MPI v4.0.3, package: Debian OpenMPI, ident: 4.0.3, repo rev: v4.0.3, Mar 03, 2020 (MPI standard: 3.1)
    Charm++> Level of thread support used: -1 (desired: 0)
    Charm++> Running in non-SMP mode: 4 processes (PEs)
    Converse/Charm++ Commit ID: v7.1.0-devel-154-g4fd8c4b81
    Isomalloc> Synchronized global address space.
    CharmLB> Load balancer assumes all CPUs are same.
    Charm4py> Running Charm4py version 1.0 on Python 3.9.5 (CPython). Using 'cython' interface to access Charm++
    Charm++> Running on 1 hosts (2 sockets x 12 cores x 2 PUs = 48-way SMP)
    Charm++> cpu topology info is gathered in 0.000 seconds.
    CharmLB> Load balancing instrumentation for communication is off.
    2022-05-07 20:21:50.290119: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/lib
    2022-05-07 20:21:50.290181: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)


    Max of Output Samples: 4.524217171074326
    Min of Output Samples: 0.43370844553525956


    ----------------- VERIFYING LOWER BOUND:  -----------------
    Total LPs used: defaultdict(<class 'int'>, {'LPSolverCount': 35, 'xferTime': 0.001024484634399414})
    Checker returned value: True
    Computed a (partial) poset of size: 5
    TLL always >= 0.299 on constraints? True
    -----------------------------------------------------------


    ----------------- VERIFYING UPPER BOUND:  -----------------
    Upper Bound verifiction used 8 total LPs.
    TLL always <= 4.51 on constraints? False
    -----------------------------------------------------------


    --------------- FINDING TIGHT LOWER BOUND:  ---------------
    Total LPs used: defaultdict(<class 'int'>, {'LPSolverCount': 35, 'xferTime': 0.0008916854858398438})
    Checker returned value: True
    Computed a (partial) poset of size: 5
    Iteration 100: -0.135 is a VALID lower bound!
    Total LPs used: defaultdict(<class 'int'>, {'LPSolverCount': 28, 'xferTime': 0.0009856224060058594})
    Checker returned value: False
    Computed a (partial) poset of size: 4
    .
    .
    .
    Checker returned value: True
    Computed a (partial) poset of size: 5
    Iteration 89: 0.2995703125 is a VALID lower bound!
    **********    verifyLB on LB processing times:   **********
    Total time required to initialize the new lb problem: 0.010710716247558594
    Total time required for region check workers to initialize: 0
    Total time required for (partial) poset calculation: 0.21567153930664062
    Iterations used: 11
    ***********************************************************


    ------------------  FOUND LOWER BOUND:  -------------------
    0.2995703125
    Total time elapsed: 0.20470857620239258 (sec)
    Minimum of samples: 0.43370844553525956
    -----------------------------------------------------------


    --------------- FINDING TIGHT UPPER BOUND:  ---------------
    Upper Bound verifiction used 12 total LPs.
    Iteration 100: -135 is a VALID lower bound!
    .
    .
    .
    Upper Bound verifiction used 176 total LPs.
    Iteration 76: 4.538595911107734 is an INVALID lower bound!
    **********    verifyUB on UB processing times:   **********
    Iterations used: 24
    Total number of LPs used for Upper Bound verification: 176
    ***********************************************************


    ------------------  FOUND UPPER BOUND:  -------------------
    4.538595911107734
    Total time elapsed: 0.022965192794799805 (sec)
    Maximum of samples: 4.524217171074326
    -----------------------------------------------------------

That's it! You have verified a simple TLL NN!

> **NOTE:** The file `example.py` is (somewhat) commented to further document how to use FastBATLLNN. More documentation will be published as time permits.


<br>
<br>
<br>

## If your host has fewer than 4 physical cores:

If your host has fewer than 4 physical (non-hyperthreading) cores, then you must make two modifications in order to run `example.py`.

First, edit line 94 of `example.py` (`vim` and `emacs` are installed in the container, or you can used VS Code or SFTP to alter this file):

```Python
pes = {'poset':[(0,4,1)],'hash':[(0,4,1)]}
```

Replace both `4`'s with the total number of physical cores on your host, then save the file.

Second, change the `4` in the command:

```Bash
cd ~/tools/FastBATLLNN
charmrun +p4 example.py
```

to the total number of physical cores.