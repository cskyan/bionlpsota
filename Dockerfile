FROM nvidia/cuda:10.1-devel-ubuntu18.04

# Configure package versions of TensorFlow, PyTorch, MXNet, CUDA, cuDNN and NCCL
ENV TENSORFLOW_VERSION=2.0.0
ENV PYTORCH_VERSION=1.4.0
ENV TORCHVISION_VERSION=0.5.0
ENV CUDNN_VERSION=7.6.5.32-1+cuda10.1
ENV NCCL_VERSION=2.5.6-1+cuda10.1
ENV MXNET_VERSION=1.5.1

# Configure Python version 2.7 or 3.6
ARG python=3.6
ENV PYTHON_VERSION=${python}

# Configure Java version 8 or 11
ENV JAVA_VERSION=8

# Configure PyLucene version
ENV PYLUCENE_VERSION=8.1.1

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Install the essential packages
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        apt-utils \
        build-essential \
        cmake \
        g++-4.8 \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn7=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers

# Get rid of the debconf messages
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install and configure locales
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Install JDK
RUN apt-get install -y openjdk-${JAVA_VERSION}-jdk ant
ENV JAVA_HOME=/usr/lib/jvm/java-${JAVA_VERSION}-openjdk-amd64

# Install Python
RUN if [[ "${PYTHON_VERSION}" == "3.6" ]]; then \
        apt-get install -y python${PYTHON_VERSION}-distutils; \
    fi
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    pip install --upgrade pip wheel

# Install TensorFlow, Keras, PyTorch and MXNet
RUN pip install future typing
RUN pip install numpy \
        tensorflow-gpu==${TENSORFLOW_VERSION} \
        keras \
        h5py
RUN pip install torch===${PYTORCH_VERSION} torchvision===${TORCHVISION_VERSION} -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install mxnet-cu101==${MXNET_VERSION}

# Install apex
RUN git clone https://github.com/NVIDIA/apex /usr/local/apex && \
    cd /usr/local/apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    cd /

# Install PyLucene, pysolr and ijson
RUN mkdir /tmp/pylucene && \
    cd /tmp/pylucene && \
    wget https://dist.apache.org/repos/dist/release/lucene/pylucene/pylucene-${PYLUCENE_VERSION}-src.tar.gz -q && \
    tar zxf pylucene-${PYLUCENE_VERSION}-src.tar.gz && \
    cd pylucene-${PYLUCENE_VERSION}/jcc && \
    NO_SHARED=1 JCC_JDK=${JAVA_HOME} python setup.py install && \
    cd .. && \
    make all install JCC='python -m jcc' ANT=ant PYTHON=python NUM_FILES=8 && \
    ldconfig && \
    rm -rf /tmp/pylucene
RUN pip install pysolr ijson

# Install NLP packages
RUN pip install tqdm openpyxl pandas scikit-learn && \
    pip install nltk ftfy spacy
RUN pip install scispacy
RUN pip install allennlp

# Install NLP models
RUN python -c "import nltk; nltk.download('popular')"
RUN python -m spacy download en_core_web_sm
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz
RUN pip install pytorch_pretrained_bert

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.1.tar.gz -q && \
    tar zxf openmpi-4.0.1.tar.gz && \
    cd openmpi-4.0.1 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install Horovod
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_GPU_ALLGATHER=MPI HOROVOD_ALLOW_MIXED_GPU_IMPL=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=1 \
    CFLAGS="-O2 -mavx -mfma" \
         pip install --no-cache-dir horovod && \
    ldconfig

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Download source code
RUN mkdir /source && \
    cd /source && \
    git clone https://github.com/cskyan/bionlp.git && \
    git clone https://github.com/cskyan/bionlpsota.git && \
    rm -rf bionlp/.git bionlpsota/.git
ENV PYTHONPATH=/source

# Prepare workspace
RUN mkdir /workspace
WORKDIR "/workspace"
