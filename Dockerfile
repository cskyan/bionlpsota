FROM nvidia/cuda:10.2-devel-ubuntu18.04

# Configure build tools
ENV CMAKE_MAJOR_VERSION=3.20
ENV CMAKE_VERSION=3.20.0

# Configure package versions of TensorFlow, PyTorch, MXNet, CUDA, cuDNN and NCCL
ENV TENSORFLOW_VERSION=2.6.0
ENV PYTORCH_MAJOR_VERSION=1.8
ENV PYTORCH_VERSION=1.8.2+cu102
ENV TORCHVISION_VERSION=0.9.2+cu102
ENV TORCHAUDIO_VERSION=0.8.2
ENV CUDA_VERSION=10.2
ENV CUDNN_MAJOR_VERSION=8
ENV CUDNN_VERSION=8.3.1.22-1+cuda10.2
ENV NCLL_MAJOR_VERSION=2
ENV NCCL_VERSION=2.11.4-1+cuda10.2
ENV MXNET_VERSION=1.8.0

# Configure Python version 2.7 or 3.6
ARG python=3.6
ENV PYTHON_VERSION=${python}

# Configure Java version 8 or 11
ENV JAVA_VERSION=8

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Get rid of the debconf messages
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Update repository
 RUN apt-get update

# Configure timezone
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

# Install and configure locales
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Add repository for legacy packages
RUN apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get update

# Install the essential packages
RUN apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        apt-utils \
        build-essential \
        ninja-build \
        g++-4.8 \
        cmake \
        git \
        curl \
        vim \
        wget \
        ccache \
        ca-certificates \
        libcudnn${CUDNN_MAJOR_VERSION}=${CUDNN_VERSION} \
        libcudnn${CUDNN_MAJOR_VERSION}-dev=${CUDNN_VERSION} \
        libnccl${NCLL_MAJOR_VERSION}=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        libopenblas-dev \
        liblapack-dev \
        libopencv-dev

# Latest build tools
RUN cd /tmp && wget https://cmake.org/files/v${CMAKE_MAJOR_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && tar -zxf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && mv cmake-${CMAKE_VERSION}-linux-x86_64 /usr/local/cmake

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

# Install TensorFlow, Keras and PyTorch
RUN pip install future typing
RUN pip install --no-cache-dir numpy \
        tensorflow-gpu==${TENSORFLOW_VERSION} \
        keras \
        h5py
RUN pip install --no-cache-dir torch===${PYTORCH_VERSION} torchvision===${TORCHVISION_VERSION} torchaudio===${TORCHAUDIO_VERSION} -f https://download.pytorch.org/whl/lts/${PYTORCH_MAJOR_VERSION}/torch_lts.html

# Install MXNet
RUN git clone --depth 1 --branch ${MXNET_VERSION} https://github.com/apache/incubator-mxnet /tmp/mxnet && \
    cd /tmp/mxnet && git submodule update --init --recursive && \
    mkdir -p build && cd build && \
    /usr/local/cmake/bin/cmake -DUSE_BLAS=open -DUSE_LAPACK=ON -DUSE_MKL_IF_AVAILABLE=OFF -DUSE_MKLDNN=OFF -DUSE_CUDA=ON -DMXNET_CUDA_ARCH=5.2\;5.3\;6.0\;6.2\;7.0\;7.2\;7.5 -DUSE_CUDNN=ON -DUSE_NCCL=ON -DNCCL_LAUNCH_MODE=PARALLEL -DUSE_OPENCV=ON -DUSE_OPENMP=ON -DCMAKE_BUILD_TYPE=Release -G Ninja .. && \
    ninja -j$(nproc) && \
    ninja install && \
    cd ../python && pip install -e . && \
    rm -rf /tmp/mxnet
#RUN pip install mxnet-cu110==${MXNET_VERSION}

# Install apex
RUN git clone https://github.com/cskyan/apex /usr/local/apex && \
    cd /usr/local/apex && \
    git checkout tags/latest && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Configure OpenMPI version
ENV OPENMPI_MAJOR_VERSION=4.1
ENV OPENMPI_VERSION=4.1.1

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v${OPENMPI_MAJOR_VERSION}/downloads/openmpi-${OPENMPI_VERSION}.tar.gz -q && \
    tar zxf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install Horovod
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_ALLOW_MIXED_GPU_IMPL=0 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=0 \
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

# Configure PyLucene version
ENV PYLUCENE_VERSION=8.8.1

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

# Configure Faiss version
ENV FAISS_VERSION=1.7.1

# Install Faiss
RUN apt-get install -y swig
RUN git clone https://github.com/facebookresearch/faiss.git /tmp/faiss && \
    cd /tmp/faiss && \
    git checkout v${FAISS_VERSION} && \
    /usr/local/cmake/bin/cmake -B build . -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DCUDAToolkit_ROOT=/usr/local/cuda && make -C build -j $(nproc) faiss && make -C build -j $(nproc) swigfaiss && \
    cd build/faiss/python && python setup.py install && \
    rm -rf /tmp/faiss

# Configure Spacy, SciSpacy and AllenNLP versions
ENV SPACY_VERSION=3.0.6
ENV SCISPACY_VERSION=0.4.0
ENV ALLENNLP_VERSION=2.7.0

# Install NLP packages
RUN pip install textdistance tqdm openpyxl pandas scikit-learn && \
    pip install nltk ftfy
RUN pip install spacy==${SPACY_VERSION} scispacy==${SCISPACY_VERSION}
RUN pip install allennlp==${ALLENNLP_VERSION}

# Install NLP models
RUN python -c "import nltk; nltk.download('popular')"
RUN python -m spacy download en_core_web_sm
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v${SCISPACY_VERSION}/en_core_sci_md-${SCISPACY_VERSION}.tar.gz
RUN pip install pytorch_pretrained_bert transformers

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
