# in a multi-stage build, args get reset after a from statement
# declare ARG without default to retrieve it
# but ENV variables do no have this problem
ARG DOCKER_IMAGE_BASE="ubuntu:22.04"
ARG DOCKER_REPO_LOCATION


# build settings
ARG PROJECT_NAME="edgeai-modelmaker"

FROM ${DOCKER_IMAGE_BASE}

ENV HOME_DIR=/opt
ENV HOME ${HOME_DIR}

ENV WORK_DIR ${HOME_DIR}/code
ENV DATA_DIR ${HOME_DIR}/data
ENV PYTHONPATH=":${HOME_DIR}/edgeai-modelmaker:$PYTHONPATH"

#setup file cpu or gpu
ARG SETUP_FIlE="setup_gpu.sh"

LABEL TI_IMAGE_SOURCE="https://bitbucket.itg.ti.com/projects/EDGEAI-ALGO/repos/edgeai-modelmaker/browse"

RUN mkdir -p ${WORK_DIR}
RUN mkdir -p ${DATA_DIR}

# user
ARG USER_NAME=edgeai
# ARG USER_ID=1000
# ARG USER_GID=$USER_ID

# build settings
ARG DEBIAN_FRONTEND=noninteractive
# avoid warnings/errors while installing pillow-simd
ARG LC_ALL=C.UTF-8

# print
RUN echo "DOCKER_IMAGE_BASE="${DOCKER_IMAGE_BASE} && \
    echo "DOCKER_REPO_LOCATION="${DOCKER_REPO_LOCATION} && \
    echo "HOME_DIR="${HOME_DIR} && \
    echo "PWD=$(pwd)" && \
    echo $(ls -al)

#####################################################################################
# proxy
ARG http_proxy=""
ARG https_proxy=""
ARG no_proxy=""

# proxy for apt
# /etc/apt/apt.conf will have to be updated if apt install is needed during docker run
# and if the proxy used is different from the proxy being used here during docker build
RUN if [ ! -z $http_proxy ]; then echo "Acquire::http::proxy \"${http_proxy}\";" > /etc/apt/apt.conf; fi && \
    if [ ! -z $https_proxy ]; then echo "Acquire::https::proxy \"${https_proxy}\";" >> /etc/apt/apt.conf; fi

# baseline
RUN apt update && apt install -y sudo git curl iputils-ping wget cmake build-essential libgtk-3-0 apt-utils \
    libbz2-dev xz-utils libreadline-dev libncurses5-dev libssl-dev libsqlite3-dev liblzma-dev

# additional packages
RUN apt install -y libffi-dev libjpeg-dev zlib1g-dev graphviz graphviz-dev protobuf-compiler

# pyenv install
ENV PYENV_ROOT ${HOME_DIR}/.pyenv
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

# we define PIP_CACHE_DIR here to point to a writable location
# to enable container user to be different by using the docker run option --user
RUN echo ' ' >> ${HOME_DIR}/.pyenvrc && \
    echo "export PIP_CACHE_DIR=${HOME_DIR}/.cache/pip" >> ${HOME_DIR}/.pyenvrc && mkdir -p ${HOME_DIR}/.cache/pip

# set paths and activate env
# .bashrc is not sourced in non interactive mode - so write it to another file
RUN echo ' ' >> ${HOME_DIR}/.pyenvrc && \
    echo '# pyenv settings' >> ${HOME_DIR}/.pyenvrc && \
    echo "command -v pyenv >/dev/null || export PATH=:$PYENV_ROOT/bin:$PATH" >> ${HOME_DIR}/.pyenvrc && \
    echo 'eval "$(pyenv init -)"' >> ${HOME_DIR}/.pyenvrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> ${HOME_DIR}/.pyenvrc && \
    echo ' ' >> ${HOME_DIR}/.pyenvrc

# .profile is needed for non-interactive shell spawns
RUN echo ' ' >> ${HOME_DIR}/.profile && \
    echo "command -v pyenv >/dev/null || source ${HOME_DIR}/.pyenvrc" >> ${HOME_DIR}/.profile && \
    echo ' ' >> ${HOME_DIR}/.profile

# .bashrc is needed for interactive shell spawns
RUN echo ' ' >> ${HOME_DIR}/.bashrc && \
    echo "command -v pyenv >/dev/null || source ${HOME_DIR}/.pyenvrc" >> ${HOME_DIR}/.bashrc && \
    echo ' ' >> ${HOME_DIR}/.bashrc

# pyenv works with bash, so change shell to bash
SHELL ["/bin/bash", "--login", "-c"]

# create virtualenv
RUN source ${HOME_DIR}/.bashrc && \
    pyenv install 3.10 && \
    pyenv virtualenv 3.10 py310 && \
    pyenv rehash && \
    echo 'pyenv activate py310' >> ${HOME_DIR}/.pyenvrc && \
    echo ' ' >> ${HOME_DIR}/.pyenvrc

# update pip setuptools
RUN source ${HOME_DIR}/.bashrc && \
    pip3 install --no-input --upgrade pip==24.2 setuptools==73.0.0 && \
    pip3 install --no-input cython wheel numpy==1.23.0


# enabling write permissions for all users, if support is needed for users passed via --user argument of docker run
RUN chmod -R ugo+rwx ${HOME_DIR}

# workdir
WORKDIR ${WORK_DIR}/${PROJECT_NAME}

# Change working directory permission to non-root user
RUN source ${HOME_DIR}/.bashrc

# Add all files to docker image
# ADD . ${WORK_DIR}

# RUN cd ${WORK_DIR}/${PROJECT_NAME} && \
#    ./${SETUP_FIlE}