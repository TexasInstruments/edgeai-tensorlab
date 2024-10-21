# base image
ARG REPO_LOCATION=""
FROM ${REPO_LOCATION}ubuntu:22.04

# user
ARG USER_NAME=edgeai
ENV HOME_DIR=/opt
# ARG USER_ID=1000
# ARG USER_GID=$USER_ID

# build settings
ARG PROJECT_NAME="modelmaker"
ARG DEBIAN_FRONTEND=noninteractive
# avoid warnings/errors while installing pillow-simd
ARG LC_ALL=C.UTF-8

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

# ##################################################################
# Method 1: these lines would have created a user and switched to it
# add user, inspired by: https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
# RUN groupadd --gid $USER_GID $USER_NAME && \
#     useradd --uid $USER_ID --gid $USER_GID --create-home $USER_NAME && \
#     chown -R ${USER_ID}:${USER_GID} ${HOME_DIR}
#
# # not having this may cause sudo commands in our setup files to issue an error
# # but any sudo intallation required may be finished above - so it should be okay.
# RUN echo ${USER_NAME} ALL=\(root\) NOPASSWD:ALL >> /etc/sudoers.d/${USER_NAME} && chmod 400 /etc/sudoers.d/${USER_NAME}
#
# # switch user, workdir, default permissions
# USER $USER_NAME
# RUN echo "umask u=rwx,g=rx,o=rx" >> ${HOME_DIR}/.bashrc
# ##################################################################
# Method 2: continue as root. during docker run, --user argument can be used to map to the host user
# also a chmod -R ugo+rwx ${HOME_DIR} is added at the end of this file to enable such mapped users to have access.
# that permission can be later restricted to only folders where a write is required.
RUN mkdir -p ${HOME_DIR}
# ##################################################################

# change working directory
WORKDIR ${HOME_DIR}

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

# Note 1: this can be reversed later as a last step after all installations are done.
# enabling write permissions for all users, if support is needed for users passed via --user argument of docker run
RUN chmod -R ugo+rwx ${HOME_DIR}

# Note 2: after all installations are done,
# it is possible to remove the file /etc/sudoers.d/${USER_NAME} (if that was used above)

# Note 3: in the container, .bashrc or .pyenvrc may need to be sourced to activate the Python environment
# if it is not being activated automatically.
# source ${HOME_DIR}/.bashrc

