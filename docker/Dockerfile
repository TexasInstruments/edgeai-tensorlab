# base image
ARG REPO_LOCATION=""
FROM ${REPO_LOCATION}ubuntu:18.04

# user
# ARG USER_ID=1000
# ARG USER_GID=$USER_ID
# ARG USER_NAME=edgeai
ARG HOME_DIR=/home/edgeai

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
RUN apt update && apt install -y sudo git iputils-ping wget cmake build-essential libgtk2.0 apt-utils

# additional packages
RUN apt install -y curl libjpeg-dev zlib1g-dev graphviz graphviz-dev libbz2-dev \
                   build-essential xz-utils libreadline-dev libncurses5-dev libssl-dev \
                   libsqlite3-dev liblzma-dev

# ##################################################################
# # Method 1: these lines would have created a user and switched to it
# # add user, inspired by: https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
# RUN groupadd --gid $USER_GID $USER_NAME && \
#     useradd --uid $USER_ID --gid $USER_GID --create-home $USER_NAME && \
#     chown -R ${USER_ID}:${USER_GID} ${HOME_DIR}
#
# # is this required? - any sudo intsallation required may be finished above.
# # but not having this may cause sudo commands in our setup files to error out.
# # echo $USER_NAME ALL=\(root\) NOPASSWD:ALL >> /etc/sudoers.d/$USER_NAME && chmod 400 /etc/sudoers.d/$USER_NAME
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
    echo 'command -v pyenv >/dev/null || . ${HOME_DIR}/.pyenvrc' >> ${HOME_DIR}/.profile && \
    echo ' ' >> ${HOME_DIR}/.profile

# .bashrc is needed for interactive shell spawns
RUN echo ' ' >> ${HOME_DIR}/.bashrc && \
    echo 'command -v pyenv >/dev/null || . ${HOME_DIR}/.pyenvrc' >> ${HOME_DIR}/.bashrc && \
    echo ' ' >> ${HOME_DIR}/.bashrc

# pyenv works with bash, so change shell to bash
SHELL ["/bin/bash", "--login", "-c"]

# create virtualenv
RUN source ${HOME_DIR}/.bashrc && \
    pyenv install 3.6 && \
    pyenv virtualenv 3.6 py36 && \
    pyenv rehash && \
    echo 'pyenv activate py36' >> ${HOME_DIR}/.pyenvrc && \
    echo ' ' >> ${HOME_DIR}/.pyenvrc

# update pip setuptools
RUN source ${HOME_DIR}/.bashrc && \
    pip install --no-input --upgrade pip setuptools && \
    pip install --no-input --upgrade wheel cython numpy

# enable write permissions for all users, if support is needed for users passed via --user argument of docker run
# this can be reversed later as a last step after all installations are done.
RUN chmod -R ugo+rwx ${HOME_DIR}
