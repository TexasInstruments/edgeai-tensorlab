# base image
ARG REPO_LOCATION=""
FROM ${REPO_LOCATION}ubuntu:18.04

#user
ARG USER_NAME=edgeai
ARG USER_ID=1000
ARG USER_GID=$USER_ID
ENV HOME_DIR=/home/$USER_NAME

# proxy, path
ARG PROXY_LOCATION=""
ARG PROJECT_NAME="modelmaker"
ENV DEBIAN_FRONTEND=noninteractive
ENV http_proxy=${PROXY_LOCATION}
ENV https_proxy=${PROXY_LOCATION}
ENV no_proxy=ti.com

# paths
ENV APPS_PATH=${HOME_DIR}/apps
ENV DOWNLOADS_PATH=${HOME_DIR}/downloads
ENV CODE_PATH=${HOME_DIR}/code
ENV CONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
ENV CONDA_URL=https://repo.anaconda.com/miniconda/${CONDA_INSTALLER}
ENV CONDA_PATH=${APPS_PATH}/conda
ENV CONDA_BIN=${CONDA_PATH}/bin
ENV PATH=:${CONDA_BIN}:${PATH}

# baseline
RUN echo 'apt update...' && \
    if [ ! -z $PROXY_LOCATION ]; then echo "Acquire::http::proxy \"${PROXY_LOCATION}\";" > /etc/apt/apt.conf; fi && \
    if [ ! -z $PROXY_LOCATION ]; then echo "Acquire::https::proxy \"${PROXY_LOCATION}\";" >> /etc/apt/apt.conf; fi && \
    apt update && \
    apt install -y sudo git iputils-ping wget cmake build-essential libgtk2.0

# additional packages
RUN apt install -y libjpeg-dev zlib1g-dev graphviz graphviz-dev

# add user, inspired by: https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
RUN groupadd --gid $USER_GID $USER_NAME && \
    useradd --uid $USER_ID --gid $USER_GID --create-home $USER_NAME && \
    echo $USER_NAME ALL=\(root\) NOPASSWD:ALL >> /etc/sudoers.d/$USER_NAME && \
    chmod 400 /etc/sudoers.d/$USER_NAME

# Change working directory permission to non-root user
RUN chown -R ${USER_ID}:${USER_GID} ${HOME_DIR}

# switch user, workdir, default permissions
USER $USER_NAME
WORKDIR ${HOME_DIR}
RUN echo "umask u=rwx,g=rwx,o=rx" >> ${HOME_DIR}/.bashrc

# conda install
RUN mkdir ${DOWNLOADS_PATH} && \
    if [ ! -f ${CONDA_INSTALLER} ]; then wget -q --show-progress --progress=bar:force:noscroll ${CONDA_URL} -O ${DOWNLOADS_PATH}/${CONDA_INSTALLER}; fi && \
    chmod +x ${DOWNLOADS_PATH}/${CONDA_INSTALLER} && \
    ${DOWNLOADS_PATH}/${CONDA_INSTALLER} -b -p ${CONDA_PATH} && \
    echo ". ${CONDA_PATH}/etc/profile.d/conda.sh" >> ${HOME_DIR}/.bashrc

# conda python
RUN conda create -y -n py36 python=3.6 && \
    echo "conda activate py36" >> ${HOME_DIR}/.bashrc
