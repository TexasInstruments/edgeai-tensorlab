# base image
ARG REPO_LOCATION=""
FROM ${REPO_LOCATION}ubuntu:18.04

# proxy, path
ARG PROXY_LOCATION=""
ARG PROJECT_NAME="modelmaker"
ENV CONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
ENV http_proxy=${PROXY_LOCATION}
ENV https_proxy=${PROXY_LOCATION}
ENV no_proxy=ti.com
ENV CONDA_URL=https://repo.anaconda.com/miniconda/${CONDA_INSTALLER}
ENV CONDA_PATH=/opt/conda
ENV CONDA_BIN=${CONDA_PATH}/bin
ENV PATH=:${CONDA_BIN}:${PATH}
ENV HOME_DIR=/root

# workdir
WORKDIR ${HOME_DIR}

# baseline
RUN echo 'apt update...' && \
    if [ ! -z $PROXY_LOCATION ]; then echo "Acquire::http::proxy \"${PROXY_LOCATION}\";" > /etc/apt/apt.conf; fi && \
    if [ ! -z $PROXY_LOCATION ]; then echo "Acquire::https::proxy \"${PROXY_LOCATION}\";" >> /etc/apt/apt.conf; fi && \
    apt update && \
    apt install -y git build-essential wget cmake libjpeg-dev zlib1g-dev

# conda
RUN if [ ! -f ${CONDA_INSTALLER} ]; then wget -q --show-progress --progress=bar:force:noscroll ${CONDA_URL} -O ${HOME_DIR}/${CONDA_INSTALLER}; fi && \
    chmod +x ${HOME_DIR}/${CONDA_INSTALLER} && \
    ${HOME_DIR}/${CONDA_INSTALLER} -b -p ${CONDA_PATH} && \
    echo ". ${CONDA_PATH}/etc/profile.d/conda.sh" >> ${HOME_DIR}/.bashrc

RUN conda create -y -n py36 python=3.6 && \
    echo "conda activate py36" >> ${HOME_DIR}/.bashrc
