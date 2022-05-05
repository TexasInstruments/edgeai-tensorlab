# base image
ARG REPO_LOCATION=""
FROM ${REPO_LOCATION}ubuntu:18.04

# proxy, path
ARG REPO_LOCATION=""
ARG PROXY_LOCATION=""
ARG SOURCE_LOCATION=""
ARG PROJECT_NAME="modelmaker"
ARG CONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
ENV http_proxy=${PROXY_LOCATION}
ENV https_proxy=${PROXY_LOCATION}
ENV no_proxy=ti.com
ENV CONDA_URL=https://repo.anaconda.com/miniconda/${CONDA_INSTALLER}
ENV CONDA_PATH=/opt/conda
ENV CONDA_BIN=${CONDA_PATH}/bin
ENV PATH=:${CONDA_BIN}:${PATH}
ENV HOME_DIR=/root

# baseline
RUN if [ ! -z $PROXY_LOCATION ]; then echo "Acquire::http::proxy \"${PROXY_LOCATION}\";" > /etc/apt/apt.conf; fi && \
    if [ ! -z $PROXY_LOCATION ]; then echo "Acquire::https::proxy \"${PROXY_LOCATION}\";" >> /etc/apt/apt.conf; fi && \
    apt update && \
    apt install -y git wget cmake && \
    apt install -y libjpeg-dev zlib1g-dev

# workdir
WORKDIR ${HOME_DIR}

# clone
COPY .ssh/ .ssh/
RUN rm -f .ssh/known_hosts
RUN ssh-keyscan bitbucket.itg.ti.com >> .ssh/known_hosts
RUN ssh-keyscan github.com >> .ssh/known_hosts

RUN echo "cloning git repositories. this may take some time..." && \
    git clone ${SOURCE_LOCATION}edgeai-benchmark.git && \
    git clone ${SOURCE_LOCATION}edgeai-mmdetection.git && \
    git clone ${SOURCE_LOCATION}edgeai-torchvision.git && \
    git clone ${SOURCE_LOCATION}edgeai-modelzoo.git --single-branch -b release
RUN rm -rf .ssh

# conda
RUN if [ ! -f ${CONDA_INSTALLER} ]; then wget -q --show-progress --progress=bar:force:noscroll ${CONDA_URL} -O ${HOME_DIR}/${CONDA_INSTALLER}; fi && \
    chmod +x ${HOME_DIR}/${CONDA_INSTALLER} && \
    ${HOME_DIR}/${CONDA_INSTALLER} -b -p ${CONDA_PATH} && \
    echo ". /opt/miniconda/etc/profile.d/conda.sh" >> ${HOME_DIR}/.bashrc && \
    echo "conda activate base" >> ${HOME_DIR}/.bashrc

WORKDIR ${HOME_DIR}/edgeai-torchvision/
RUN bash -c "setup.sh"

WORKDIR ${HOME_DIR}/edgeai-mmdetection/
RUN bash -c "setup.sh"

WORKDIR ${HOME_DIR}/edgeai-benchmark/
RUN bash -c "setup.sh"

COPY . ${HOME_DIR}/edgeai-modelmaker/
WORKDIR ${HOME_DIR}/edgeai-modelmaker/
RUN bash -c "setup.sh"
