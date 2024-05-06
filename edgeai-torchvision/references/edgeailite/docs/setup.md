## Environment
(Ignore this if you already have an appropriate python environment)

We have tested this on Ubuntu 22.04 OS and pyenv Python environment manager. Here are the setup instructions.

Make sure that you are using bash shell. If it is not bash shell, change it to bash. Verify it by typing:
```
echo ${SHELL}
```

Install system packages
```
sudo apt update
sudo apt install build-essential curl libbz2-dev libffi-dev liblzma-dev libncursesw5-dev libreadline-dev libsqlite3-dev libssl-dev libxml2-dev libxmlsec1-dev llvm make tk-dev wget xz-utils zlib1g-dev
```

Install pyenv using the following command.
```
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

echo '# pyenv settings ' >> ${HOME}/.bashrc
echo 'command -v pyenv >/dev/null || export PATH=":${HOME}/.pyenv/bin:$PATH"' >> ${HOME}/.bashrc
echo 'eval "$(pyenv init -)"' >> ${HOME}/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ${HOME}/.bashrc
echo '' >> ${HOME}/.bashrc

exec ${SHELL}
```

From SDK/TIDL version 9.0, the Python version required is 3.10 (Note: Prior to SDK/TIDL version 9.0, the Python version required was 3.6). Create a Python 3.10 environment if you don't have it and activate it before following the rest of the instructions.
```
pyenv install 3.10
pyenv virtualenv 3.10 torchvision
pyenv activate torchvision
pip install --upgrade pip
pip install --upgrade setuptools
```

Activation of Python environment - this activation step needs to be done everytime one starts a new terminal or shell. (Alternately, this also can be written to the .bashrc, so that this will be the default penv environment).
```
pyenv activate torchvision
```


## Installation Instructions
Install this repository as a Python package by running:
```
./setup.sh
```
