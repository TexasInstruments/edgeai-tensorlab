##installing pandaset devkit

CWD=$(pwd)
echo "Installing PandaSet"
rm -r -f qpandaset-devkit
git clone https://github.com/scaleapi/pandaset-devkit.git
cp -f ${CWD}/requirements/pandaset_requirements.txt ${CWD}/pandaset-devkit/python/requirements.txt
cd ${CWD}/pandaset-devkit/python
pip install .
cd ${CWD}
rm -r -f pandaset-devkit