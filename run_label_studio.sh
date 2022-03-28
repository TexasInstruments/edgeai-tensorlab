
# give the dataset a name - modify as per your dataset
DATASET_NAME="animal_detection"

BASE_PATH="./data/labelstudio"
REAL_BASE_PATH=$(realpath $BASE_PATH)
DATASET_PATH="${REAL_BASE_PATH}/${DATASET_NAME}"

# enable local folder access
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true

# to limit access to a certain folder and its subfolders only
export LOCAL_FILES_DOCUMENT_ROOT=${DATASET_PATH}

# the dataset path
export LABEL_STUDIO_BASE_DATA_DIR=${DATASET_PATH}

# can use predefined unsername/password (after it has been registered in the GUI)
#export LABEL_STUDIO_USERNAME=foo@bar.baz
#export LABEL_STUDIO_PASSWORD=foobarbaz

label-studio start
