# # Setup
mu2einit
setup dhtools
getToken

# source make_file_list <defname> <n_files>

# Get inputs
DEFNAME=$1
N_FILES=$2

# Define variables
OUT_FILE_NAME="${DEFNAME%.*}.${N_FILES}.txt"
OUT_FILE_DIR="../input/filelists"
if [ ! -d "${OUT_FILE_DIR}" ]; then
    mkdir -p "$OUT_FILE_DIR"
fi
OUT_FILE_PATH="${OUT_FILE_DIR}/${OUT_FILE_NAME}"
if [ -f $OUT_FILE_PATH ]; then
    rm $OUT_FILE_PATH
fi

# Run command
CMD=$(samweb list-definition-files "${DEFNAME}" | head -n "${N_FILES}" > "${OUT_FILE_PATH}")
$CMD

echo "Wrote ${OUT_FILE_PATH}"

head ${OUT_FILE_PATH}
