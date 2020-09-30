#!/usr/bin/env bash

if [[ -z "$DIR_TMP" ]]; then    # If project root not defined.
    # get the directory of this file
    export CURRENT_FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    # setup root directory.
    export DIR_TMP=$(cd "${CURRENT_FILE_DIR}/.."; pwd)
fi

export DIR_TMP=$(cd "${DIR_TMP}"; pwd)
echo "The path of project root: ${DIR_TMP}"


# check if data exist.
if [[ ! -d ${DIR_TMP}/data ]]; then
    mkdir ${DIR_TMP}/data
fi

# download the chaos nli data.
cd ${DIR_TMP}/data
if [[ ! -d  chaosNLI_v1.0 ]]; then
    wget https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip
    unzip "chaosNLI_v1.0.zip"
    rm -rf "chaosNLI_v1.0.zip" && rm -rf "__MACOSX"
    echo "ChaosNLI Ready"
fi

# download the model prediction.
cd ${DIR_TMP}/data
if [[ ! -d  model_predictions ]]; then
    wget https://www.dropbox.com/s/qy7uk6ajm5x6dl6/model_predictions.zip
    unzip "model_predictions.zip"
    rm -rf "model_predictions.zip" && rm -rf "__MACOSX"
    echo "Model Prediction Ready"
fi


