#!/usr/bin/env bash

if [[ -z "$DIR_TMP" ]]; then    # If project root not defined.
    # get the directory of this file
    export CURRENT_FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    # setup root directory.
    export DIR_TMP=$(cd "${CURRENT_FILE_DIR}/"; pwd)
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

# download snli
cd ${DIR_TMP}/data
if [[ ! -d  snli_1.0 ]]; then
    wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    unzip "snli_1.0.zip"
    rm -rf "snli_1.0.zip" && rm -rf "__MACOSX"
    echo "SNLI Ready"
fi


# download mnli
cd ${DIR_TMP}/data
if [[ ! -d  multinli_1.0 ]]; then
    wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
    unzip "multinli_1.0.zip"
    rm -rf "multinli_1.0.zip" && rm -rf "__MACOSX"
    echo "MNLI Ready"
fi


# download unli
cd ${DIR_TMP}/data
if [[ ! -d  unli ]]; then
    wget http://nlp.jhu.edu/unli/u-snli.zip
    unzip "u-snli.zip" -d "unli"
    rm -rf "u-snli.zip" 
    echo "UNLI Ready"
fi


# download pk2019
cd ${DIR_TMP}/data
if [[ ! -d  pk2019 ]]; then
    mkdir pk2019
    wget https://raw.githubusercontent.com/epavlick/NLI-variation-data/master/sentence-pair-analysis/preprocessed-data.jsonl  
    mv preprocessed-data.jsonl pk2019
    echo "PK2019 Ready"
fi

# download abdnli
cd ${DIR_TMP}/data
if [[ ! -d  anli ]]; then
    wget https://storage.googleapis.com/ai2-mosaic/public/abductive-commonsense-reasoning-iclr2020/anli.zip
    unzip "anli.zip" 
    rm -rf "anli.zip" 
    echo "AbdNLI Ready"
fi


