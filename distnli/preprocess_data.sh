#!/usr/bin/env bash

if [[ -z "$DIR_TMP" ]]; then    # If project root not defined.
    # get the directory of this file
    export CURRENT_FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    # setup root directory.
    export DIR_TMP=$(cd "${CURRENT_FILE_DIR}/"; pwd)
fi

export DIR_TMP=$(cd "${DIR_TMP}"; pwd)
echo "The path of project root: ${DIR_TMP}"

# get dist dev set
cd ${DIR_TMP}/data/chaosNLI_v1.0
head chaosNLI_snli.jsonl -n 100 > snli100.jsonl
tail chaosNLI_snli.jsonl -n +101 > snli1414.jsonl

head chaosNLI_mnli_m.jsonl -n 100 > mnli100.jsonl
tail chaosNLI_mnli_m.jsonl -n +101 > mnli1499.jsonl

head chaosNLI_alphanli.jsonl -n 100 > alpha100.jsonl
tail chaosNLI_alphanli.jsonl -n +101 > alpha1432.jsonl
echo "Finish splitting datasets"

# switch dev and test in abdnli to avoid overlapping examples
cd ${DIR_TMP}/data/anli/
mv dev.jsonl switchedtest.jsonl
mv dev-labels.lst switchedtest-labels.lst
mv test.jsonl switcheddev.jsonl
mv test-labels.lst switcheddev-labels.lst

# preprocess unli
cd ${DIR_TMP}/data/unli/
python ${DIR_TMP}/src/preprocessing/clean_unli.py
python ${DIR_TMP}/src/preprocessing/make_chaos_format.py unli
echo "Finish Preprocessing UNLI"


# preprocess pk2019
cd ${DIR_TMP}/data/pk2019/
python ${DIR_TMP}/src/preprocessing/clean_pk.py
python ${DIR_TMP}/src/preprocessing/make_chaos_format.py pk
echo "Finish Preprocessing PK2019"







