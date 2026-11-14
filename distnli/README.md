# Distributed NLI: Learning to Predict Human Opinion Distributions for Language Reasoning

The maintained CPU evaluation and prediction postprocessing workflow is documented under [Evaluate existing predictions on CPU](#evaluate-existing-predictions-on-cpu) and is covered by CI. The training, training-data preprocessing, and model inference instructions are retained for the original experiments; those legacy workflows have not been validated by the current CI.

**Outline**

* [Dependencies](#dependencies)
* [Data Preprocessing](#data-preprocessing)
* [Training Models](#training-models)
* [Evaluation](#evaluation)
* [Citation](#citation)

## Dependencies

The original training and model inference code was tested on Python 3.7 and PyTorch 1.6.0. Keep that environment separate from the root uv project, which manages CPU evaluation and prediction postprocessing with Python 3.12.

Install the legacy training dependencies from `distnli/requirements.txt` in that separate environment. For CPU scoring and postprocessing, use the root uv project as described below.

## Data Preprocessing

The code in this repository are used to replicate the experiments we conducted in the Distributed NLI paper. For the training set, our models are trained on either [Abductive NLI](https://github.com/allenai/abductive-commonsense-reasoning), or [MNLI](https://cims.nyu.edu/~sbowman/multinli/)+[SNLI](https://nlp.stanford.edu/projects/snli/). 

The evaluation datasets used in our experiments include [ChaosNLI](https://github.com/easonnie/ChaosNLI), and additionally [the UNLI dataset](https://nlp.jhu.edu/unli/) and [the PK2019 dataset](https://github.com/epavlick/NLI-variation-data).

**Note that we resplit the datasets so the dev/test results are not directly comparable to other papers using the same dataset.** Please refer to our paper for detailed description of the design choices of our evaluation setup. 

The original ChaosNLI Dropbox download is no longer available. First follow the [Hugging Face download instructions in the main README](../README.md#where-can-i-download-the-data) from the root of the ChaosNLI repository. The [community mirror](https://huggingface.co/datasets/earino/chaosnli) provides the three original JSONL files with the IDs and label distributions used by this code.

Then run the following commands from the root of the ChaosNLI repository to copy the data into this subproject and download the remaining datasets:

```bash
cd distnli
mkdir -p data/chaosNLI_v1.0
cp ../data/chaosNLI_v1.0/README.txt data/chaosNLI_v1.0/
cp ../data/chaosNLI_v1.0/chaosNLI_snli.jsonl data/chaosNLI_v1.0/
cp ../data/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl data/chaosNLI_v1.0/
cp ../data/chaosNLI_v1.0/chaosNLI_alphanli.jsonl data/chaosNLI_v1.0/
bash download_data.sh
bash preprocess_data.sh
```

The remaining commands in this README use `distnli/` as their working directory. Once `distnli/data/chaosNLI_v1.0/` has been populated, `download_data.sh` skips its legacy Dropbox download and proceeds to the other datasets.

SNLI is also available on [Hugging Face](https://huggingface.co/datasets/stanfordnlp/snli). That version exposes `premise`, `hypothesis`, and an integer `label`, whereas this project's loader expects the original `sentence1`, `sentence2`, `gold_label`, and `pairID` fields. It cannot be substituted directly without adapting the loader and preserving the original IDs; `download_data.sh` currently uses the original Stanford archive for SNLI. MNLI, alphaNLI, UNLI, and PK2019 continue to use the other sources in that script.

## Training Models
The scripts to train all the models used in our paper are in `scripts/train`. The scripts used to train the Abductive NLI models are in `scripts/train/abdnli` and the scripts for NLI models are in `scripts/train/nli`. 

Our model contains five model variants (Baseline, MC Dropout, Deep Ensemble, Recalibration and Dist. Distillation).  The training procedure for the first four methods are exactly the same for all these variants and is in `scripts/train/[task]/baseline.sh`. The Dist. Distillation variant uses different training labels and supervisions, and can be reproduced using `scripts/train/[task]/dist_train.sh` after generating the training data using commands provided in the next section. Please make sure you are at the root folder before running these commands.

By default, the trained model will be saved in the `saved_models` directory. The checkpoints will be located under `saved_models/[MODEL NAME]/checkpoints/` and the predictions on the dev set will be located under `saved_models/[MODEL NAME]/predictions/`.

## Evaluation 

### Evaluate existing predictions on CPU

For evaluating prediction files, use the root uv project. The scoring, format conversion, ensemble, and temperature calibration commands below run on CPU in its locked environment. Their import paths are configured automatically. Running `uv` from `distnli/` discovers the parent `pyproject.toml` while keeping `distnli/` as the working directory.

To download only ChaosNLI into this subproject, run the following commands from `distnli/`:

```bash
uv sync --locked
uv run --locked python ../scripts/download_data.py --output-dir data/chaosNLI_v1.0
uv run --locked python src/eval_scripts/evaluate.py \
    --task_name uncertainty_nli \
    --data_file data/chaosNLI_v1.0/chaosNLI_snli.jsonl \
    --prediction_file ../data/prediction_samples/snli_random_baseline.json
```

For raw prediction JSONL, `unify_prediction_format.py` and `get_ensemble_prediction.py` infer the task from the number of classes, so paths do not need to contain `nli` or `alpha`. You can supply `--task_name uncertainty_nli` or `--task_name uncertainty_abdnli` explicitly, and use `--model_name NAME` to override the default output name `roberta-large`. Ensemble members must contain the same unique example IDs; the predicted label is derived from the averaged probabilities. Save the ensemble output outside the directory containing member JSONL files.

`temperature_calibration.py` fits a temperature in `[0.001, 1000]` on the calibration examples' `label_dist` values by optimizing `log(T)` with SciPy on CPU. Keep calibration examples separate from the examples used for final evaluation. The command accepts an optional fourth positional argument to apply an existing positive temperature; `--model_name NAME` selects a model if the input contains more than one. UNLI uses `--task_name unli` and reports Pearson and Spearman correlations against `unli_label`.

See the main README for the [metric definitions](../README.md#how-can-i-evaluate-my-own-results).

### Main Evaluation Results

1. Get the model's prediction on the testing dataset. 

   * To get the result on the testing dataset of the Baseline (or Deep Ensemble, Recalibration, and Dist. Distillation) variant, run `bash scripts/eval/[abdnli or nli]/baseline.sh`
   * To get the MC Dropout's result on the testing dataset, run `bash scripts/eval/[abdnli or nli]/mc.sh`

   Replace the `[MODEL PATH]/model.pt` placeholder in each inference script with your checkpoint path. These scripts require the original PyTorch/Transformers environment described above and default to four CUDA GPUs. Use `--cpu` or `--single_gpu` when invoking the Python entry point for a different device setup. Distributed inference uses port `29500` by default and respects `MASTER_ADDR` and `MASTER_PORT` from the environment.

   NLI inference saves predictions under `saved_models/[run]/eval_predictions/`; alphaNLI test inference uses `saved_models/[run]/predictions_ontest/`. Each run directory includes a timestamp and the experiment name.

2. Calculate the distributional evaluation metrics.

   Assume the location of the prediction file we get from the last step is `[pred_file]` and the location of the test examples is `[data_file]`, then 

   * To get the result of the **Baseline** (or the **MC Dropout** and **Dist. Distillation**) variant,  run the following commands sequentially
     1. `uv run --locked python src/eval_scripts/unify_prediction_format.py [pred_file] [output_file_for_new_format]`
     2. `uv run --locked python src/eval_scripts/evaluate.py --task_name [uncertainty_nli or uncertainty_abdnli] --data_file [data_file] --prediction_file [output_file_for_new_format]`
   * To get the result of the **Deep Ensemble** variant, use predictions from multiple models trained with different seeds, and then
     1. Put all the predictions of different seeds under a single directory `[ens dir]` with file name `{seed number}.jsonl`
     2. Run `uv run --locked python src/eval_scripts/get_ensemble_prediction.py [ens dir] [output_file_for_new_format]`
     3. Run `uv run --locked python src/eval_scripts/evaluate.py --task_name [uncertainty_nli or uncertainty_abdnli] --data_file [data_file] --prediction_file [output_file_for_new_format]`
   * To get the result of the **Recalibration** variant, run the following command sequentially
     1. `uv run --locked python src/eval_scripts/unify_prediction_format.py [pred_file] [output_file_for_new_format]`
     2. `uv run --locked python src/eval_scripts/temperature_calibration.py [output_file_for_new_format] [label_file_for_calibration] [output_file_for_calibrated_prediction]`
     3. `uv run --locked python src/eval_scripts/evaluate.py --task_name [uncertainty_nli or uncertainty_abdnli] --data_file [data_file] --prediction_file [output_file_for_calibrated_prediction]`

### Get Training data for Dist. Distribution Models

To train the Dist. Distribution model, we need to first convert model's prediction on the training data, and convert its format back to the original dataset format. Specifically, 

1. Follow the previous instructions and get the calibration temperature `[temp]` returned by running `python src/eval_scripts/temperature_calibration.py [output_file_for_new_format] [label_file_for_calibration] [output_file_for_calibrated_prediction]`
2. First get the model's prediction on the training set by run `bash scripts/eval/[abdnli or nli]/eval_on_train.sh`. The prediction file will be saved under `saved_models/[experiment name]/predictions`. 
3. For NLI models, run `python src/eval_scripts/pred_dict_to_train_nlitrain.py [pred_file] [train_dataset_file] [output_path] [temp]` to output the training data. The output path should contain the phrase `relabel` to be identified by the code. The new training file will be saved at `[output_path]`. For Abductive NLI models, run `python src/eval_scripts/pred_dict_to_train_alphatrain.py` instead.

## Citation

```
@inproceedings{xzhou2022distnli,
	Author = {Xiang Zhou and Yixin Nie and Mohit Bansal},
	Booktitle = {Findings of the Association for Computational Linguistics: ACL 2022},
	Publisher = {Association for Computational Linguistics},
	Title = {Distributed NLI: Learning to Predict Human Opinion Distributions for Language Reasoning},
	Year = {2022}
}
```
