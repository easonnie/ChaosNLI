# Distributed NLI: Learning to Predict Human Opinion Distributions for Language Reasoning
**Outline**

* [Dependencies](#dependencies)
* [Data Preprocessing](#data-preprocessing)
* [Training Models](#training-models)
* [Evaluation](#evaluation)
* [Citation](#citation)

## Dependencies

The code is tested on Python 3.7 and Pytorch 1.6.0

Other dependencies are listed in `requirements.txt` and can be installed by running `pip install -r requirements.txt`

## Data Preprocessing

The code in this repository are used to replicate the experiments we conducted in the Distributed NLI paper. For the training set, our models are trained on either [Abductive NLI](https://github.com/allenai/abductive-commonsense-reasoning), or [MNLI](https://cims.nyu.edu/~sbowman/multinli/)+[SNLI](https://nlp.stanford.edu/projects/snli/). 

The evaluation datasets used in our experiments include [ChaosNLI](https://github.com/easonnie/ChaosNLI), and additionally [the UNLI dataset](https://nlp.jhu.edu/unli/) and [the PK2019 dataset](https://github.com/epavlick/NLI-variation-data).

**Note that we resplit the datasets so the dev/test results are not directly comparable to other papers using the same dataset.** Please refer to our paper for detailed description of the design choices of our evaluation setup. 

To download and preprocess the training and evaluation datasets, first run `bash download_data.sh` under the root directory, and then run `bash preprocess_data.sh`.

## Training Models
The scripts to train all the models used in our paper are in `scripts/train`. The scripts used to train the Abductive NLI models are in `scripts/train/abdnli` and the scripts for NLI models are in `scripts/train/nli`. 

Our model contains five model variants (Baseline, MC Dropout, Deep Ensemble, Recalibration and Dist. Distillation).  The training procedure for the first four methods are exactly the same for all these variants and is in `scripts/train/[task]/baseline.sh`. The Dist. Distillation variant uses different training labels and supervisions, and can be reproduced using `scripts/train/[task]/dist_train.sh` after generating the training data using commands provided in the next section. Please make sure you are at the root folder before running these commands.

By default, the trained model will be saved in the `saved_models` directory. The checkpoints will be located under `saved_models/[MODEL NAME]/checkpoints/` and the predictions on the dev set will be located under `saved_models/[MODEL NAME]/predictions/`.

## Evaluation 

### Main Evaluation Results

1. Get the model's prediction on the testing dataset. 

   * To get the result on the testing dataset of the Baseline (or Deep Ensemble, Recalibration, and Dist. Distillation) variant, run `bash scripts/eval/[abdnli or nli]/baseline.sh`
   * To get the MC Dropout's result on the testing dataset, run `bash scripts/eval/[abdnli or nli]/mc.sh`

   The prediction result will be saved under `saved_models/[experiment name]/predictions`

2. Calculate the distributional evaluation metrics.

   Assume the location of the prediction file we get from the last step is `[pred_file]` and the location of the test examples is `[data_file]`, then 

   * To get the result of the **Baseline** (or the **MC Dropout** and **Dist. Distillation**) variant,  run the following commands sequentially
     1. `python src/eval_scripts/unify_prediction_format.py [pred_file] [output_file_for_new_format]`
     2.  `python src/eval_scripts/evaluate.py --task_name [uncertainty_nli or uncertainty_abdnli] --data_file [data_file] --prediction_file [output_file_for_new_format] ` 
   * To get the result of the **Deep Ensemble** variant, you need to train multiple models with the same seeds, and then
     1. Put all the predictions of different seeds under a single directory `[ens dir]` with file name `{seed number}.jsonl`
     2. Run `python src/eval_scripts/get_ensemble_prediction.py [ens dir] [output_file_for_new_format]`
     3. Run `python src/eval_scripts/evaluate.py --task_name [uncertainty_nli or uncertainty_abdnli] --data_file [data_file] --prediction_file [output_file_for_new_format] ` 
   * To get the result of the **Recalibration** variant, run the following command sequentially
     1. `python src/eval_scripts/unify_prediction_format.py [pred_file] [output_file_for_new_format]`
     2. `python src/eval_scripts/temperature_calibration.py [output_file_for_new_format] [label_file_for_calibration] [output_file_for_calibrated_prediction]` 
     3.  `python src/eval_scripts/evaluate.py --task_name [uncertainty_nli or uncertainty_abdnli] --data_file [data_file] --prediction_file [output_file_for_calibrated_prediction] ` 

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

## 