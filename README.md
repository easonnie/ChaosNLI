# Chaos NLI
What Can We Learn from **C**ollective **H**um**A**n **O**pinion**S** on **N**atural **L**anguage **I**nference Data (**ChaosNLI**)?

[Yixin Nie](https://easonnie.github.io/), [Xiang Zhou](https://owenzx.github.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/), EMNLP 2020 \[[bib](https://github.com/easonnie/ChaosNLI#citation)\]

Distributed NLI: Learning to Predict Human Opinion Distributions for Language Reasoning

[Xiang Zhou](https://owenzx.github.io/)\*, [Yixin Nie](https://easonnie.github.io/)\*, [Mohit Bansal](https://www.cs.unc.edu/~mbansal/), Findings of ACL 2022 [[bib](https://github.com/easonnie/ChaosNLI#citation)\] 

**Outline**

- [Chaos NLI](#chaos-nli)
  - [Quick start](#quick-start)
  - [Papers](#papers)
  - [Motivation](#motivation)
  - [Examples](#examples)
    - [NLI Example](#nli-example)
    - [Abductive NLI Example](#abductive-nli-example)
  - [Scoreboard](#scoreboard)
    - [ChaosNLI - SNLI](#chaosnli---snli)
    - [ChaosNLI - MNLI](#chaosnli---mnli)
    - [ChaosNLI - Abductive NLI](#chaosnli---abductive-nli)
  - [Data and Format](#data-and-format)
    - [Where can I download the data？](#where-can-i-download-the-data)
    - [What is the format?](#what-is-the-format)
      - [Data Format](#data-format)
      - [Model Prediction Format](#model-prediction-format)
  - [Results](#results)
    - [How to reproduce the results on the paper?](#how-to-reproduce-the-results-on-the-paper)
    - [Improving Distribution Estimation](#improving-distribution-estimation)
  - [Evaluate](#evaluate)
    - [How can I evaluate my own results?](#how-can-i-evaluate-my-own-results)
  - [Development and verification](#development-and-verification)
  - [Citation](#citation)
  - [License](#license)


## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the repository root:

```bash
uv sync --locked
uv run --locked python scripts/download_data.py
uv run --locked python src/scripts/evaluate.py \
    --task_name uncertainty_nli \
    --data_file data/chaosNLI_v1.0/chaosNLI_snli.jsonl \
    --prediction_file data/prediction_samples/snli_random_baseline.json
```

The uv project manages Python 3.12 and the CPU dependencies for dataset evaluation, plotting, and prediction postprocessing. `uv sync --locked` creates `.venv` and installs the versions in `uv.lock`; `uv run --locked` uses that environment without manual activation. The example above evaluates the checked-in random baseline. Replace the prediction file with your own results to evaluate a model.

The original training and model inference code in `distnli/` uses a separate legacy PyTorch/Transformers environment; see its [README](distnli/README.md). Those dependencies are not part of the root uv project.

For a pip-only setup, use Python 3.12 in a separate virtual environment and install `requirements.txt`. That file is exported from `uv.lock`, rather than maintained independently.

## Papers
[What Can We Learn from Collective Human Opinions on Natural Language Inference Data?](https://arxiv.org/abs/2010.03532)

[Distributed NLI: Learning to Predict Human Opinion Distributions for Language Reasoning](https://arxiv.org/abs/2104.08676)

## Motivation
Most NLU evaluations have focused on using the **majority label** with presumably high agreement as the ground truth. Less attention has been paid to the **distribution of human opinions**.  
We believe that we should consider to take a step back and look at **collective human opinions** on NLP tasks.  

![everything is an opinion](https://github.com/easonnie/ChaosNLI/blob/master/img/opinion.jpg)

## Examples
**Chaos NLI** is a dataset with 100 annotations per example (a total of 4,645 * 100 annotations) for some existing data points in the development set of [SNLI](https://nlp.stanford.edu/projects/snli/), [MNLI](https://cims.nyu.edu/~sbowman/multinli/), and [Abductive NLI](http://abductivecommonsense.xyz/).  
Common practice for model evaluation, use the majority labels in the original dataset as the one gold labels and calculate model accuracy based on that.  
A more fine-grained evaluation should consider comparing **model distribution outputs** with the **human label distributions**.  

### NLI Example
Is the hypothesis entailed or contradicted (or neither) by the premise?  

Premise | Hypothesis | New Annotations | Old Annotations | BERT-Large Prediction
--- | --- | --- | --- | ---
There are a number of expensive jewelry and other duty-free shops, all with goods priced in US dollars (duty-free goods must always be paid for in foreign currency).|You can pay using the US dollar when buying goods from the duty-free shops.|E(51), N(3), C(46)|C, C, E, N, C|E(50.03%), N(5.33%), C(44.63%)
This number represents the most reliable, albeit conservative, estimate of cases closed in 1999 by LSC grantees.|This is an actual verified number of closed cases.|E(21), N(12), C(67)|C, E, C, E, E|E(15.72%), N(14.04%), C(70.24%)

### Abductive NLI Example
Which of the two hypotheses is more likely to cause Observation-Beginning to turn into Observation-Ending?  

Observation Start | Hypothesis 1 | Hypothesis 2 | Observation End | New Annotation | Old Annotation | BERT-Large Prediction
--- | --- | --- | --- | --- | --- | ---
Ruth was playing video games.|Ruth started playing a new game.|Ruth challenged her friend to a game of golf.|She won the game easily.|1(58), 2(42)|1|1(99.99%), 2(0.01%)
Gina entered her classroom and went to her desk.|Tami had asked Gina not to talk with the gossip girls.|She started talking to some friends, but a bully named Tami butted in.|The fact that Tami was still talking to them made Gina uncomfortable.|1(50), 2(50)|2|1(0.01%), 2(99.99%)
Amy and her friends were out at 3 AM.|They started getting followed by a policeman, ran, and hid behind a building.|The decided to break into the football field. When suddenly they saw a flashlight comming towards them. They all started running for the bleachers.|They stayed there breathing hard, and praying they hadn't been seen.|1(50), 2(50)|2|Unavailable

The model probabilities for the third example are omitted because the original values sum to 90%.

## Scoreboard
### ChaosNLI - SNLI
Model | Link | Date | JSD | KL | Old Accuracy | New Accuracy
--- | --- | --- | --- | --- | --- | ---
BERT-Large | [Nie et al., 2020](https://arxiv.org/abs/2010.03532) | 09-29-2020 | 0.23 | 0.5017 | 0.7266 | 0.7384
RoBERTa-Large | [Nie et al., 2020](https://arxiv.org/abs/2010.03532) | 09-29-2020 | 0.221 | 0.4937 | 0.749 | 0.7867
XLNet-Large | [Nie et al., 2020](https://arxiv.org/abs/2010.03532) | 09-29-2020 | 0.2259 | 0.5054 | 0.7431 | 0.7807	    	   	    	

### ChaosNLI - MNLI
Model | Link | Date | JSD | KL | Old Accuracy | New Accuracy
--- | --- | --- | --- | --- | --- | ---
BERT-Large | [Nie et al., 2020](https://arxiv.org/abs/2010.03532) | 09-29-2020 | 0.3152 | 0.8449 | 0.6123 | 0.5691
RoBERTa-Large | [Nie et al., 2020](https://arxiv.org/abs/2010.03532) | 09-29-2020 | 0.3112 | 0.8701 | 0.6742 | 0.6354
XLNet-Large | [Nie et al., 2020](https://arxiv.org/abs/2010.03532) | 09-29-2020 | 0.3116 | 0.8818 | 0.6742 | 0.6185     	    	    	    	

### ChaosNLI - Abductive NLI
Model | Link | Date | JSD | KL | Old Accuracy | New Accuracy
--- | --- | --- | --- | --- | --- | ---
BERT-Large | [Nie et al., 2020](https://arxiv.org/abs/2010.03532) | 09-29-2020 | 0.3055 | 3.7996 | 0.6802 | 0.6821
RoBERTa-Large | [Nie et al., 2020](https://arxiv.org/abs/2010.03532) | 09-29-2020 | 0.2128 | 1.3898 | 0.8531 | 0.8368
XLNet-Large | [Nie et al., 2020](https://arxiv.org/abs/2010.03532) | 09-29-2020 | 0.2282 | 1.8166 | 0.814 | 0.8133         	    	    	     	   	    	    	

If you want your results to be showed on the Scoreboard, please email us (<yixin1@cs.unc.edu> or <nyixin318@gmail.com>) with **the name of the entry**, **a link to your method**, and **your model prediction file** (please follow the [instruction](https://github.com/easonnie/ChaosNLI#how-can-i-evaluate-my-own-results) to build and test your prediction file).

## Data and Format
### Where can I download the data？
The original ChaosNLI Dropbox download is no longer available. Download the data from the [ChaosNLI community mirror on Hugging Face](https://huggingface.co/datasets/earino/chaosnli) instead. The mirror preserves the original JSONL format, including example IDs, label counts, and label distributions. It contains all three components: SNLI (1,514 examples), MNLI matched (1,599 examples), and alphaNLI (1,532 examples).

Run the following command from the root of this repository. It downloads the original files directly from Hugging Face into the paths expected by the evaluation scripts, using only the Python standard library. The revision is pinned for reproducibility, and every file is verified against its SHA-256 checksum. Existing valid files are reused; a failed download exits with an error.

```bash
uv run --locked python scripts/download_data.py
```

The downloader also works independently with `python3 scripts/download_data.py` or `bash scripts/download_data.sh`, using only the Python standard library. To choose another destination, pass `--output-dir /path/to/chaosNLI_v1.0`. The default destination is this repository's `data/chaosNLI_v1.0/`, even when the script is called from another working directory. The resulting data layout is:

```
├── LICENSE
├── README.md
├── data
│   ├── chaosNLI_v1.0
│   │   ├── README.txt
│   │   ├── chaosNLI_alphanli.jsonl
│   │   ├── chaosNLI_mnli_m.jsonl
│   │   └── chaosNLI_snli.jsonl
│   └── prediction_samples
│       ├── abdnli_random_baseline.json
│       ├── mnli_random_baseline.json
│       └── snli_random_baseline.json
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── scripts
├── setup.sh
└── src
```

The Hugging Face mirror contains the dataset, not the historical model predictions. Evaluating your own predictions only requires the dataset and your prediction file; see [Evaluate](#evaluate). Reproducing the published model tables additionally requires `data/model_predictions/model_predictions_for_abdnli.json` and `data/model_predictions/model_predictions_for_snli_mnli.json`, which are not included in this mirror or in the repository. The checked-in random baseline files are examples of the prediction format, not replacements for those historical predictions.

### What is the format?
#### Data Format
**ChaosNLI** is in `JSONL`. Every line in the file is one single JavaScript Object and can be easily loaded in python dictionary.  
The fields of the objects are self-explanatory. Please see the following sample to learn the details about the field.
```JS
// ChaosNLI-(S+M)NLI  'chaosNLI_mnli_m.jsonl' and 'chaosNLI_snli.jsonl'

{   "uid": "46359n",                                // The unique id of this example. This is the same id as the one in SNLI/MNLI pairID field and can be used to map back to the SNLI/MNLI data.                            
    "label_counter": {"e": 76, "n": 20, "c": 4},    // The dictionary indicates the count of each newly collected ChaosNLI label. e=entailment; n=neutral; c=contradiction.
    "majority_label": "e",                          // The majority label of the new ChaosNLI labels. (This might be different from the old label.)
    "label_dist": [0.76, 0.2, 0.04],                // The label distributions. (The order matters and should always be: [E, N, C])
    "label_count": [76, 20, 4],                     // The label counts. (The order matters and should always be: [E, N, C])
    "entropy": 0.9510456605801273,                  // The entropy of the label distriction. (base is 2)
    "example": {
        "uid": "46359n", 
        "premise": "Although all four categories of emissions are down substantially, they only achieve 50-75% of the proposed cap by 2007 (shown as the dotted horizontal line in each of the above figures).", 
        "hypothesis": "The downturn in the emission categories simply isn't enough in our estimation.", 
        "source": "mnli_agree_3"
    }, 
    "old_label": "n",                                                               // The old majority (gold) label.
    "old_labels": ["neutral", "entailment", "neutral", "neutral", "contradiction"]  // The old individual annotations.
}
```

```JS
// ChaosNLI-alphaNLI 'chaosNLI_alphanli.jsonl'
{   "uid": "a05ed03f-9713-4272-9cc0-c20b823bf5e4-1",    // The unique id of this example. This is the same id as the one in alaphanli story_id field and can be used to map back to the alaphanli data.
    "label_counter": {"2": 9, "1": 91},                 // The dictionary indicates the count of each newly collected ChaosNLI label.
    "majority_label": 1,                                // The majority label of the new ChaosNLI labels. (This might be different from the old label.)
    "label_dist": [0.91, 0.09],                         // The label distributions. (The order matters and should always be: [1, 2])
    "label_count": [91, 9],                             // The label counts. (The order matters and should always be: [1, 2])
    "entropy": 0.4364698170641029,                      // The entropy of the label distriction. (base is 2)
    "example": {
        "uid": "a05ed03f-9713-4272-9cc0-c20b823bf5e4-1", 
        "obs1": "Maya was walking alongside a river, looking for frogs.", 
        "obs2": "Luckily, she was able to get back up and walk home safely.", 
        "hyp1": "She ended up falling into the river.", 
        "hyp2": "Maya slipped on some rocks and broke her back.", 
        "source": "abdnli_dev"}, 
    "old_label": 1                                      // The old label. (Abductive NLI only have one original label for each example.)
}
```

#### Model Prediction Format
The original release provided predictions of BERT, RoBERTa, XLNet, BART, ALBERT, and DistilBERT on SNLI, MNLI, and alphaNLI in `data/model_predictions`. These historical prediction files are separate from the Hugging Face dataset mirror. The excerpts below use comments and ellipses for explanation; prediction files must contain complete, valid JSON.
```JS
// ChaosNLI-alphaNLI 'data/model_predictions/model_predictions_for_abdnli.json'
{
    "bert-base": 
        {
            "58090d3f-8a91-4c89-83ef-2b4994de9d241":    // Notices: the order in "logits" matters and should always be [1, 2]
                {"uid": "58090d3f-8a91-4c89-83ef-2b4994de9d241", "logits": [17.921875, 22.921875], "predicted_label": 2},
            "91f9d1d9-934c-44f9-9677-3614def2874b2":
                {"uid": "91f9d1d9-934c-44f9-9677-3614def2874b2", "logits": [-10.7421875, -15.8671875], "predicted_label": 1},
            ...... // Predictions for other examples
        }
    "bert-large":
        {
            "58090d3f-8a91-4c89-83ef-2b4994de9d241": 
                {"uid": "58090d3f-8a91-4c89-83ef-2b4994de9d241", "logits": [26.046875, 26.234375], "predicted_label": 2}, 
            "91f9d1d9-934c-44f9-9677-3614def2874b2": 
                {"uid": "91f9d1d9-934c-44f9-9677-3614def2874b2", "logits": [-19.484375, -24.21875], "predicted_label": 1},
            ...... // Predictions for other examples
        }
    ......  // Prediction for other models
}
```

```JS
// ChaosNLI-(S+M)NLI 'data/model_predictions/model_predictions_for_snli_mnli.json'
{   "bert-base": 
        {
            "4705552913.jpg#2r1n":                      // Notices: the order in "logits" matters and should always be [E, N, C]
                {"uid": "4705552913.jpg#2r1n", "logits": [-2.0078125, 4.453125, -2.591796875], "predicted_label": "neutral"}, 
            "4705552913.jpg#2r1e": 
                {"uid": "4705552913.jpg#2r1e", "logits": [3.724609375, -0.66064453125, -2.443359375], "predicted_label": "entailment"},
            ......  // Predictions for other examples
        },
    "bert-large": 
        {
            "4705552913.jpg#2r1n":                      // Notices: the order in "logits" matters and should always be [E, N, C]
                {"uid": "4705552913.jpg#2r1n", "logits": [-2.0234375, 4.67578125, -2.78515625], "predicted_label": "neutral"}, 
            "4705552913.jpg#2r1e": 
                {"uid": "4705552913.jpg#2r1e", "logits": [3.39453125, -0.673828125, -3.080078125], "predicted_label": "entailment"},
            ......  // Predictions for other examples
        },
     ......  // Prediction for other models
```

## Results
### How to reproduce the results on the paper?
After downloading the dataset and separately obtaining the historical model prediction files listed above, run the following commands from the repository root:
```bash
uv sync --locked
uv run --locked python src/evaluation/model_pref.py
```

Downloading the Hugging Face dataset alone is sufficient to evaluate your own predictions, but does not provide the historical model predictions needed by this command.

The following is the original release's output. Per-model JSD, KL, and accuracy retain the original metric definitions. `All Correct Count` now uses the intersection of the examples every model predicts correctly; its corrected values can differ from this historical output.
```
Load Jsonl: data/chaosNLI_v1.0/chaosNLI_alphanli.jsonl
1532it [00:00, 54898.24it/s]
------------------------------------------------------------
Data: ChaosNLI - Abductive Commonsense Reasoning (alphaNLI)
All Correct Count: 930
Model Name          	JSD       	KL        	Old Acc.  	New Acc.
bert-base           	0.3209    	3.7981    	0.6527    	0.6534
xlnet-base          	0.2678    	1.0209    	0.6743    	0.6867
roberta-base        	0.2394    	0.8272    	0.7154    	0.7396
bert-large          	0.3055    	3.7996    	0.6802    	0.6821
xlnet-large         	0.2282    	1.8166    	0.814     	0.8133
roberta-large       	0.2128    	1.3898    	0.8531    	0.8368
bart-large          	0.2215    	1.5794    	0.8185    	0.814
albert-xxlarge      	0.2208    	2.9598    	0.844     	0.8473
distilbert          	0.3101    	1.0345    	0.592     	0.607
------------------------------------------------------------
Load Jsonl: data/chaosNLI_v1.0/chaosNLI_snli.jsonl
1514it [00:00, 71640.88it/s]
------------------------------------------------------------
Data: ChaosNLI - Stanford Natural Language Inference (SNLI)
All Correct Count: 1063
Model Name          	JSD       	KL        	Old Acc.  	New Acc.
bert-base           	0.2345    	0.481     	0.7008    	0.7292
xlnet-base          	0.2331    	0.5121    	0.7114    	0.7365
roberta-base        	0.2294    	0.5045    	0.7272    	0.7536
bert-large          	0.23      	0.5017    	0.7266    	0.7384
xlnet-large         	0.2259    	0.5054    	0.7431    	0.7807
roberta-large       	0.221     	0.4937    	0.749     	0.7867
bart-large          	0.2203    	0.4714    	0.7424    	0.7827
albert-xxlarge      	0.235     	0.5342    	0.7153    	0.7814
distilbert          	0.2439    	0.4682    	0.6711    	0.7021
------------------------------------------------------------
Load Jsonl: data/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl
1599it [00:00, 50454.33it/s]
------------------------------------------------------------
Data: ChaosNLI - Multi-Genre Natural Language Inference (MNLI)
All Correct Count: 816
Model Name          	JSD       	KL        	Old Acc.  	New Acc.
bert-base           	0.3055    	0.7204    	0.5991    	0.5591
xlnet-base          	0.3069    	0.7927    	0.6373    	0.5891
roberta-base        	0.3073    	0.7807    	0.6391    	0.5922
bert-large          	0.3152    	0.8449    	0.6123    	0.5691
xlnet-large         	0.3116    	0.8818    	0.6742    	0.6185
roberta-large       	0.3112    	0.8701    	0.6742    	0.6354
bart-large          	0.3165    	0.8845    	0.6635    	0.5922
albert-xxlarge      	0.3159    	0.862     	0.6485    	0.5897
distilbert          	0.3133    	0.6652    	0.5472    	0.5103
------------------------------------------------------------
```

To examine **the factor of human agreement on model performance**, check out this informative [jupyter-notebook](src/notebook/binned_plot.ipynb) to find the partitioned results according to the entropy range. The notebook requires the historical model prediction files listed above.

### Improving Distribution Estimation
In the [distnli](https://github.com/easonnie/ChaosNLI/tree/master/distnli) folder, we provide implementation of several methods (e.g. Monte Carlo Dropout, Deep Ensemble, Re-Calibration, etc.) used in the Distributed NLI paper that can improve the distribution estimation performance upon the vanilla BERT-style baseline. Please see the README under that folder for detailed instructions.

## Evaluate
### How can I evaluate my own results?
Build your prediction file according to the sample files in `data/prediction_samples`. The JSON object maps each model name to a dictionary of example IDs. Each prediction must contain `predicted_label` and either `predicted_probabilities` or `logits`. NLI vectors use the order `[entailment, neutral, contradiction]`, with string labels `entailment`, `neutral`, or `contradiction` (also accepted as `e`, `n`, or `c`). AlphaNLI vectors use `[1, 2]`, with integer labels `1` or `2`. Every evaluated example must have a prediction.

From the repository root, install the evaluation dependencies and download the data:

```bash
uv sync --locked
uv run --locked python scripts/download_data.py
```

These examples run on CPU using the checked-in random baseline predictions. Replace `--prediction_file` with your own prediction file to evaluate your method. The commands configure their own import paths, so sourcing `setup.sh` is optional.

```bash
# For MNLI:
uv run --locked python src/scripts/evaluate.py \
    --task_name uncertainty_nli \
    --data_file data/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl \
    --prediction_file data/prediction_samples/mnli_random_baseline.json

# For SNLI:
uv run --locked python src/scripts/evaluate.py \
    --task_name uncertainty_nli \
    --data_file data/chaosNLI_v1.0/chaosNLI_snli.jsonl \
    --prediction_file data/prediction_samples/snli_random_baseline.json

# For alphaNLI:
uv run --locked python src/scripts/evaluate.py \
    --task_name uncertainty_abdnli \
    --data_file data/chaosNLI_v1.0/chaosNLI_alphanli.jsonl \
    --prediction_file data/prediction_samples/abdnli_random_baseline.json
```

For compatibility with the published results, the column named `JSD` reports Jensen-Shannon **distance** (the square root of the divergence), using natural logarithms. `KL` is `KL(human || model)`. Both are averaged over examples. Model probabilities are floored at `1e-15` and renormalized for finite KL values. Invalid distributions and missing predictions produce an error rather than a partial score. Empty entropy bins and empty all-correct subsets have undefined metrics (`NaN`).

## Development and verification

Run the regression checks in the locked evaluation environment:

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

To verify a fresh download, the three random baseline scores, and the NLI/alphaNLI conversion, ensemble, calibration, and held-out evaluation workflows:

```bash
uv run --locked python scripts/verify_evaluation.py
```

This command uses a temporary directory and removes its downloaded data and predictions when it finishes. To reuse existing data for an offline check, pass `--data-dir data/chaosNLI_v1.0`. GitHub Actions runs the regression and real-data checks on Linux and macOS.

Declare evaluation dependencies in `pyproject.toml` and commit the updated `uv.lock`. Regenerate the pip compatibility file after dependency changes:

```bash
uv lock
uv export --locked --no-dev --no-hashes --no-emit-project --output-file requirements.txt
```

Generated environments, downloaded datasets, Python caches, and plot PDFs are ignored by Git. Keep notebook outputs cleared before committing them.

## Citation

**What Can We Learn from Collective Human Opinions on Natural Language Inference Data?**

```
@inproceedings{ynie2020chaosnli,
	Author = {Yixin Nie and Xiang Zhou and Mohit Bansal},
	Booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
	Publisher = {Association for Computational Linguistics},
	Title = {What Can We Learn from Collective Human Opinions on Natural Language Inference Data?},
	Year = {2020}
}
```

**Distributed NLI: Learning to Predict Human Opinion Distributions for Language Reasoning**

```
@inproceedings{xzhou2022distnli,
	Author = {Xiang Zhou and Yixin Nie and Mohit Bansal},
	Booktitle = {Findings of the Association for Computational Linguistics: ACL 2022},
	Publisher = {Association for Computational Linguistics},
	Title = {Distributed NLI: Learning to Predict Human Opinion Distributions for Language Reasoning},
	Year = {2022}
}
```

## License

**Chaos NLI** is licensed under Creative Commons-Non Commercial 4.0. See the LICENSE file for details.
