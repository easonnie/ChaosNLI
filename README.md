# Chaos NLI
What Can We Learn from **C**ollective **H**um**A**n **O**pinion**S** on **N**atural **L**anguage **I**nference Data (**ChaosNLI**)?

## Paper
[What Can We Learn from Collective Human Opinions on Natural Language Inference Data?](https://www.google.com)

## Data and Format
### Where can I download the data？
**ChaosNLI** is available at https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip.  

Alternatively, you can download the data with the following script:
```bash
# make sure you are at the root of chaos_nli directory
source setup.sh     # setup path
bash scripts/download_data.sh
```
The script will download the data and the predictions of model in the `chaos_nli/data`.  

If you want to use the scripts in this repository to reproduce the results, please make sure the data is downloaded in the correct path.  
The repository structure should be something like:
```
├── LICENSE
├── README.md
├── data
│   ├── chaosNLI_v1.0
│   │   ├── README.txt
│   │   ├── chaosNLI_alphanli.jsonl
│   │   ├── chaosNLI_mnli_m.jsonl
│   │   └── chaosNLI_snli.jsonl
│   └── model_predictions
│       ├── model_predictions_for_abdnli.json
│       └── model_predictions_for_snli_mnli.json
├── requirements.txt
├── scripts
├── setup.sh
└── src
```

### What is the format?
#### Data
ChaosNLI is in `JSONL`. Everyone line in the file is one single JavaScript Object and can be easily loaded in python dictionary.  
The fields of the objects are self-explanatory. Please see the following sample to learn the details about the field.
```JS
// ChaosNLI-(S+M)NLI

{   "uid": "46359n",                                // The unique id of this example. This is the same id as the one in MNLI pairID field and can be used to map back to the MNLI data.                            
    "label_counter": {"e": 76, "n": 20, "c": 4},    // The dictionary indicates the count of each label. e=entailment; n=neutral; c=contradiction.
    "majority_label": "e",                          // The majority label of the new ChaosNLI labels. (This might be different from the old label.)
    "label_dist": [0.76, 0.2, 0.04],                // The label distributions. (The order matters and should always be: E, N, C.)
    "label_count": [76, 20, 4],                     // The label counts. (The order matters and should always be: E, N, C.)
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

```
ChaosNLI-alphaNLI

```
#### Model Prediction


## Results
### How to reproduce the results on the paper?


## Evaluate
### I got a new method to produce a label distribution over each example in ChaosNLI. How can I evaluate my method?

## Scoreboard

If you want your results to be showed on the Scoreboard, please email us (<yixin1@cs.unc.edu> or <nyixin318@gmail.com>) with **the name of the entry**, **a link to your method**, and **your model prediction file**.


## License
**Chaos NLI** is licensed under Creative Commons-Non Commercial 4.0. See the LICENSE file for details.