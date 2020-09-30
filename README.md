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
**ChaosNLI** is in `JSONL`. Everyone line in the file is one single JavaScript Object and can be easily loaded in python dictionary.  
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
    "old_label": 1                                      // The old label. (AlphaNLI only have one original label for each example.)
}
```

#### Model Prediction
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


## Results
### How to reproduce the results on the paper?


## Evaluate
### I got a new method to produce a label distribution over each example in ChaosNLI. How can I evaluate my method?

## Scoreboard

If you want your results to be showed on the Scoreboard, please email us (<yixin1@cs.unc.edu> or <nyixin318@gmail.com>) with **the name of the entry**, **a link to your method**, and **your model prediction file**.


## License
**Chaos NLI** is licensed under Creative Commons-Non Commercial 4.0. See the LICENSE file for details.