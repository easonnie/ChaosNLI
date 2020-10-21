# Chaos NLI
What Can We Learn from **C**ollective **H**um**A**n **O**pinion**S** on **N**atural **L**anguage **I**nference Data (**ChaosNLI**)?  
\[[bib](https://github.com/easonnie/ChaosNLI#citation)\] (EMNLP 2020)

**Outline**
* [Paper](#paper)
* [Motivation](#motivation)
* [Examples](#examples)
    * [NLI Examples](#nli-example)
    * [Abductive NLI Examples](#abductive-nli-example)
* [Scoreboard](#scoreboard)
    * [ChaosNLI - SNLI](#chaosnli---snli)
    * [ChaosNLI - MNLI](#chaosnli---mnli)
    * [ChaosNLI - Abductive NLI](#chaosnli---abductive-nli)
* [Data and Format](#data-and-format)
    * [Where can I download the data](#where-can-i-download-the-data)
    * [What is the format?](#what-is-the-format)
        * [Data Format](#data-format)
        * [Model Prediction Format](#model-prediction-format)
* [Results](#results)
    * [How to reproduce the results on the paper?](#how-to-reproduce-the-results-on-the-paper)
* [Evaluate](#evaluate)
    * [How can I evaluate my own results?](#how-can-i-evaluate-my-own-results)
* [Citation](#citation)
* [License](#license)


## Paper
[What Can We Learn from Collective Human Opinions on Natural Language Inference Data?](https://arxiv.org/abs/2010.03532)

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
Amy and her friends were out at 3 AM.|They started getting followed by a policeman, ran, and hid behind a building.|The decided to break into the football field. When suddenly they saw a flashlight comming towards them. They all started running for the bleachers.|They stayed there breathing hard, and praying they hadn't been seen.|1(50), 2(50)|2|1(57.53%), 2(32.47%)

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
**ChaosNLI** is available at https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip.  

Alternatively, you can download the data with the following script:
```bash
# make sure you are at the root of chaos_nli directory
source setup.sh     # setup path
bash scripts/download_data.sh
```
The script will not only download the ChaosNLI data and but also the predictions of model in the `chaos_nli/data`.  

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
We also provide the predictions of BERT, RoBERTa, XLNet, BART, ALBERT, DistilBERT on SNLI, MNLI, and alphaNLI. The data can also be found in `data/model_predictions`.
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
To reproduce the results, simply run the following script:
```bash
source setup.sh
python src/evaluation/model_pref.py
```

The outputs should match with all the numbers in the table of the paper.
```
Load Jsonl: /Users/yixin/projects/released_codebase/chaos_nli/data/chaosNLI_v1.0/chaosNLI_alphanli.jsonl
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
Load Jsonl: /Users/yixin/projects/released_codebase/chaos_nli/data/chaosNLI_v1.0/chaosNLI_snli.jsonl
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
Load Jsonl: /Users/yixin/projects/released_codebase/chaos_nli/data/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl
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

To examine **the factor of human agreement on model performance**, check out this informative [jupyter-notebook](https://github.com/easonnie/ChaosNLI/blob/master/src/notebook/binned_plot.ipynb) to find the partitioned results according to the entropy range. 

## Evaluate
### How can I evaluate my own results?
Build your prediction file according to the sample file in `data/prediction_samples`.  
**Tip: You will need to popularize both the "`predicted_probabilities`" and "`predicted_label`" fields.**

Then, you can evaluate your method with the following script:
```
# For MNLI:
python src/scripts/evaluate.py \
    --task_name uncertainty_nli \ 
    --data_file [path_of_chaos_nli_repo]/chaos_nli/data/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl \
    --prediction_file [path_of_chaos_nli_repo]/chaos_nli/data/prediction_samples/mnli_random_baseline.json

# For SNLI:
python src/scripts/evaluate.py \
    --task_name uncertainty_nli \
    --data_file [path_of_chaos_nli_repo]/chaos_nli/data/chaosNLI_v1.0/chaosNLI_snli.jsonl \
    --prediction_file [path_of_chaos_nli_repo]/chaos_nli/data/prediction_samples/snli_random_baseline.json

# For alphaNLI:
python src/scripts/evaluate.py \
    --task_name uncertainty_abdnli
    --data_file [path_of_chaos_nli_repo]/chaos_nli/data/chaosNLI_v1.0/chaosNLI_alphanli.jsonl
    --prediction_file [path_of_chaos_nli_repo]/chaos_nli/data/prediction_samples/abdnli_random_baseline.json
```

## Citation
```
@inproceedings{ynie2020chaosnli,
	Author = {Yixin Nie and Xiang Zhou and Mohit Bansal},
	Booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
	Publisher = {Association for Computational Linguistics},
	Title = {What Can We Learn from Collective Human Opinions on Natural Language Inference Data?},
	Year = {2020}
}
```

## License
**Chaos NLI** is licensed under Creative Commons-Non Commercial 4.0. See the LICENSE file for details.