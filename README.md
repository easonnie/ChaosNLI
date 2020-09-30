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
Your repository file should be something like:
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

## Results
### How to reproduce the results on the paper?


## Evaluate
### I got a new method to produce a label distribution over each example in ChaosNLI. How can I score my method?

## Scoreboard

If you want your results to be showed on the Scoreboard, please email us (<yixin1@cs.unc.edu> or <nyixin318@gmail.com>) with **the name of the entry**, **a link to your method**, and **your model prediction file**.


## License
**Chaos NLI** is licensed under Creative Commons-Non Commercial 4.0. See the LICENSE file for details.