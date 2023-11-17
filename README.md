# Finding Support Examples for In-Context Learning (Findings of EMNLP 2023)

The official repository for paper "Finding Support Examples for In-Context Learning" (EMNLP 2023 Findings). 
Check out our [paper](https://arxiv.org/pdf/2302.13539.pdf) for more information.

## Datasets

Download the [train_data](https://drive.google.com/file/d/1mPeiiBXxervQ9vwBvH-whPRN1KPCExN3/view?usp=sharing) and [test_data](https://drive.google.com/file/d/1uIRfXELtqKm4zyVY7dOjp4ZhJhUY2rZy/view?usp=sharing) from Google Drive, and put them in the current folder.

## Installation

```
conda create -n icl_support_examples python=3.8
conda activate icl_support_examples
conda install pytorch=1.7.1 -c pytorch
pip install transformers==4.3.0
pip install fitlog
```

## Run Our Method

```
# dataset=[sst2,sst5,amazon_b,mr,subj,agnews,trec,dbpedia]
CUDA_VISIBLE_DEVICES=0 bash commands/run_[dataset].sh
```

and then see the performance through [fitlog](https://github.com/fastnlp/fitlog) as:

```
fitlog log fitlog_search_examples
```
Then see fitlog in your browser.

## Citation
```
@article{icl_support_example,
  author       = {Xiaonan Li and
                  Xipeng Qiu},
  title        = {Finding Supporting Examples for In-Context Learning},
  journal      = {CoRR},
  volume       = {abs/2302.13539},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2302.13539},
  doi          = {10.48550/ARXIV.2302.13539},
  eprinttype    = {arXiv},
  eprint       = {2302.13539},
  timestamp    = {Tue, 28 Feb 2023 14:02:05 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2302-13539.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```