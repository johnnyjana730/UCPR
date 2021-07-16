# UCPR

UCPR: User-Centric Path Reasoning towards Explainable Recommendation, SIGIR 2021

This repository is the implementation of UCPR ([ACM](https://dl.acm.org/doi/10.1145/3404835.3462847)):

![image](https://user-images.githubusercontent.com/20666568/125889563-c0979eb7-fb67-44b2-af05-d1dce354bf7e.png)

# dataset

https://drive.google.com/file/d/1sJ18jvwhaEB-UACbWFNQ8vUZfvixgqC2/view?usp=sharing


## Introduction
We propose UCPR, a user-centric path reasoning network that constantly guides the search from the aspect of user demand and enables explainable recommendations. 

## Citation 
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{10.1145/3404835.3462847,
author = {Tai, Chang-You and Huang, Liang-Ying and Huang, Chien-Kun and Ku, Lun-Wei},
title = {User-Centric Path Reasoning towards Explainable Recommendation},
year = {2021},
isbn = {9781450380379},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3404835.3462847},
doi = {10.1145/3404835.3462847},
booktitle = {Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {879–889},
numpages = {11},
keywords = {recommendation system, knowledge graphs, path reasoning, explainable recommendation, reinforcement learning},
location = {Virtual Event, Canada},
series = {SIGIR '21}
}
```
## Files in the folder

- `data/`: datasets
  - `amazon-book_20core/`
  - `Amazon_Beauty_Core/`
  - `Amazon_Cellphones_Core/`
  - `Amazon_Clothing_Core/`
  - `MovieLens-1M_Core/`
- `src/model/`: implementation of UCPR.
- `eval/`: storing log files
- `misc/`: storing users being evaluating, popular items, and sharing embeddings.

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* torch == 1.8.0
* numpy == 1.15.4
* scipy == 1.1.0
* sklearn == 0.20.0

## Build Environment(conda)
```
$ cd UCPR
$ conda deactivate
$ conda env create -f requirements.yml
$ conda activate UCPR
```

## Example to Run the Codes

* UCPR
```
$ cd bash
$ bash bash_run.sh $dataset $gpu
```

* `dataset`
  * It specifies the dataset.
  * Here we provide three options, including  * `az`, `mv`, `bu`, `cell`, or `cl`.

* `gpu`
  * It specifies the gpu, e.g. * `0`, `1`, and `2`.
