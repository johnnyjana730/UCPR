# UCPR

UCPR: User-Centric Path Reasoning towards Explainable Recommendation, SIGIR 2021


# dataset

https://drive.google.com/file/d/1sJ18jvwhaEB-UACbWFNQ8vUZfvixgqC2/view?usp=sharing


## Introduction
todo

## Citation 
If you want to use our codes and datasets in your research, please cite:
todo

## Files in the folder

- `data/`: datasets
  - `MovieLens-1M/`
  - `amazon-book_20core/`
  - `last-fm_50core/`
  - `music/`
- `src/model/`: implementation of KBHP.
- `output/`: storing log files
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
