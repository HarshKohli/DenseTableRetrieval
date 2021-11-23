# DenseTableRetrieval

Repository for Dense Table Retrieval given a Natural Language Question. 

# How to Run:

## Download Pretraining Data

The pretraining data for our code is derived from the [Turning Tables Repo](https://github.com/oriyor/turning_tables) Questions are synthetically created using the ExampleGenerator method described in [this paper](https://arxiv.org/abs/2107.07261)

1. Download the pretrain data from [here](https://drive.google.com/file/d/1tbF7RFkar3mmsA_cVgW5oSfnYbN6SHBX/view?usp=sharing)

2. Unzip its contents to the datasets/preprocessed directory

## Download finetuning data

We finetune and demonstrate results on the NQ-Tables dataset. This dataset consists of examples from the [Natural Questions] (https://ai.google.com/research/NaturalQuestions/) dataset where the answer is present within a table. About 11k such questions are isolated for our training.

1. Download the finetune data from [here](https://drive.google.com/file/d/1VClhPH01VO7RqsGf-7vac7Fkmgk8rU-f/view?usp=sharing)

2. Unzip its contents to the datasets/nq_data directory 

## Download Wikipeda Tables

Download 1.6m wikipedia tables data from [here](http://websail-fe.cs.northwestern.edu/TabEL/tables.json.gz) and copy the unzipped json to the datasets/ directory

## Run preprocess code

python preprocess.py
