# AI homework 9

author:高梓源   stu. Num: 2019K8009929026 University: UCAS

---

## Usage

### Independence

You should have `python >3.6` at best, the developping platform is on `python 3.8.5`

For `python` library, you should have `numpy, json, os, sys`

### Run program

After download the repository, enter the direcoty `PLSA`

First, run `python preprocess.py text.txt data` to preprocess raw data.

Then run `python PLSA.py data` to run PLSA algorithm on data.

We want you to enter some arguments for the model:

- `topic_num`: number of topics, super argument
- `equ_epsilon`: control when the likelihood converge, should be `0~15`
- `max_itertime`: maximum iteration time of PLSA
- `hot_words_num`: print how many hot words as output to see the result of algorithm

all default argument number is 10, if you enter wrong input, we will use 10 as your argument.

current result is based on arguments: 9, 12, 30, 10, only upload `topic_top_words.txt`

## Copyright

This is a good complemet of PLSA algorithm with **NO** `for` sentence during calculation, using `numpy` and `lambda` to substitude.

You are allowed to borrow thinkings, and if you want to use code for presentation, please cite me as `UCAS 高梓源`