# ml-privacy-csf18
Code for the linear- and tree-model experiments in the CSF 2018 paper "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting"
* [Link to paper](https://ieeexplore.ieee.org/document/8429311)
* [Link to arXiv version](https://arxiv.org/abs/1709.01604)

### How to run the code
_Note: The Netflix Prize dataset is not included in this repository due to its size_

The code was (lightly) tested and should be compatible with both Python 2.7 and Python 3.6 on Bash.

To reproduce the results in the paper, clone the repository and go to the `code/` directory.
Then, run `./run-errors.sh` to compute the training and test errors.
**This is required before running any of the other experiments.**
The script should take about 1-2 hours to run.

##### Membership Inference Attack
Run `./run-membership.sh`.
The results will be found in `results-sklearn/iwpc/membership/` and `results-sklearn/eyedata/membership/`.

##### Attribute Inference Attack
Run `./run-attribute.sh cyp2c9` and `./run-attribute.sh vkorc1`.
The results will be found in `results-sklearn/iwpc/attribute/`.

##### Reduction from Membership Inference to Attribute Inference
Run `./run-reduction.sh cyp2c9` and `./run-reduction.sh vkorc1`.
The results will be found in `results-sklearn/iwpc/reduction/`.