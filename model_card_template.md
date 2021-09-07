# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a Random Forest Classifier from scikit-learn. The Hyperparameters has been determined after a Randomized search on hyperparameters and can be found in the ml folder.

## Intended Use
This model should be used to predict the whether a person makes over 50K a year based off a handful of attributes. 

## Metrics
The model was optimized for the F1 score. The value on the test set of the F1 score is 0.8, the recall is 0.53 and the precision is 0.60

## Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). Among 14 attributes, only 6 categorical attributes have been used (workclass, education, marital-status, occupation, relationship, race, sex, native-country).

The original data set has 48842 rows, and a 80-20 split was used to break this into a train and test set. The validation has been done with a 5-fold cross-validation. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Ethical Considerations
The model discriminates on race, gender and origin country. In additional to probable unfairness, using such features could be unethical, if not illegal in some production settings.

## Caveats and Recommendations
The data comes from the 1994 census database and it is no longer representative of the current situation of the US society.
In addition to the ethical considerations mentionned above, this model should not be used in production.

Regarding research recommendations, other model and additional features should be considered. Bias could also be researched with the library Aequitas per example.
