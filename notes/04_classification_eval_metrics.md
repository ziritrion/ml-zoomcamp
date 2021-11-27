> Previous: [Classification / Logistic Regression](03_classification.md)

> [Back to Index](README.md)

> Next: [Deployment](05a_deployment.md)
>> Extra: [Virtual environments on Intel Macs](05b_virtenvs.md)

# Evaluation Metrics for Classification Models

## Accuracy and Dummy Model

On a _binary classification model_, it's possible to predict positive values that turned out to be negative, as well as negative values that turned out to be positive.

The _accuracy_ reflects the percentage of accurate predictions regardless of whether they were false negatives or positives.

A first approach to improving our accuracy would be to change the ***classification threshold***. We used `0.5` as a threshold before but a different threshold value may lead to better results.

By moving the threshold to both extremes (`threshold = 0` and `threshold = 1`) and training different models between both extremes, we can create ***dummy models*** for which we can calculate the accuracy and calculate which threshold value has the highest accuracy.

The threshold at both extremes can tell us interesting info about our model. In the _churn database_ example, a threshold of `1` turns out to have an accuracy of 73%; the maximum accuracy was with threshold `0.5` and turned out to be 80%, which is better, but not by a large margin.

By analyzing the dataset we find that it has severe ***class imbalance***, becase the proportion of `churn` clients to `no_churn` is about `3:1`.

Therefore, the accuracy metric in cases with class imbalance is misleading and does not tell us how well our model performs compared to a dummy model.

## Confusion tables

For binary classification, based on the prediction and the ground truth, there are 4 posible outcome scenarios:
* Ground truth positive, prediction positive > Correct > ***True positive***
* Ground truth positive, prediction negative > Incorrect > ***False positive***
* Ground truth negative, prediction posiive > Incorrect > ***False negative***
* Ground truth negative, prediction negative > Correct > ***True negative***

The ***confusion table*** is a matrix whose columns (x dimension) are the predictions and the rows (y dimension) is the ground truth:
     
     TP  FP
     FN  TP

Each position contains the element count for each scenario. We can also convert the count values to percentages.

## Precision and Recall

Based on the confusion table values, we can redefine _accuracy_ with the following formula:

* `accuracy = (TP + TN) / (TP + TN + FP + FN)`

We can also define additional metrics that will help us understand the data better, such as ***precision*** and ***recall***.

***Precision*** is the _fraction of positive predictions that are correct_. We only look at the positive predictions, and we calculate the fraction of correct predictions in that subset.

* `precision = TP / (TP + FP)`

***Recall*** is the _fraction of correctly identified positive examples_. Instead of looking at all of the positive predictions, we look at the _ground truth positives_ and we calculate the fraction of correctly identified positives in that subset.

* `recall = TP / (TP + FN)`

In the `churn` example model, the accuracy is of 80% but the precision drops to 67% and the recall is only 54%. That means that we failed to identify almost half of the churning customers (recall) and we're also misidentifying a third of customers as churning when they're not. Thus, the 80% accuracy value is very misleading.

## ROC curves

***ROC*** stands for ***Receiver Operating Characteristic***.

We begin by defining the ***False Positive Rate*** and ***True Positive Rate***:

* `FPR = FP / (TN + FP)`
* `TPR = TP / (TP + FN)`

Note that `TPR = recall`.

For _FPR_, we take a look at the negative ground truth and we calculate the fraction of false positives in the subset. In other words: FPR takes a look at one subset of the data and the TPR looks at the other "half" of the data.

We want the FPR to be as low and TPR to be as high as possible in any model.

If we try different thresholds and calculate confusion tables for each threshold, we can also calculate the TPR and FPR for each threshold.

When we plot the FPR (x axis) against the TPR (y axis), a random baseline model should describe an ascending straight diagonal line, a perfect model would increase inmediately to 1 and stay up, and our model most likely will be somewhere in between in a bow shape, ascending quickly at first and then decreasing the growth until it reaches the point (1,1), almost asymptotically.

A good model would be a very "arched" bow, as close as possible to the perfect model and as far away as possible form the diagonal.

## ROC AUC (Area Under the Curve)

The ***ROC AUC*** is the _area under the ROC curve_. The ROC AUC is a great metric of measuring performance.

For a random model with a diagonal ROC curve (worse scenario), the ROC AUC will be `0.5`. For a perfect model with a perfect ROC curve that instantly rises to `1` (best scenario), the ROC AUC will be `1.0`.

What the AUC actually is is the _probability of a random positive sample having a higher score than a random negative sample_.

## K-fold Cross Validation

***K-fold Cross Validation*** consists on evaluating the same model on different subsets of data. We set apart a test data subset but for the remaining data, we split it in `K` parts (_folds_) and we reserve some parts for training and others for validation, and then repeat the process multiple times for different permutations. We can then compute the AUC score for each permutation and then calculate the mean and standard deviation of all of them to get the average prediction and the spread within predictions.

> Previous: [Classification / Logistic Regression](03_classification.md)

> [Back to Index](README.md)

> Next: [Deployment](05a_deployment.md)
>> Extra: [Virtual environments on Intel Macs](05b_virtenvs.md)