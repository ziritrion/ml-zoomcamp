> Previous: [Linear Regression](02_linear_regression.md)

> [Back to Index](README.md)

> Next: [Evaluation metrics for Classification models](04_classification_eval_metrics.md)

# Binary Classification (Logistic Regression)

* `g(x_i) ≈ y_i`
* `y_i ∈ {0,1}`

## (Aside) Feature importance

### Difference and Risk Ratio

From the Telco churn rate exercise, let's take churn rate as the feature we're studying.

* `global` = the total population of the feature
* `group` = a filtered subset of the feature's population

1. Difference:
    * `global - group`
    * `difference < 0` -> more likely to churn.
    * `difference > 0` -> less likely to churn.
2. Risk ratio
    * `group / global``
    * `risk > 1` -> more likely to churn.
    * `risk < 1` -> less likely to churn.

### Mutual information

The ***mutual information*** of 2 random variables is a measure of the mutual dependence between them.

In Scikit-Learn, in the Metrics package, the `mutual_info_score` method allows us to input 2 features and it will output the mutual information score.

The score can be between `0` and `1`. The closest to `1`, the more important the feature is.

### Correlation

The ***correlation coefficient*** measures the linear correlation between 2 sets of data -> ratio between the covariance of 2 variables and the product of their standard deviations `𝝈`. In other words, it's a normalized covariance.

* `r` (also sometimes `𝝆`) = correlation coefficient.
* The value of `r` is always in the interval `[-1 ,1]`.
* If `r` is negative, when one of the variables grows, the other one decreases.
* If `r` is possitive, when one of the variables grows, the other one does as well.
* Values between `|0.0|` and `|0.2|`, the correlation is very low and growth/decrease is very softly reflected on the other variable.
* Values between `|0.2|` and `|0.5|` show moderate correlation.
* Values between `|0.5|` and `|1.0|` show strong correlation.

## Logistic Regression

In Logistic Regression, the model `g(x_i)` will return a number between the values `[0,1]`. We can understand this value as the ***probability*** of `x_i` belonging to the "positive class"; if the value were `1` then it would belong to this class, but if it were `0` it would belong to the opposite class of our binary classification problem.

* `g(x_i) = sigmoid( w_o + w^T · x_i )`
* Logistic Regression is similar to Linear Regression except that we wrap the original formula inside a _sigmoid_ function. The sigmoid function always returns values between `0` and `1`.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

## Accuracy

We can check the accuracy of the model by comparing the predictions with the target (in other words, the error of our predictions) and calculating the mean of the error array. Even if the comparison vector is made of Booleans, NumPy will automatically convert them to `1`'s and `0`'s and calculate the mean.

## Logistic Regression workflow recap

1. Prepare the data
    1. Download and read the data with pandas
    1. Look at the data
    1. Clean up the feature/column names
    1. Check if all the columns read correctly (correct types, no NaN's, convert categorical target into numerical, etc)
    1. Check if the target data needs any preparation
1. Set up the validation framework (splits) with scikit-learn
1. Exploratory Data Analysis
    1. Check missing values
    1. Look at the target variable
        * Look at the distribution; use `normalize` for ease.
    1. Look at numerical and categorical variables
    1. Analyze feature importance
        * Difference and risk ratio
        * Mutual information
        * Correlation
1. Encode categorical features in one-hot vectors
1. Train the model with Logistic Regression
    1. Keep the prediction probabilities rather than the hard predictions if you plan on modifying the thresholds.
    1. Calculate the accuracy of the model with the validation dataset.
1. Interpret the model
    1. Look at the coefficients
    1. Train a smaller model with fewer features
1. Use the model
    * Combine the train and validation datasets for your final model and test it with the test dataset.

> Previous: [Linear Regression](02_linear_regression.md)

> [Back to Index](README.md)

> Next: [Evaluation metrics for Classification models](04_classification_eval_metrics.md)