# Linear regression

## General formula for ML

* `g(X) ≈ y`
* `g` = model
* `X` = feature matrix
* `y` = target

For our Linear Regression example, `g` will be Linear Regression model, our `y` target will be Price.

## Simplified model (one single observation)

* `g(Xᵢ) ≈ yᵢ`
* `Xᵢ` = a car
* `yᵢ` = price of the car
* `Xᵢ = (xᵢ₁, xᵢ₂, ..., xᵢₙ)`
* Each `xᵢⱼ` is a characteristic (feature) of the car.

Thus

* `g(xᵢ₁, xᵢ₂, ..., xᵢₙ) ≈ yᵢ`

For our example, we'll pick 3 features: horse power, milles per gallon and popularity.

* `Xᵢ = [453, 11, 86]`

Here is the formula for our Linear Regression.

* `g(Xᵢ) = w₀ + w₁·xᵢ₁ + w₂·xᵢ₂ + w₃·xᵢ₃`
* `w₀` is the ***bias term*** weight.
* All other `wⱼ` are weights for each of the features.

Alternatively:

* `g(Xᵢ) = w₀ + ∑( wⱼ·xᵢⱼ, j=[1,3] )`

Depending on the values of the weights, our predicted price `yᵢ` will be different.

## Linear Regression in vector form

Linear Regression formula for `n` features:

*  `g(Xᵢ) = w₀ + ∑( wⱼ·xᵢⱼ, j=[1,n] )`

The `∑( wⱼ·xᵢⱼ, j=[1,n] )` term is actually a ***dot product***. Thus:

* `g(Xᵢ) = w₀ + Xᵢᵀ · W`
* `Xᵢᵀ` is the transposed feature vector.
* `W` is the weights vector.

We can make this formula even shorter by incorporate the bias term `w0` to our dot product, "simulating" a new feature `xi0` which is always equal to one.

* `W = [w₀, w₁, w₂, ... , wₙ]`
* `Xᵢ = [1, xᵢ₁, xᵢ₂, ... , xᵢₙ]`
* `Wᵀ · Xᵢ = Xᵢᵀ · W`

We have now converted the complete Linear Regression formula into a dot product of vectors.

Generalizing for multiple observations, `X` now becomes a matrix of size `m * n` where `n` is the number of features just like before, and `m` is the number of observations. Each row of `X` is an observation, identical in form to our previous `Xᵢ`.

        1   x₁₁   x₁₂   ...   x₁ₙ
        1   x₂₁   x₂₂   ...   x₂ₙ
        ... ...   ...   ...   ...
        1   xₘ₁   xₘ₂   ...   xₘₙ 

We can now multiply `X` with `W` to get our predictions.

        1   x₁₁   x₁₂   ...   x₁ₙ       w₀        X₁ᵀ·W
        1   x₂₁   x₂₂   ...   x₂ₙ   ·   w₁    =   X₂ᵀ·W 
        ... ...   ...   ...   ...       ...       ...
        1   xₘ₁   xₘ₂   ...   xₘₙ       wₙ        Xₘᵀ·W

The resulting vector `Ŷ` is the prediction vector.

But how do we calculate the weights?

## Normal equation

* `g(X) = X·W ≈ y`

We want to solve for `W`. We invert `X` and use it to solve the equation:

* `X⁻¹ · X · W = X⁻¹ · y`
* `X⁻¹ · X = I`

Thus:
* `I · W = X⁻¹ · y` -> `I` is the Identity Matrix and does not change `W`.
* `W = X⁻¹ · y`

However, `X` may not be a square matrix and there is no guarantee that `X⁻¹` will exist at all. However, there is a workaround using transposed matrices:

* `Xᵀ · X` -> Gram matrix. A Gram matrix is ALWAYS square because it's of size `(n+1) * (n+1)`. Thus, it can be inverted.

We can now calculate the inverse of the Gram matrix and use it to solve our equation:

* Starting from `Xᵀ · X · W = Xᵀ · y`, we use the inverse of the Gram matrix to cancel out the terms.
* `(Xᵀ · X)⁻¹ · Xᵀ · X · W = (Xᵀ · X)⁻¹ · Xᵀ· y`

Thus, knowing that `(Xᵀ · X)⁻¹ · Xᵀ  · X = I` and can therefore be cancelled out, we finally get the closest solution possible for `W`:

* `W = (Xᵀ  · X)⁻¹ · Xᵀ  · y`

## RMSE (Root Mean Square Error)

MSRE is a convenient way to measure the accuracy (or the error) of our model.

* `RMSE = √( 1/m * ∑( (g(Xᵢ) - yᵢ)²	 , i=[1,m] ) )`
* `g(Xᵢ)` is the prediction for `Xᵢ`.
* `yᵢ` is the actual value.

The lower the RMSE, the more accurate the model is.

## Categorical features

Some features are categorical rather than numerical in nature. For example, `Car_maker` would be a categorical feature.

We can encode these features as ***one-hot vectors***.  A one-hot vector is a vector with as many elements as elements there are in a category, and all of the elements are `0` except for a single `1` representing a particular element.

* `category = ['cat', 'dog', 'bird']`
* `cat = [1,0,0]`
* `dog = [0,1,0]`
* `bird = [0,0,1]`

In pandas, we simply add a new feature to the DataFrame for each category element with value `0` or `1`. For the example above, we would add 3 columns to the DataFrame.

## Regularization

Sometimes there are features which are _linear combinations_ of other features (sum/product of other columns). This can lead to columns in the `X` matrix which are identical. The consequence of this is that the Gram matrix `Xᵀ · X` becomes a _singular matrix_ and thus cannot be inverted.

In the case of noisy data which can lead to almost-but-not-quite identical features, the Gram matrix is invertable but the resulting values within are disproportionally big. This distorts the training and leads to huge errors.

We can solve this with ***regularization***. We will use a _regularization parameter_ that will modify our normal equation in a way that will result in greatly reduced weights.

For Linear Regression, regularization is applied to the diagonal of the Gram matrix. The regularization parameter is simply added to the diagonal.

The regularization parameter is usually a small decimal value such as `0.00001` or `0.01`. The larger the parameter, the smaller the final values in the inverted matrix will be. However, a very big regularization parameter can lead to worse performance than smaller parameters.

In numpy, we can implement regularization easily by creating an identity matrix with `np.eye()`, multiplying it with our regularization parameter and finally adding the resultant matrix to the Gram matrix.

## Tuning the model

We want to find the best value for the regularization parameter. For Linear Regression, we can simply try multiple parameters in a for loop during training because the computational cost is low.

## Linear Regression workflow recap

1. Explore the data
    1. Understand the target distribution.
    1. Find out which changes you need to do to it in order to use it.
1. Clean up the data.
    1. Do transformations on it such as getting rid of spaces, lower case everything, fill in the NaNs, etc.
1. Prepare the data.
    1. Apply feature engineering -> convert categorical features into numerical ones with one-hot encoding.
    1. Shuffle and split the data into train-validation-test splits.
    1. Create the feature matrix for each split; make sure that you add the "virtual bias" column to it.
1. Train the model.
    1. Use the normal equation to calculate the weights.
    1. Make sure you apply a regularization parameter.
1. Tune the model
    1. Use RMSE to check accuracy.
    1. Predict values with the validation dataset and compare with the ground truth.
    1. Adjust regularization parameter accordingly.
    1. Plot histogram for easy visual check.
1. Test your model with the test dataset.
1. Use your model!