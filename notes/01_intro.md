> [Back to Index](README.md)

> Next: [Linear Regression](02_linear_regression.md)

# Introduction to Machine Learning (ML)

Let's suppose you want to sell a car. Depending on your car's make, model, age, mileage, etc., you can sell it for a certain price. An expert can look at the car's _features_ and determine a price based on them. In other words, the expert took _data_ and extracted _patterns_ from that data.

**Machine Learning** is a technique which allows us to build **models** that extract these patterns from data, just like the expert in our example.

* **Features** are the _characteristics_ of the data we've got (year, make, mileage, etc).
* The **target** is a feature we want to predict.
* A **model** is a "description of statistical patterns" that predicts a target given some input. Models are ***trained*** with ***algorithms*** that take some input features as well as reference targets for those features. The algorithms then extract _patterns_ that calculate the target given the feature inputs within some error margin, and those patterns are stored in the model.

Once we've trained a model, we can use it to process new completely original input and predict the target for the input's features.

# ML vs Rule-Based Systems

In the traditional programming paradigm, the developer defines how a system will behave by defining specific rules. However, for complex or ever-changing behaviors, this method can become unsustainable or even impossible.

For example: we can try to create a spam filter by using specific rules, such as filtering words, blocking certain senders, etc., but human language is so complex and spam changes so quickly that it's impossible to keep up and our filter would be obsolete inmediately and would never work with acceptable effectiveness.

ML offers a solution to this issue:
* We can **gather data** (in our example, emails, both regular email and spam) to create a _dataset_.
* We can **define and calculate the features** which are relevant to our dataset and the problem we're trying to solve.
* Finally, we can **train and use a model** which is able to recognize the patterns that distinguish regular email from spam, allowing us to act on it by filtering spam.

ML does not necessary discard all Rule-Based Systems. We could use (some of) the rules defined on a Rule-Based System and use them as features for our ML model. Following the spam filter example: a feature could be whether the sender is from a specific domain, or whether the subject contains certain words.

Essentially, ML is a ***paradigm shift*** compared to traditional programming. Traditional programming follows this structure:

`data + code => outcome`

But ML changes this equation and becomes like this:

`data + outcome => model`

And the resulting `model` allows us to replace `code` in the original equation:

`data + model => outcome`

# Types of ML

The 2 examples shown above belong to a class of techniques called **Supervised Learning**. They're called _supervised_ because they rely on data for which a known target exists (often referred to as ***labels***), thus giving a reference to the ML algorithms in order to predict data.

Other types exist such as _Unsupervised Learning_ or _Reinforcement Learning_, but they fall outside the scope of this course.

# Understanding Supervised ML

## Notation

Given a set of features (data) and a target (desired output), we need to process them in order to obtain a model.

We can structure our features into a ***feature matrix*** **`X`**, a two-dimensional matrix where every row is an _observation_ or _object_, and every column is a _feature_. Each observation has a target, so we can place our targets in a ***target vector*** **`y`** and arrange it as a _column matrix_.

       features  (X)      target (y)
            f1 f2 f3 f4
    obs. 1: 1  1  0  1     1
    obs. 2: 1  0  0  1     0
    obs. 3: 0  1  0  0     0
    obs. 4: 1  1  1  1     1

We can now define our model as a _function_ `g` that takes our feature matrix `X` and outputs a value which is approximate to our target `y`:
* `g(X) ≈ y`

## Problem types

The problems that Supervised ML can solve van be classified into 2 main types: ***regression*** and ***classification*** problems. A third type is sometimes considered, ***ranking***; however, all the approaches to solving this type can be classified under regression or classification, depending on the method.

* **Regression problems** are problems whose targets are _continuous values_. In other words, we're trying to predict numbers, such as price, for example.
* **Classification problems** are problems whose targets are _categories_. The spam filter example above was an example of a classification problem in which we're trying to predict whether a given email belongs to the spam category or not.
    * **Multiclass classification** is when the target consists of multiple categories. For example, trying to predict whether an animal is a cat, a dog or a bird.
    * **Binary classification** is when the target consists of only 2 categories, which are often positive and negative. For example, the spam filter problem is a binary classification problem.

# CRISP-DM

**CRISP-DM** (_cross-industry standard process for data mining_) is a methodology for organizing ML projects. Even though it was created in the 90's, it's still relevant today and describes the steps to carry out a successful ML project.

![crips-dm](images/01_d01.png)

1. **Business understanding**
    * Analyze the problem. How serious is it and to what extent? Is it just one user complaining or a company-wide issue?
    * Will Machine Learning help? Can the problem be solved with easier methods?
    * Define a **goal** to achieve. The goal must be **measurable**.
        * For the spam example, the goal could be _reduce the amount of spam messages_, or perhaps _reduce the amount of complaints about spam_.
        * Our measue will be to _reduce the amount of spam by 50%_.
1. **Data undestanding**
    * Once you've decided on using ML, analyze and identify available data sources and decide if we need to get more data.
        * Spam example: do we have a _report spam_ button? Is the data behind this button good enough? Is the button reliable? Do we track spam correctly? Is our dataset large enough? Do we need to get more data?
    * Understanding the data may give us new insights into the problem and influence the goal. We may go back to the Business Understanding step and adjust it accordingly.
1. **Data preparation**
    * Transform the data so it can be put into a ML algorithm.
        * Clean the data (remove noise)
        * Build the pipelines (raw data -> transformations -> clean data)
        * Convert into tabular form
1. **Modeling**
    * Training the models. The actual Machine Learning happens here.
        * Try different models
        * Select the best one
    * Which model to choose?
        * Logistic regression
        * Decission tree
        * Neural network
        * Many others!
    * Sometimes we may have to go back to the Data Preparation step to add new features or fix data issues.
1. **Evaluation**
    * Measure how well the model solves the business problem.
    * Is the model good enough?
        * Have we reached our goal?
        * Do our metrics improve?
    * Do a retrospective:
        * Was the goal achievable?
        * Did we solve/measure the right thing?
    * After the retrospective, we may decide to:
        * Go back to the Business Understanding step and adjust the goal
        * Roll the model to more/all users
        * Stop working on the project (!)
1. **Evaluation + Deployment**
    * Often the 2 steps happen together:
        * Online evaluation: evaluation of live users
        * It means that we deploy first and then evaluate it on a small percentage of users
    * This is where modern practices differ slightly from the original CRISP-DM methodology.
1. **Deployment**
    * Roll the model to all users
    * Proper monitoring
    * Ensuring quality and maintainability
    * Essentially, this is the "engineering" step.
1. **Iterate!**
    * ML projects require many iterations! This also differs from the original CRISP-DM
    * After Deployment, we may go back once again to Business Understanding and wonder whether the project can be improved upon

Additional iteration guidelines:
1. Start simple
1. Learn from feedback
1. Improve

# Modeling step: model selection

## Evaluation and accuracy

In order to choose a model among many others, we need ways to evaluate how good a given model is.

This is done by _splitting the dataset_ into a **training dataset** and a **validation dataset**. This split is usually done as a 80/20 split for training/validation. The validation dataset is "hidden" from the model and we only use the training dataset for training.

We can extract a feature matrix `X` and a target `y` from the training dataset and train a model `g`. We can then extract another feature matrix `Xv` and target `yv` from the validation dataset, and then obtain **predictions**:

`g(Xv) = ŷv`

Finally, we compare our prediction `ŷv` with out target `yv` and check how different they are, thus obtaining an **accuracy score**.

By comparing accuracy scores between our models, we can decide which model is better for our problem.

However, we must be wary of the **multiple comparisons problem** (AKA the _look-elsewhere effect_). This problem is defined as:
* While the chance of noise affecting one result may be small, the more measurements we make, the larger the probability that a random fluctuation is missclassified as a meaningful result.

In ML, this means that when we use multiple models and compare results, one model may get "lucky" and show significant better results than other just because the data split was favorable to that model.

## Solving the multiple comparisons problem

In order to protect our models against the multiple comparisons problem, we split the original dataset into 3 parts:
* 60% training
* 20% validation
* 20% **test**

We now create the models following these steps:
1. Train the models with the training dataset
1. Validate the models with the validation dataset and select the best one
1. Finally, test the chosen model with the test dataset. If the accuracy score is similar to the scores obtained during evaluation, then the model is likely good. If the result however is very different, then we got a "lucky" model and we must discard it.

## Model selection final process

1. Split the original dataset into train/validation/test datasets
1. Train a model
1. Validate the model. Go back to step 2 and repeat as many times as models we want to train.
1. Select the best model
1. Test the model with the test data.
1. Check if the results are satisfactory

Usually, between steps 4 and 5, we will combine the training and validation datasets in order to create a "_full training_" dataset and retrain our best model with it. This way, the validation dataset split isn't wasted after step 3. This should also help improve the model a little bit.

# NumPy

**NumPy** is a Python library which adds support for large, multi-dimensional arrays and matrices, along with many high-level functions to operate on them. NumPy is used extensively in ML.

Please check the [Python and libraries cheatsheet](https://gist.github.com/ziritrion/9b80e47956adc0f20ecce209d494cd0a#numpy) for reference.

# Linear Algebra

ML makes use of many algebraic concepts and operations due to its extensive use of arrays and matrices. Here is a quick refresher.

## Linear combination operations

* Scalar product
    * Product of a real number and a 1-D vector. Results in a new 1-D vector.
    * `k * v = k * [v1, v2, v3] = [k * v1, k * v2, k * v3]`
    * Called "scalar product" because the resulting vector is a scaled version of the original vector.
    * In linear algebra notation, the vector is often represented as a _column vector_.
    * NumPy: `np.multiply(k, v)`
    * Python: `k * v`
* Vector addition
    * Addition of each component of 2 1-D vectors Results in a new 1-D vector.
    * `u + v = [u1, u2, u3] + [v1, v2, v3] = [u1 + v1, u2 + v2, u3 + v3]`
    * Geometrically, it's as if we took one of the vectors and made it start from the point where the other vector ends. The line between the point of origin of the first vector and the end of the second vector is the resulting vector.
    * Both vectors are alos represented as column vectors.
    * NumPy: `np.add(u,v)`
    * Python: `u + v`
* Combining scalar products and vector additions is called a **linear combination of vectors**.

## Products

* Dot product (vector-vector product)
    * Product of 2 1-D vectors. Results in a scalar.
    * `u · v = [u1 , u2, u3] ⋅ [v1, v2, v3] = u1v1 + u2v2 + u3v3`
    * Geometrically, it can be understood as the multiple of the lengths of the 2 vectors and the angle between them.
    * In linear algebra notation, the first vector is represented as a row vector and the second as a column vector.
    * Both vectors must be of the same length.
    * NumPy: the operation can be done with `np.dot(u,v)`
    * Python (v3.5+): `u @ v`
* Matrix-vector product
    * Product of a 2-D matrix and a 1-D vector. Results in a 1-D vector.
    * In algebra notation, the matrix goes first and the vector second, displayed as a column matrix.
    * Essentially it's a dot product of each row of the matrix with the vector. Each dot product result is a component of the result vector.
    * The matrix must have as many columns as the vector has components.
    * For a matrix of size `(n,k)` and a vector of size `(k)`, the resulting vector will be of size `(n)`.
    * NumPy: `np.dot(U,v)` can also handle matrix-vector multiplications.
    * Python (3.5+): `U @ v`
* Matrix-matrix product
    * Product of 2 2-D matrices. Results in a 2-D matrix.
    * The first matrix must have as many columns as the second matrix has rows.
    * For 2 matrices of sizes `(n,k)` and `(k,m)`, the resulting matrix will be of size `(n,m)`.
    * Essentially it's a matrix-vector product of the first matrix and each column of the second matrix. The resulting vectors will be the columns of the resulting matrix.
    * NumPy: `np.matmul(U, V)`
    * Python (3.5+): `U @ V`

## Identity matrix

* An **identity matrix** is a _square matrix_ (matrix of size `(n,n)`) entirely made of `0`'s, except for its _diagonal_ which is made of `1`'s.

        1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 1
* The identity matrix is the equivalent of the number `1` in scalar product. Given a matrix `U` and the identity matrix `I`:
    * `U · I = I · U = U`
* NumPy: `np.eye(n)` will return an identity matrix
    * `n`: size of the identity matrix.

## Matrix inverse

* Given a square matrix `A`, we can define its **inverse matrix `A⁻¹`** as a matrix that multiplied by the original matrix `A` results in the identity matrix `I`:
    * `A · A⁻¹ = A⁻¹ · A = I`
* Only square matrices may have inverse matrices. Not every square matrix has an inverse matrix. A non-invertible matrix is called _singular_ or _degenerate_.
* NumPy: `np.linalg.inv(A)`
* Inverse matrices are important because they are necessary to solve complex operations.

## Conclusion

Vectors and matrices are extremely convenient for ML because they allow us to solve complex systems of equations in a straightforward manner. All ML libraries make use of matrices and vectors in the background. The next lesson goes more in depth into the maths.

# Pandas

**Pandas** is a python data analysis library. It offers data structures and operations for manipulating tabular data. Its main data structure is called a **dataframe**, which is basically a table.

Please check the [Python and libraries cheatsheet](https://gist.github.com/ziritrion/9b80e47956adc0f20ecce209d494cd0a#pandas) for reference.

> [Back to Index](README.md)

> Next: [Linear Regression](02_linear_regression.md)