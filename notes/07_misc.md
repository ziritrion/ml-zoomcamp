> Previous: [Decision Trees](06_trees.md)

> [Back to Index](README.md)
>> [Back to Midterm Project](../07_midterm_project/README.md)

> Next: [Deep Learning](08_deep_learning.md)

Session 7 is actually reserved for the midterm project. During these weeks we did not have any theory videos but we did have some "office hours" live sessions which contained useful tips and tricks. These notes are a recopilation of them.

# Multiclass classification

## Multiclass Logistic Regression

We can use scikit-learn's Logistic Regression for multiclass classification. When dealing with multiclass problems, scikit-learn will generate multiple binary classification models, one per each binary feature.

For example, if we want to classifiy between dogs, cats and birds, the 3 internal models will be dog-no_dog, cat-no_cat and bird-no_bird.

## Multiclass Decision Tree Classifier

In the case of Decision trees, leaves simply lead to different classes, so one branch can lead to one class and another leads to a different class.

# Feature importance for continuous target (regression)

* Numeric feature + numeric target
    *  ***Correlation***. Already seen on the [notes for week 3](03_classification.md).
* Categorical feature + numeric target
    1. Convert numeric target to _categorical_ (***binning***).
        * Pandas has the `cut()` function which transforms a numerical feature into intervals (_bins_) and treats them as categories.
    1. Use ***mutual information***, as shown on the [notes for week 3](03_classification.md).

# Evaluating your model when there's time information

If the data is time-sensitive or related with time in any fashion, splitting and shuffling the dataset into the typical train/validation/test splits could destroy that information.

In these cases, the split is done without shuffling. The 60% split for train will contain the oldest data, the 20% validation split will contain more recent data and the 20% test split will contain the latest data.

# Advanced uses for Vectorizers
## Working with text

If a feature consists of text (for example, appartment descriptions, reviews, etc), one way of dealing with them is by codifying the text to vectors.

Scikit-learn has tools that allows us to do exactly this:

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
cv.fit(feature)
```

`CountVectorizer` will count all the words in the feature and create vectors with as many dimensions as words. A sentence is codified with word counts, thus allowing us to work with the feature.

Scikit also offers other similar tools:

* `TfidVectorizer`: similar to `CountVectorizer` but uses [tf-idf](https://www.wikiwand.com/en/Tf%E2%80%93idf) to assign weights between 0 and 1 to each position in the vector, thus giving more importance to some words than others.

## Using `CountVectorizer` with categorical features to reduce feature size

If you have a categorical feature with many different values, thus resulting in a feature matrix too large to handle, you can use `CountVectorizer` for processing these features and reduce size. This is useful in instances where a categorical feature has too many categories, but many of these categories have very low counts (_long tail distribution_), which could lead to worse model performance.

Using the regular `DictVectorizer` would create a column for each of these categories, therefore creating a huge feature matrix.

Instead, we can follow these steps to get rid of some categories and make a more manageable feature matrix:

1. Preprocess the features so that no whitespaces remain (substitute the whitespaces with underscores, etc).
1. Create a `string` containing the contents of all the features that need processing.
    * `my_features = 'feature_1=' + df.feature_1 + ' ' + 'feature_2=' + df.feature_2`
    * This will output a single string with all of the contents of the chosen features in the `df` dataframe.
1. Train a `CountVectorizer` instance.
    ```python
    from sklearn.feature_extraction.text import CountVectorizer

    cv_features = CountVectorizer(token_pattern='\S+', min_df=50, dtype='int32')
    cv_features.fit(my_features)
    ```
    * `token_pattern` allows you to define a ***regular expression*** to parse the string. Lowercase `\s` means whitespace, uppercase `\S` is everything that is not a whitespace; `+` means that at least 1 instance of the preceding character or token must occur; therefore, this regular expression will parse any collection of consecutive non-whitespace characters as an entity, and whitespaces will be used as the separation between entities.
    * `min_df` stands for _minimum document frequency_. `CountVectorizer` will only take into account the categories that appear at least as many times as the amount specified in this parameter (in our example, 50). All other categories with counts smaller than the specified amount will be discarded.
    * The default type is `int64`; we switch to `int32` because the generated vectors will be made of zeros and ones, so we can cut the memory usage by half by changing the type without losing any info.
    * `cv_features.get_feature_names()` should return a list similar to this:
    ```python
    ['feature_1=type_1',
    'feature_1=type_2',
    'feature_2=type_A',
    'feature_2=type_B']
    ```
1. Convert the string to vectors.
    ```python
    X = cv_features.transform(my_features)
    ```

    * This will convert your string to vectors, as shown in the [Working with text section](#working-with-text).

## Joining processed text and categorical features

After using `CountVectorizer`, it's likely that you'll want to rejoin the features together, especially if you've transformed text features on one hand and categorical features on the other.

Here's a code snippet that shows how to do this.

```python
# Categorical features
cv_categories= CountVectorizer(token_pattern='\S+', min_df=50, dtype='int32')
cv_categories.fit(my_features)

# Text features
cv_texts = CountVectorizer()
cv_texts.fit(my_text_features)

# Creating the feature matrices
X_categories = cv_categories.transform(my_features)
X_texts = cv_texts.transform(my_text_features)

# Stacking the 2 feature matrices together into one
import scipy
X = scipy.sparse.hstack([X_categories, X_texts])

# Optional matrix reformatting
X = X.tocsr()
```

* The matrices created by the `transform()` function are ***sparse matrices*** (matrices mostly composed of zeroes). Because sparse matrices take a lot of memory and most of the spaces are empty, scikit-learn uses a special format called _Compressed Sparse Row Format_ (CSR), which compresses the matrices to make them much smaller. Using this example code, you can check this by inputiing `X_texts` in your notebook.
* SciPy is a library for scientific computing which is built on top of NumPy arrays. We use `scipy.sparse.hstack()` for stacking sparse matrices; using Numpy's `np.hstack()` would create an array with 2 sparse matrix objects, which isn't what we want nor need.
* `scipy.sparse.hstack()` outputs a single matrix composed of the 2 input matrices stacked one next to the other horizontally, but changes the format from CSR to _COOrdinate format_ (COO). COO is a fast format but does not allow arithmetic operations or slicing, so if you need to operate on this new matrix, you need to convert it to CSR with `tocsr()`. In this example no further operations are done with the matrix, so this conversion isn't needed.

# Scikit-learn pipelines

In the previous example, we would also need to join our processed features with other numeric features. This could end up being a substantial amount of code for a simple and repetitive task. **SciKit-learn's Pipelines** aims to solve this issue.

A ***pipeline*** is a mechanism that chains multiple processes into one. This is useful for situations where there are a fixed sequence of steps in processing the data. Pipelines are often used along with ***transformers***, which may clean, reduce, expand or generate feature representations.

## Transformers

For our example use case (preprocessing of different types of features), we will use `ColumnTransformer`, which helps performing different transformations for different features within a _pipeline_ that is safe from data leakage.

1. We define a `transformations` array that contains the name of the features we want to preprocess, along with the class of preprocessing we want to apply to them.
1. We create a `ColumnTransformer` object using our `transformations`.
1. We train the transformer by calling `transformer.fit(df)`, where `df` is our dataframe.
1. Once a transformer is trained, the feature matrix is generated with `X = transformer.transform(df)`.

In code form:

```python
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

# Data import and cleanup code is omitted

# transformations array
transformations = [
    ('numerical', 'passthrough', ['num_feat1', 'num_feat2', 'num_feat3']),
    ('categories', OneHotEncoder(dtype='int32'), ['cat_feat1', 'cat_feat2', 'cat_feat_3']),
    ('name', CountVectorizer(min_df=100, dtype='int32'), 'name')
]

transformer = ColumnTransformer(transformations, reminder='drop')

transformer.fit(df_train)

X = transformer.transform(df_train)
y = df_train.target_feature.values
```

* In this example, `num_feat*` are numerical features that won't need additional preprocessing, `cat_feat*` are categorical features that we will one-hot encode and `name` is a text feature that we want to codify with `CountVectorizer`.
* `transformations` is a list of tuples. Each tuple contains a name (for the transformer), a transformation we want to apply, and a list of features that will receive such transformation.
    * We don't need additional preprocessing for our numerical features, so point this out by using the string `'passthrough'` as its transformation.
* The `reminder` option in `ColumnTransformer` is set to `drop` because we discard all features that are not specified in the `transformations` array. This is the default parameter value.
    * If we want to keep those features, we could do instead `reminder='passthrough'`, or we could apply another transformation.

## Pipelines with transformers

We don't actually have to train our transformer and generate our feature matrix and target column ourselves, because `pipeline` can take care of it for us:

1. We create our transformations array and our transformer like in the previous code example.
1. We create a `Pipeline` object which takes a list as input, containing our transformer and our model.
1. We train our pipeline with `pipeline.fit(df, df.target.values)` and make our predictions with `pipeline.predict(new_df)` as usual.

In code form:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# transformations array
transformations = [
    ('numerical', 'passthrough', ['num_feat1', 'num_feat2', 'num_feat3']),
    ('categories', OneHotEncoder(dtype='int32'), ['cat_feat1', 'cat_feat2', 'cat_feat_3']),
    ('name', CountVectorizer(min_df=100, dtype='int32'), 'name')
]

tranformer = ColumnTransformer(transformations, remainder='drop')

pipeline = Pipeline([
    ('transformer', transformer),
    ('lr', LinearRegression())
])

pipeline.fit(df_train, df_train.target_feature.values)
pipeline.predict(df_val)
```

* `pipeline` is an object composed of a list of tuples. Each tuple contains a name for the pipeline and the object it represents; in our case, the first element is our transformer and the second one is the model we will use for our predictions. The pipeline will then run each part consecutively.
* We train and predict using our regular training and validation dataframes.
    * Note that we use the Pandas dataframes, not NumPy arrays; `Pipeline` can handle Pandas dataframes for us.

## Creating custom transformers

In the [CountVectorizer section](#using-countvectorizer-with-categorical-features-to-reduce-feature-size) we preprocessed a bunch of categorical features by reducing the amount of categories and codifying the values, instead of doing a simple one-hot encoding. We can create a custom transformation object that allows us to do this and insert it in our pipeline.

To do so, we will use a ***mixin***, a class that contains methods for use by other classes without having to be the parent class of those other classes.

Scikit-learn provides `TransformerMixin`, which gives us the necessary methods to create our custom transformer.

The example custom transformer will create a custom string just like in the `CountVectorizer` example.

```python
from sklearn.base import TransformerMixin

class ConcatenatingTranformer(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        columns = list(X.columns)           
        res = ''            
        for c in columns:
            res = res + ' ' + c + '=' + X[c]
        return res.str.strip()
```

* We create a new class, `ConcatenatingTransformer` which extends `TransformerMixin`.
* A class that extends `TransformerMixin` needs to implement 2 methods: `fit` and `transform`.
    * `fit` doesn't need to be changed in our case.
    * `transform` does our transformation.
        * `X` is a dataframe with the columns we need.
        * We create the custom string in `res`, contatenating the name of each column and its contents on each iteration.
        * The output is a single string with all of the contents of the features we want to process. `strip()` removes all whitespaces from the beginning and the end.

With our custom class in place, we can now instantiate it and add it to a pipeline that will process the features we want.

```python
ct = ConcatenatingTranformer()

p2 = Pipeline([
    ('concatenate', ConcatenatingTranformer()),
    ('vectorize', CountVectorizer(token_pattern='\S+', min_df=100))
])

# Optional
X = p2.fit_transform(df[['cat_feat1', 'cat_feat2', 'cat_feat_3']])
```

* We first apply our custom transformation and then we reduce the feature size and vectorize the categories.
* Remember that `CountVectorizer` needs a regex in order to tokenize each "word" of the single string we pass to it, and we need a minimum count value in order to discard any features with a smaller count in the dataframe.
* `p2.fit_transform()` trains and applies the transformation we specified.
    * It returns a sparse matrix of type `numpy.int64`
    * We don't actually need to do a fit/transform, because we will integrate our pipeline with the pipeline we defined earlier.

We can now integrate our custom transformation pipeline with our regular pipeline:

```python
# transformations array
transformations = [
    ('numerical', 'passthrough', ['num_feat1', 'num_feat2', 'num_feat3']),
    ('categories', Pipeline([
                ('concatenate', ConcatenatingTranformer()),
                ('vectorize', CountVectorizer(token_pattern='\S+', min_df=100))
    ]), ['cat_feat1', 'cat_feat2', 'cat_feat_3']),
    ('name', CountVectorizer(min_df=100, dtype='int32'), 'name')
]

tranformer = ColumnTransformer(transformations, remainder='drop')

pipeline = Pipeline([
    ('transformer', transformer),
    ('lr', LinearRegression())
])

pipeline.fit(df_train, df_train.target_feature.values)
pipeline.predict(df_val)
```

Pipelines can be exported to files with `pickle`, which makes it useful for deploying purposes.

# Dealing with unbalanced datasets

Your dataset may contain certain features which have very uneven distributions. For example: if your dataset deals with prices and you have an "expensive threshold" which only a handful of rows manages to pass, then most of the values will be below the threshold. This is a problem when splitting the dataset: you can be unlucky and end up with splits that do not take this distribution into account: you may end up with a training dataset with very or even none of the expensive rows but validation and test datasets could end up with many, or any other similar situation.

In order to avoid these unbalanced splits, `train_test_split()` contains a special parameter, `stratify`, which takes a feature as input and makes sure that the distribution for that feature will be respected in all splits.

```python
df_train, df_test = train_test_split(df, random_state=1, test_size=0.2, stratify=df.my_weird_feature)
```

# K-nearest neighbors (k-NN)

The esential idea for **k-nearest neighbors regression** is to look at items close to the item you want to predict and compute the average of the neighbors. For **k-nearest neighbors classification**, the item is classified according to the majority class of its closest items.

```python
from sklearn-neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5)

knn.fit(X_train, y_train)

knn.predict(X_val)
```

* `n_neighbors` defines the amount of neighbors we will look at to compute our result.

k-NN has good performance but the models are very big because they need to "remember" the whole dataset.

k-NN is a good fit for **search** problems. We take a _query_ and we compare it to elements in our database, looking for the _highest similarity_ (or _lowest distance/error_, which is the same). We **rank** the found elements and present them as our prediction.

> Previous: [Decision Trees](06_trees.md)

> [Back to Index](README.md)
>> [Back to Midterm Project](../07_midterm_project/README.md)

> Next: [Deep Learning](08_deep_learning.md)