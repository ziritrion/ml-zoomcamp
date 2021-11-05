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

# Working with text

If a feature consists of text (for example, appartment descriptions, reviews, etc), one way of dealing with them is by codifying the text to vectors.

Scikit-learn has tools that allows us to do exactly this:

    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer()
    cv.fit(feature)

`CountVectorizer` will count all the words in the feature and create vectors with as many dimensions as words. A sentence is codified with word counts, thus allowing us to work with the feature.

Scikit also offers other similar tools:

* `TfidVectorizer`: similar to `CountVectorizer` but uses [tf-idf](https://www.wikiwand.com/en/Tf%E2%80%93idf) to assign weights between 0 and 1 to each position in the vector, thus giving more importance to some words than others.

# Using `CountVectorizer` with categorical features to reduce feature size

If you have a categorical feature with many different values, thus resulting in a feature matrix too large to handle, you can use `CountVectorizer` for processing these features and reduce size. This is useful in instances where a categorical feature has too many categories, but many of these categories have very low counts (_long tail distribution_), which could lead to worse model performance.

Using the regular `DictVectorizer` would create a column for each of these categories, therefore creating a huge feature matrix.

Instead, we can follow these steps to get rid of some categories and make a more manageable feature matrix:

1. Preprocess the features so that no whitespaces remain (substitute the whitespaces with underscores, etc).
1. Create a `string` containing the contents of all the features that need processing.
    * `my_features = 'feature_1=' + df.feature_1 + ' ' + 'feature_2=' + df.feature_2`
    * This will output a single string with all of the contents of the chosen features in the `df` dataframe.
1. Train a `CountVectorizer` instance.
    *       from sklearn.feature_extraction.text import CountVectorizer

            cv_features = CountVectorizer(token_pattern='\S+', min_df=50, dtype='int32')
            cv_features.fit(my_features)
    * `token_pattern` allows you to define a ***regular expression*** to parse the string. Lowercase `\s` means whitespace, uppercase `\S` is everything that is not a whitespace; `+` means that at least 1 instance of the preceding character or token must occur; therefore, this regular expression will parse any collection of consecutive non-whitespace characters as an entity, and whitespaces will be used as the separation between entities.
    * `min_df` stands for _minimum document frequency_. `CountVectorizer` will only take into account the categories that appear at least as many times as the amount specified in this parameter (in our example, 50). All other categories with counts smaller than the specified amount will be discarded.
    * The default type is `int64`; we switch to `int32` because the generated vectors will be made of zeros and ones, so we can cut the memory usage by half by changing the type without losing any info.
    * `cv_features.get_feature_names()` should return a list similar to this:

            ['feature_1=type_1',
            'feature_1=type_2',
            'feature_2=type_A',
            'feature_2=type_B']
1. `X = cv_features.transform(my_features)`
    * This will convert your string to vectors, as shown in the [Working with text section](#working-with-text).

# Joining processed text and categorical features

After using `CountVectorizer`, it's likely that you'll want to rejoin the features together, especially if you've transformed text features on one hand and categorical features on the other.

Here's a code snippet that shows how to do this.

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

* The matrices created by the `transform()` function are ***sparse matrices*** (matrices mostly composed of zeroes). Because sparse matrices take a lot of memory and most of the spaces are empty, scikit-learn uses a special format called _Compressed Sparse Row Format_ (CSR), which compresses the matrices to make them much smaller. Using this example code, you can check this by inputiing `X_texts` in your notebook.
* SciPy is a library for scientific computing which is built on top of NumPy arrays. We use `scipy.sparse.hstack()` for stacking sparse matrices; using Numpy's `np.hstack()` would create an array with 2 sparse matrix objects, which isn't what we want nor need.
* `scipy.sparse.hstack()` outputs a single matrix composed of the 2 input matrices stacked one next to the other horizontally, but changes the format from CSR to _COOrdinate format_ (COO). COO is a fast format but does not allow arithmetic operations or slicing, so if you need to operate on this new matrix, you need to convert it to CSR with `tocsr()`. In this example no further operations are done with the matrix, so this conversion isn't needed.

# Dealing with unbalanced datasets

Your dataset may contain certain features which have very uneven distributions. For example: if your dataset deals with prices and you have an "expensive threshold" which only a handful of rows manages to pass, then most of the values will be below the threshold. This is a problem when splitting the dataset: you can be unlucky and end up with splits that do not take this distribution into account: you may end up with a training dataset with very or even none of the expensive rows but validation and test datasets could end up with many, or any other similar situation.

In order to avoid these unbalanced splits, `train_test_split()` contains a special parameter, `stratify`, which takes a feature as input and makes sure that the distribution for that feature will be respected in all splits.

`df_train, df_test = train_test_split(df, random_state=1, test_size=0.2, stratify=df.my_weird_feature)`

