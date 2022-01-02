# diSimplex Comment Processor

A simple Bayesian based comment processor

Since the current diSimplex editor is an English speaker (alas only
English), comments to be read and understood need to be written in
(reasonable) English.

This suggests that the comment processor should target classification
using English. Since comments will come from non-native English speakers,
this comment processor *must* be able to classify imperfect English. This
suggests that any Natural Language Processing tools must not use models of
perfect English. Among other things, this suggest we should only use
simple stemming, rather than more complex lemmation or parts of speech
analysis.

We will use [Snowball](https://snowballstem.org/) or a
[Porter](https://tartarus.org/martin/PorterStemmer/) algorithm to reduce
the variations in English words.

We will ignore words which are equally prevalent in all classes
(effectively stop words based upon our own corpus). We will only compute
probabilities using the X *most* distinguishing words rather than all words
in the comment (this is similar to [Paul Graham's
suggestion](http://www.paulgraham.com/better.html) to use the 15 that only
occur in one of the classes).

We will use [Mutinomial naive Bayes
classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_na%C3%AFve_Bayes)
using [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) weights
instead of raw term frequencies.

We will classify comments in to four categories:

1. **S**pam
2. General **P**ublic
3. **R**esearcher
4. **E**xpert

To ensure greater specificity as we move between the categories, the
non-class will always be the collection of all more expert categories:

1. non-spam is the union of Public, Researcher and Expert comments
2. non-public is the union of Researcher and Expert comments
3. non-researcher is the collection of Expert comments

We will estimate the probable classification of each of the three categorical choices:

1. spam / non-spam
2. public / non-public
3. researcher / non-researcher

The chosen classification will be the most expert predicted classification
obtained from all three tests.

### Additional tools

We *might* eventually consider using a Neural Network approach using
[TensorFlow (lite?)](). We might even consider using TensorFlow to
[implement the Naive Bayes
estimator](https://nicolovaligi.com/articles/naive-bayes-tensorflow/).

We *should* [check
emails](https://blog.bounceless.io/how-to-check-if-an-email-address-is-fake/)

## Resources

- [Naive Bayes Classifier
  Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
  ([Document
  Example](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Document_classification))

- [NumPy](https://numpy.org/)
  ([Documentation](https://numpy.org/doc/stable/) -
  [Why](https://numpy.org/doc/stable/user/whatisnumpy.html))

- [Scikit-learn Naive Bayes
  API/Code](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes)

- [A plan for spam](http://www.paulgraham.com/spam.html)

- [Better Bayesian filtering](http://www.paulgraham.com/better.html)

- [Link-Based Text Classification Using Bayesian
Networks](https://www.researchgate.net/publication/221232818_Link-Based_Text_Classification_Using_Bayesian_Networks)

- [pyArgrum](https://agrum.gitlab.io/)

- [Use Sentiment Analysis With Python to Classify Movie
Reviews](https://realpython.com/sentiment-analysis-python/)

- [Sentiment Analysis: First Steps With Python's NLTK
Library](https://realpython.com/python-nltk-sentiment-analysis/)

- [Email Spam
Classification](https://www.kaggle.com/balaka18/email-spam-classification/notebook)

- [Email Spam Classification in
Python](https://www.askpython.com/python/examples/email-spam-classification)

- [Naive Bayes Classification using
Scikit-learn](https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn)

- [Machine Learning with Python
Tutorial](https://www.tutorialspoint.com/machine_learning_with_python/index.htm)

- [spaCy](https://spacy.io/)

- [NLTK](https://www.nltk.org/)

- [Tensor Flow](https://www.tensorflow.org/)
  [Lite](https://www.tensorflow.org/lite)
  [ModelMaker](https://www.tensorflow.org/lite/guide/model_maker)

- [Compare Tensorflow Deep Learning Model with Classical Machine Learning
  models — KNN, Naive Bayes, Logistic Regression, SVM — IRIS
  Classification](https://medium.com/analytics-vidhya/compare-tensorflow-deep-learning-model-with-classical-machine-learning-models-knn-naive-bayes-61b40bb3382)

## Datasets

- [Spam Filtering with Naive Bayes -- Which Naive
  Bayes?](http://www2.aueb.gr/users/ion/docs/ceas2006_paper.pdf) with
  [dataset](http://www2.aueb.gr/users/ion/data/enron-spam/)

- [Spamassassin public
  corpus](https://spamassassin.apache.org/old/publiccorpus/)

- [UCI machine learning
  repository](https://archive.ics.uci.edu/ml/datasets.php))

    - [Spambase Data Set](http://archive.ics.uci.edu/ml/datasets/Spambase)

    - [SMS Spam Collection Data
      Set](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

- [10 Open-Source Datasets For Text
  Classification](https://analyticsindiamag.com/10-open-source-datasets-for-text-classification/)

- [Spam filtering
  datasets](https://aclweb.org/aclwiki/Spam_filtering_datasets)
