# Machine Learning refresher

## What is Machine Learning?

Back at the beginning of the ML Zoomcamp course, in [Lesson 1](../notes/01_intro.md), we saw an explanation of what **Machine Learning** (ML) is:

In the classic programming paradigm, the basic workflow is as follows:

```
data + rules = outcome
```

But ML shuffles things around into this:

```
data + outcome = model
```

So that we can do this:

```
data + model = outcome
```

This _model_ is a new set of rules that we can generate by means of several statistics, calculus and computer science techniques based on the available data. This is great because it allows us to generate very complex rules that would otherwise be extremely hard or time-consuming to define by regular human means.

However, what we've seen so far in the course is a sub-type of ML called _Supervised Learning_ (SL). A SL model learns a _mapping_ between _input examples_ and a _target variable_: we provide a bunch of data for which the outcome we want to calculate (the target varaible) is already known and we fit a model that learns how to calculate the target variable based on the available data with the hopes that it will do a good job when we present the model with brand new data.

If you're wondering why this kind of ML is called _supervised_, it's because we "supervise" the learning by means of providing a ***ground truth*** in the form of ***labels*** (the known outcomes for the available data) so that our ML algorithm can compare the model's results with them and check how well it's performing.

However, SL is just one part of Machine Learning. What happens when we don't have labels? What can we do if our data is scarce? How do autonomous cars work?

## Beyond Supervised Learning

There are many different ML algorithms depending on the characteristics of the problem to solve and how we gather and process data. There are also a few different ways of classifying such algorithms.

[Yann LeCun](https://www.wikiwand.com/en/Yann_LeCun), one of the "Godfathers of Deep Learning", came up with his famous "cake analogy" to explain the different kinds of Machine Learning. According to his analogy, there are 2 main attributes in ML:

* How many samples you use for training
* How much info you get out of each sample

This results in 3 different kinds of ML according to his criteria:

* ***Reinforcement Learning***: the input samples are scarce and we get little info out of each sample.
    * E.g. teaching a robot to pick up an object. The data is limited to what we can detect around the robot and we need lots of tries in order to "predict" a simple action such as "crouch" or "open/close hand".
* ***Superviser Learning***: the input samples are bigger but costly (manually labeling data) and we predict categories or numbers for each input.
    * E.g. object classification in images, as we've already seen in [lesson 8 - deep learning](../notes/08_deep_learning.md).
* ***Unsupervised Learning***: the input samples are massive and unlabeled, and we make use of it to predict "everything".
    * E.g. predicting frames in a video. We make use of every pixel of each image for info.

![black forest cake](https://miro.medium.com/max/4416/1*bvMhd_xpVxfJYoKXYp5hug.png)

>Note: _Self-supervised Learning_ is a sub-type of Unsupervised Learning. This slide is an updated 2019 version from the [original 2016 one](https://miro.medium.com/max/1400/0*sQmcKODThlssh2V5.png). The cake analogy was actually meant to explain how intelligence works and to emphasize the importance of Unsupervised Learning. You can learn more about it [in this link](https://medium.com/syncedreview/yann-lecun-cake-analogy-2-0-a361da560dae).

Another way of classifying ML algorithms is [Alex Graves](https://www.wikiwand.com/en/Alex_Graves_(computer_scientist))' "types of learning" table:

|   | With teacher | Without teacher |
| --- | --- | --- |
| **Active agent** | Reinforcement Learning / Active Learning | Intrinsic Motivation / Exploration |
| **Passive agent** | Supervised Learning | Unsupervised Learning |

>Source: https://youtu.be/3RVGrz7MjMg?t=24

* The **teacher** is the mechanism that lets the ML algorithm know that it's performing well, just like a school teacher would behave with a student. In the case of Supervised Learning, the labels are the teacher.
* The ***agent*** is the mechanism by which the ML algorithm receives and processes data.
    * Supervised Learning is a _passive agent_ type of ML because we're directly feeding the data in an ordered manner, like in the case of tabular data or image datasets.
    * An _active agent_ means that the algorithm must look for data on its own. If we're teaching a robot to pick up an object, the robot must actively generate its own data by observing its surroundings.

Both of these classification systems are valid and emphasize different aspects of ML.

In this article, I'd like to focus on Reinforcement Learning.