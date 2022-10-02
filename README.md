[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

# User Audience Prediction

Artificial Intelligence Fall 2021 Course Project

## Technologies
- **Scikitlearn**: Python's most versatile and robust library for machine learning and statistical modelling.
- **Tensorflow**: A Machine learning and artificial intelligence library focusing on training and inference of deep neural networks.
- **Numpy**: The fundamental package for scientific computing with Python.
- **Pandas**: A Data analysis and manipulation tool offering data structures and operations for manipulating numerical tables and time series.
- **Plotly**: A Graphing library that makes interactive, high-quality graphs.
- **Hazm**: A Python library for tokenizing, text cleaning, lemmatizing, tagging, and parsing Persian text.
- **Cleantext**: A Package to clean raw text data and stem the words (converting words with similar meanings into a single word)
- **NLPAug**: A Python library for textual augmentation in machine learning experiments.


## Project Description 
The [Natural Language Processing Laboratory](http://nlp.sbu.ac.ir) of the Faculty of Computer Engineering is developing a system called Soha (Intelligent Communication System) as the faculty chatbot and intelligent student assistant. Soha's task is to talk to the user, answer his questions, and provide the necessary guidance or refer questions, criticisms, suggestions, or requests of the user to the person responsible for answering. Therefore, in one of the small modules, the system must determine the addressee of a statement. This way, while students gain the experience of participating in real practical work, they can apply what they learned during the semester.

The purpose of this project is to build a model to classify the user's query(question) into these five categories:

- **Category 1:** *Faculty Education* aka. *آموزش دانشکده*
```
Example #1: آخرین مهلت حذف چه موقع است؟
Example #2: از کجا می تونم کارنامه مهر خورده بگیرم؟
```
- **Category 2:** *Information Desk* aka. *میز اطلاعات*
```
Example #1: کتابخانه کجاست؟
Example #2: کجا می توانم رییس دانشکده را ببینم؟
```
- **Category 3:** *Site/Library* aka. *سایت/کتابخانه*
```
Example #1: چگونه می توانم روي سرور دانشکده اکانت بگیرم؟
Example #2:  اگر کتاب رو دیر بیارم چقدر جریمه میشم؟
```
- **Category 4:** *Information and Suggestions Box* aka. *صندوق اطلاعات و پیشنهادات*
```
Example #1: نمیخواهید بالاخره کلاسها رو حضوري کنین؟
Example #2: آسانسور خراب شده، لطفا رسیدگی شود
```
- **Category 5:** *Others* aka. *سایر*
```
Example #1: هوا خوب است یا هوا ابری است؟
Example #2: دیشب پرسپولیس چند تا از استقلال گل خورد؟
```

## Dataset

The dataset files are available at https://www.kaggle.com/competitions/sbu-ai-finalproject/data. The training data has two columns which are described below. Note that the label column has numerical values in the range of [1, 5]:

- **Label 1:**  It corresponds to *Faculty Education* aka. *آموزش دانشکده*.
- **Label 2:**  It corresponds to *Information Desk* aka. *میز اطلاعات*.
- **Label 3:**  It corresponds to *Site/Library* aka. *سایت یا کتابخانه*.
- **Label 4:**  It corresponds to *Information and Suggestions Box* aka. *صندوق اطلاعات و پیشنهادات*.
- **Label 5:**  It corresponds to *Others* aka. *سایر*.


## Data Generation:
In this step, one paraphrase was added manually to generate more data for each 100 written sentences in the given dataset.

## Preprocessing:
Hazm library was used for sentence preprocessing. With various tests on the dataset, it was observed that the length of the sentences is effective in the final accuracy, and the minimum sentence length of 3 words and the maximum sentence length of 25 words had the best accuracy, and sentences shorter or longer than these numbers were discarded.
 
Also, in the training data, the number of sentences in different categories had a significant difference, and by equalizing the number of sentences in different categories, the model's accuracy improved.

In data preprocessing, removing the half-space or replacing it with a space decreased the model's accuracy, indicating its use in upcoming steps. Therefore, in removing special characters, we do not change this character.

## Data Augmentation
Removing repeated statements and detecting contradictions in two similar statements are done at this stage. In case of contradiction, only the first one is kept, and the rest are discarded.
Also, automatic data augmentation methods have been used in this step, which covers the following modes:
1. **Synonym replacement**: 
  ```
  Example: سهیل امروز با تاکسی از خانه تا رستوران رفت تا کباب بخورد
  Augmented #1: سهیل امروز با تاکسی از منزل تا رستوران رفت تا کباب بخورد
  ```
2. **Random insertion**:
  ```
  Augmented #2: سهیل امروز با تاکسی از خانه تا رستوران رفت تا کباب غذاخوری بخورد.
  ```
3. **Random swap**: 
  ```
  Augmented #3: سهیل امروز با تاکسی از رستوران تا خانه رفت تا کباب بخورد
  ```
4. **Random deletion**:
  ```
  Augmented #4: سهیل امروز از خانه تا رستوران رفت تا کباب بخورد
  ```
In this way, for all the data in the dataset, four other data items are generated automatically.


## Comparison of Models:
Different variations of Naïve Bayes were examined, which are suitable for certain types of processes with minor changes. For example, the second model is used more specifically for natural language processing and is suitable for discrete data. 

1. First, the **Naïve Bayes** model was implemented in its simple form.

2. **Multinomial Naïve Bayes** was used as the second model. It uses a multinomial distribution for each feature and guesses the tag of a text using the Bayes theorem. It calculates each tag's likelihood for a given sample and outputs the tag with the greatest chance. Compared to the last model, which requires much time for training and prediction, it provides competitive accuracy with much less time.

3. **Gaussian Naïve Bayes** was used as the third model. It assumes that each class follow a Gaussian distribution. In this way, continuous features can also be covered. However, because it is upgraded to use continuous features and the data used is discrete, it gives less accuracy in this project.
 
4. The fourth was the **Bernoulli Naïve Bayes** model. It is based on the Bernoulli Distribution and accepts only binary values, i.e., 0 or 1. Bernoulli Naive Bayes should be used when the dataset's features are binary.
 
5. The fifth model was the **Complement Naïve Bayes** model. It is somewhat an adaptation of the standard Multinomial Naive Bayes algorithm particularly suited to work with imbalanced datasets. In this approach, instead of calculating the likelihood of a word occurring in a class, we calculate the likelihood that it occurs in other classes.

6. As the last model, **ParsBERT** was used for data classification, which is the most accurate model. It is a monolingual language model based on Google's BERT architecture. Considering that the model has already been trained on one billion data and we used its existing pre-train for classification, we expect it to be more accurate. However, it takes much time to train and predict it.


[contributors-shield]: https://img.shields.io/github/contributors/aidaaminian/user-audience-prediction.svg?style=for-the-badge
[contributors-url]: https://github.com/aidaaminian/user-audience-prediction/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/aidaaminian/user-audience-prediction.svg?style=for-the-badge
[forks-url]: https://github.com/aidaaminian/user-audience-prediction/network/members
[stars-shield]: https://img.shields.io/github/stars/aidaaminian/user-audience-prediction.svg?style=for-the-badge
[stars-url]: https://github.com/aidaaminian/user-audience-prediction/stargazers
[issues-shield]: https://img.shields.io/github/issues/aidaaminian/user-audience-prediction.svg?style=for-the-badge
[issues-url]: https://github.com/aidaaminian/user-audience-prediction/issues
