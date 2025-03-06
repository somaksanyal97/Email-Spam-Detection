# Spam Detection in emails. 

This project is an extension of the previous [Email Spam Classification Project](https://github.com/somaksanyal97/Email-Spam-Classifier).In the previous project, a combined dataset with 17570 email samples where used along with their labels to detect spam / ham using traditional ML algoriths in a supervised learning setting. 

In this project, we have used email samples of 6046 entries and have implemented several ML and DL algorithms to the check the performance of these algorithms on email spam classification. The precision is noted with importance for each of these aglorithms and hyperparater tuning has been used to improve model performance. <br>

<img src = "https://github.com/somaksanyal97/Email-Spam-Classifier/blob/main/Pictures/readme%20pic.png" style="width:1000px; height:300px;"> <br>

## Data Cleaning
In the preprocessing pipeline for the email spam classification dataset, the dataset is cleaned. This involved removing unnecessary columns, filtering out rows where the 'Body' column was empty and removing duplicates. The data has been checked for null values. <br>

## Text Preprocessing
Subsequently, the following pre-processing steps were implemented - the data was cleaned from non-alphabetical characters and stopwords, the text was converted to lowercase, split into individual words, and PorterS. I used stemming instead of lemmatization for this project because it is faster and I do not need the root form of the words to be meaningful for this project. The processed text was then compiled into a new list called corpus. <br>

## Feature Engineering
The cleaned text is converted to numerical feature representations using CountVectorizer and Tfidf Vectorizer with a maximum of 6000 features. The text is converted to Word2Vec embeddings by looking up pre-trained Word2Vec vectors and transformed into 500 dimensional vectors. The word2Vec embedding has been mostly used for all the algorithms in this project and they are able to hold the context of the text which is essential in determination of the output. <br>

## Oversampling
The dataset is then split into training and test datasets and SMOTE is applied to address the class imbalance in the dataset. <br>

## Training and evaluation
 
