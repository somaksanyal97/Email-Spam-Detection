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
Several classifiers are trained using GridSearchCV to optimize hyperparameters. The models are evaluated based on accuracy, precision, recall, and F1-score, and confusion matrices are plotted to visualize performance. The script iterates through different classifiers for each feature extraction method and selects the best-performing model.
As this is a spam detection project, the precision is priotised the most to handle the false negatives in the output. 

In the first part of the code, traditional machine learning algorithms like Naive Bayes classifier, Logistic Regression, KNN Classifier, Decision Tree Classifier and Random Forest Classifier is implement with all the text vector representations from Count Vectorizer, Tfidf Vectorizer and Word2Vec. THe iterations were repeated over each classifier, performing hyperparameter tuning using GridSearchCV with 5-fold cross-validation on the oversampled training data. The best model was then used to make predictions on the test data, and performance metrics were calculated and stored. Confusion matrices and performance metrics were printed and visualized for each classifier.

Important to note here, the Naive Bayes classifier assumes that features (words) are independent, but Word2Vec embeddings capture word dependencies and semantic relationships, violating this assumption. Additionally, Naive Bayes works best with discrete, count-based features (e.g., Bag-of-Words or TF-IDF), whereas Word2Vec produces dense, continuous vectors, making it incompatible with Naive Bayes' probability-based calculations. Thus Word2Vec was not used for Naive Bayes classifier. 

## Performance Metrics of ML Algorithms with CountVectorizer

| Model        | Accuracy   | Precision | Recall   | F1 Score | Best Parameters |
|----------------|-----------|--------------------|----------------|-----------|--------------------|
| Naive Bayes | 0.98 | 0.98     | 0.98 | 0.98 |  {'alpha': 0.5}   |
| Logistic Regression | 0.98 | 0.98    | 0.98 | 0.98 |   {'C': 10}   |
| K-Nearest Neighbors | 0.79 | 0.88     | 0.79 | 0.80 |   {'n_neighbors': 3}   |
| Decision Tree | 0.93 | 0.93     | 0.93 | 0.93 |  {'max_depth': None}   |
| Random Forest | 0.98 | 0.98    | 0.98 | 0.98 |   {'max_depth': None, 'n_estimators': 100} |

## Performance Metrics of ML Algorithms with TfidfVectorizer

| Model        | Accuracy   | Precision | Recall   | F1 Score | Best Parameters |
|----------------|-----------|--------------------|----------------|-----------|--------------------|
| Naive Bayes | 0.98 | 0.98    | 0.98 | 0.98 |  {'alpha': 0.5}   |
| Logistic Regression | 0.98 | 0.98    | 0.98 | 0.98 |   {'C': 10}   |
| K-Nearest Neighbors | 0.55 | 0.83    | 0.55 | 0.55 |    {'n_neighbors': 3}   |
| Decision Tree | 0.94 | 0.94     | 0.94 | 0.94 |  {'max_depth': None}   |
| Random Forest | 0.98 | 0.98     | 0.98 | 0.98 |   {'max_depth': None, 'n_estimators': 200} |
