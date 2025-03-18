# Spam Detection in emails. 

This project is an extension of the previous [Email Spam Classification Project](https://github.com/somaksanyal97/Email-Spam-Classifier).In the previous project, a combined dataset with 17570 email samples where used along with their labels to detect spam / ham using traditional ML algoriths in a supervised learning setting. 

In this project, we have used email samples of 6046 entries and have implemented several ML and DL algorithms to the check the performance of these algorithms on email spam classification. The precision is noted with importance for each of these aglorithms and hyperparater tuning has been used to improve model performance. <br>

<img src = "https://github.com/somaksanyal97/Email-Spam-Classifier/blob/main/Pictures/readme%20pic.png" style="width:1000px; height:300px;"> <br>

## Data Cleaning
In the preprocessing pipeline for the email spam classification dataset, the dataset is cleaned. This involved removing unnecessary columns, filtering out rows where the 'Body' column was empty and removing duplicates. The data has been checked for null values. <br>

## Text Preprocessing
Subsequently, the following pre-processing steps were implemented - the data was cleaned from non-alphabetical characters and stopwords, the text was converted to lowercase, split into individual words, and PorterStemmer has been used for stemming. I used stemming instead of lemmatization for this project because it is faster and I do not need the root form of the words to be meaningful for this project. The processed text was then compiled into a new list called corpus. <br>

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

With the word2vec embeddings, several advanced ML algorithms such as XGBoost, LightGBM, AdaBoost and CatBoost are trained. GridSearchCV is used for hyperparameter tuning, optimizing parameters such as n_estimators (number of trees), iterations (for CatBoost), learning_rate, and boosting_type. The implementation can be found [here](https://github.com/somaksanyal97/Email-Spam-Detection/blob/main/email_spam%20ML%20advanced.ipynb).

Also, deep learning models - RNN, LSTM, and GRU are trained using pre-trained Word2Vec embeddings and Bidirectional layers for LSTM and GRU. Tokenization & Padding are applied to standardize input sequences. These models are compiled with Adam optimizer and trained using binary cross-entropy loss. The code implementation can be found [here](https://github.com/somaksanyal97/Email-Spam-Detection/blob/main/Email_Spam_DL.ipynb).
## Performance Metrics of ML and DL Algorithms with Word2Vec embeddings

| Model        | Accuracy   | Precision | Recall   | F1 Score | Best Parameters |
|----------------|-----------|--------------------|----------------|-----------|--------------------|
| Logistic Regression | 0.94 | 0.94    | 0.94 | 0.94 |   {'C': 10}   |
| K-Nearest Neighbors | 0.94 | 0.95    | 0.94 | 0.94 |    {'n_neighbors': 3}   |
| Decision Tree | 0.91 | 0.92    | 0.91 | 0.91 |  {'max_depth': 20}   |
| Random Forest | 0.96 | 0.96     | 0.96 | 0.96 |   {'max_depth': 30, 'n_estimators': 100} |
| XGBoost  | 0.97 | 0.96    | 0.97 | 0.97 |  {'learning_rate': 0.1, 'n_estimators': 100} |
| LightGBM | 0.97 | 0.97     | 0.97 | 0.97 |   {'learning_rate': 0.1, 'n_estimators': 100} |
| Adaboost | 0.89 | 0.89   | 0.89 | 0.89 |  {'learning_rate': 0.1, 'n_estimators': 100} |
| Adaboost | 0.96 | 0.96   | 0.96 | 0.96 |  {'learning_rate': 0.1, 'iterations': 100} |
| RNN | 0.96 | 0.94   | 0.98 | 0.96 |   |
| LSTM (Birdirectional) | 0.98 | 0.99   | 0.98 | 0.98 |   |
| GRU (Bidirectional) | 0.98 | 0.99   | 0.97 | 0.98 |   |

The final implementation is in email spam dectection project is using DistilBERT (a lighter and faster version of BERT - Bidirectional Encoder Representations from Transformers), a transformer-based deep learning model. It starts by loading and preprocessing the dataset, removing duplicates and missing values. The text data is split into training and testing sets, and DistilBERT's tokenizer is applied to convert text into tokenized inputs with attention masks. The processed data is wrapped in Hugging Face's Dataset format for compatibility with transformer models. DistilBertForSequenceClassification is loaded with pretrained weights from "distilbert-base-uncased". A DistilBERT model is initialized for binary classification, and training is performed using the Trainer API with three epochs, batch size of 8, and epoch-based evaluation and saving strategies. After training, the model is evaluated using accuracy, precision, recall, and F1-score, and a confusion matrix is plotted to visualize classification performance. The code implementation can be found [here](https://github.com/somaksanyal97/Email-Spam-Detection/blob/main/Email_Spam_BERT_%2B_Transformer.ipynb). 

## Performance Metrics of transformer based DL model using DistilBERT
| Model        | Accuracy   | Precision | Recall   | F1 Score |
|----------------|-----------|--------------------|----------------|-----------|
| Transformer based DL model | 0.98 | 0.97   | 0.98 | 0.97 |   |


## CONCLUSION

This project successfully implements various machine learning and deep learning approaches for spam detection in emails. Traditional models like Naïve Bayes, Logistic Regression, Decision Trees, and Random Forests were evaluated alongside advanced techniques such as XGBoost, LightGBM, and AdaBoost, with hyperparameter tuning using GridSearchCV. Additionally, deep learning models, including RNN, LSTM, and GRU, were trained with Word2Vec embeddings, and a transformer-based approach using DistilBERT was implemented to leverage state-of-the-art natural language processing capabilities. The results show that DistilBERT and deep learning models outperform traditional machine learning classifiers in handling complex textual patterns, with further fine-tuning. Ultimately, transformer-based models offer superior accuracy and robustness, making them highly effective for real-world spam detection applications.
