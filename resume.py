import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
import gensim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load and preprocess dataset
resume_df = pd.read_csv('data/resume_data.csv', encoding='latin-1')
resume_df = resume_df[['resume_text', 'class']]
resume_df['class'] = resume_df['class'].apply(lambda x: 1 if x == 'flagged' else 0)
resume_df['resume_text'] = resume_df['resume_text'].apply(lambda x: x.replace('\r', '').replace('\n', ' '))

# NLTK stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
additional_stops = ['from', 'subject', 'edu', 're', 'use', 'email', 'com']
stop_words.update(additional_stops)

# Preprocess text data
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stop_words and len(token) > 2:
            result.append(token)
    return ' '.join(result)

resume_df['cleaned'] = resume_df['resume_text'].apply(preprocess)

# Plotting class distribution
sns.countplot(resume_df['class'], label='Count Plot')
plt.title("Distribution of Classes")
plt.show()

# Generating and plotting WordCloud for each class
for class_label in [1, 0]:
    text = ' '.join(resume_df[resume_df['class'] == class_label]['cleaned'])
    wc = WordCloud(max_words=2000, width=1600, height=800, stopwords=stop_words).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Class {class_label}')
    plt.show()

# Feature extraction with TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(resume_df['cleaned'])
y = resume_df['class']

# Handling imbalanced data with SMOTE
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Using a RandomForest classifier
classifier = RandomForestClassifier(class_weight='balanced', random_state=42)
classifier.fit(X_train, y_train)
y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)

# Evaluation on training data
print("Training Data Evaluation:")
print("Accuracy of model on Train Dataset = {:.2f}".format(accuracy_score(y_train, y_pred_train)))
print("Precision of model on Train Dataset = {:.2f}".format(precision_score(y_train, y_pred_train, average='weighted')))
print("Recall of model on Train Dataset = {:.2f}".format(recall_score(y_train, y_pred_train, average='weighted')))
print("F1 Score of model on Train Dataset = {:.2f}".format(f1_score(y_train, y_pred_train, average='weighted')))
print("\n")

# Evaluation on testing data
print("Testing Data Evaluation:")
print("Accuracy of model on Test Dataset = {:.2f}".format(accuracy_score(y_test, y_pred_test)))
print("Precision of model on Test Dataset = {:.2f}".format(precision_score(y_test, y_pred_test, average='weighted')))
print("Recall of model on Test Dataset = {:.2f}".format(recall_score(y_test, y_pred_test, average='weighted')))
print("F1 Score of model on Test Dataset = {:.2f}".format(f1_score(y_test, y_pred_test, average='weighted')))
print("\n")

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix for Test Data')
plt.show()

# Classification report for detailed evaluation
print("Classification Report for Test Data:")
print(classification_report(y_test, y_pred_test))
