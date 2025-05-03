import pickle
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import os

# Ensure the 'model' directory exists
if not os.path.exists('../model'):
    os.makedirs('../model')

# Load the dataset (update the path accordingly)
data = pd.read_csv('/mnt/c/Users/nikhi/Downloads/ap/dataset/job_postings.csv')

# Print columns to check the dataset structure
print(data.columns)  # This will print the names of the columns in the dataset

# Data preprocessing
def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))  # Remove non-alphabetical characters
    text = text.lower()  # Convert to lowercase
    return text

data['cleaned_description'] = data['description'].apply(preprocess_text)

# Check the column names to identify the label column
# If the column name for the labels is different, update the following line accordingly
print(data.head())  # Print the first few rows to inspect the structure

# Use 'fraudulent' column for labels instead of 'category'
y = data['fraudulent'].values  # This column holds the 1/0 labels (fraudulent/non-fraudulent)
X = data['cleaned_description']  # 'description' column to be used as features

# Vectorizing the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X).toarray()  # Transform the text data into a feature array

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
log_reg_model = LogisticRegression(max_iter=500)
log_reg_model.fit(X_train, y_train)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained models and the vectorizer
# Save Logistic Regression model
with open('../model/fake_job_model.pkl', 'wb') as model_file:
    pickle.dump(log_reg_model, model_file)

# Save Random Forest model
with open('../model/random_forest_model.pkl', 'wb') as rf_model_file:
    pickle.dump(rf_model, rf_model_file)

# Save the vectorizer
with open('../model/vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Predictions and evaluation
log_reg_pred = log_reg_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Logistic Regression Evaluation
print("Logistic Regression Model Evaluation:")
print(classification_report(y_test, log_reg_pred))

# Random Forest Evaluation
print("Random Forest Model Evaluation:")
print(classification_report(y_test, rf_pred))

# Optionally, print accuracy scores
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, log_reg_pred):.2f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.2f}")
