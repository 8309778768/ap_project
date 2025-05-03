import pickle
import re
import pandas as pd
import numpy as np

# Load the saved models and vectorizer
with open('../model/random_forest_model.pkl', 'rb') as rf_model_file:
    rf_model = pickle.load(rf_model_file)

with open('../model/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load the dataset (job_postings.csv) containing valid job titles
df = pd.read_csv('/mnt/c/Users/nikhi/Downloads/ap/dataset/job_postings.csv')  # Make sure the CSV is in the correct path

# Display the first few rows to check the columns and sample data
print("First few rows of the dataset:")
print(df.head())  
print("Columns in the dataset:", df.columns)  # Showing the columns of the dataset

# Display the job titles available in the dataset
valid_job_titles = set(df['title'].str.lower())  # Set of lowercase job titles from the dataset

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))  # Remove non-alphabetical characters
    text = text.lower()  # Convert to lowercase
    return text

# User input for job titles
job_titles_input = input("Enter the job titles separated by commas: ")

# Split the input into individual job titles
job_titles = job_titles_input.split(',')

# Initialize counters for real and fake job predictions
real_jobs_count = 0
fake_jobs_count = 0

# Set a higher threshold for prediction decision
threshold = 0.6  # Increased threshold

# Loop through each job title, process, and make prediction
for job_title in job_titles:
    job_title = job_title.strip()  # Remove any extra spaces around job titles

    # Check if the job title exists in the dataset (case insensitive)
    if job_title.lower() not in valid_job_titles:
        print(f"Job Title: {job_title}")
        print(f"Random Forest Prediction: Fake (Not found in dataset)")
        fake_jobs_count += 1
    else:
        # Preprocess the job title
        cleaned_job_title = preprocess_text(job_title)

        # Vectorize the job title
        X_new = vectorizer.transform([cleaned_job_title]).toarray()

        # Predict the probability of the job being fraudulent (class '1')
        rf_prob = rf_model.predict_proba(X_new)[0][1]  # Probability of class '1' (fraudulent)

        # Print the job title and its prediction
        print(f"Job Title: {job_title}")
        print(f"Random Forest Probability (Fraudulent): {rf_prob:.3f}")

        # Make decision based on threshold
        if rf_prob > threshold:
            print(f"Random Forest Prediction: Fraudulent")
            fake_jobs_count += 1
        else:
            print(f"Random Forest Prediction: Real")
            real_jobs_count += 1

    print("-" * 50)  # For separating each job title prediction

# Output total count of real and fake jobs
print(f"Total Real Jobs: {real_jobs_count}")
print(f"Total Fake Jobs: {fake_jobs_count}")
