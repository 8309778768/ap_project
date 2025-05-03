# Fake Job Detection

This project aims to classify job postings as **real** or **fake** using machine learning models. The model is trained using a dataset of job postings with labels indicating whether a job is fraudulent (fake) or legitimate (real).

## Project Structure

- `dataset/`: Contains the job postings dataset (CSV file).
- `model/`: Contains the trained model (saved using `pickle`).
- `src/`: Contains the Python scripts for training, testing, and making predictions.
- `requirements.txt`: Lists the Python libraries required to run this project.
- `README.md`: Project documentation.

## Dataset

The dataset contains job postings with the following columns:

- `job_id`: Unique identifier for each job posting.
- `title`: Title of the job posting.
- `location`: Job location.
- `description`: Job description.
- `fraudulent`: Label indicating if the job is real (0) or fake (1).

## Installation and Setup

### Requirements

To run this project, you need Python 3.x and the following libraries:
- `pandas`
- `scikit-learn`
- `nltk`
- `matplotlib`
- `seaborn`

### Installation

1. Clone or download the project files.
2. Install the required libraries:
   ```bash
   pip3 install -r requirements.txt
