from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import urllib.request

# Writable directories
DATA_DIR = "/tmp/airflow/data"
MODEL_DIR = "/tmp/airflow/models"

RAW_DATA_FILE = os.path.join(DATA_DIR, "titanic.csv")
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "titanic_processed.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "titanic_model.pkl")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Function to download the Titanic dataset if it doesn't exist
def download_dataset():
    if not os.path.exists(RAW_DATA_FILE):
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, RAW_DATA_FILE)
        print(f"Saved dataset to {RAW_DATA_FILE}")
    else:
        print(f"Dataset already exists at {RAW_DATA_FILE}")

# Function to preprocess the Titanic dataset
def preprocess():
    download_dataset()
    df = pd.read_csv(RAW_DATA_FILE)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df.fillna(0, inplace=True)
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"Preprocessed dataset saved at {PROCESSED_DATA_FILE}")

# Function to train the model
def train_model():
    df = pd.read_csv(PROCESSED_DATA_FILE)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc}")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved at {MODEL_FILE}")

# Define the DAG
with DAG(
    dag_id="train_titanic_dag",
    start_date=datetime(2025, 9, 17),
    schedule="@daily",  # modern Airflow uses 'schedule'
    catchup=False,
    tags=["ml", "titanic"]
) as dag:

    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    preprocess_task >> train_task
