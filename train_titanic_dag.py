from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
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

# Function to download dataset
def download_dataset():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    if not os.path.exists(RAW_DATA_FILE):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, RAW_DATA_FILE)
        print(f"Saved dataset to {RAW_DATA_FILE}")
    else:
        print(f"Dataset already exists at {RAW_DATA_FILE}")

# Function to preprocess dataset
def preprocess():
    df = pd.read_csv(RAW_DATA_FILE)

    # Encode 'Sex'
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Handle missing values
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df['Embarked'] = df['Embarked'].fillna(0)
    df.fillna(0, inplace=True)

    # Extract Title from Name
    df['Title'] = df['Name'].str.extract(r', (\w+)\.')[0]
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss').replace(['Mme'], 'Mrs')
    title_mapping = {t: i for i, t in enumerate(df['Title'].unique())}
    df['Title'] = df['Title'].map(title_mapping)

    # Drop columns not needed for training
    df = df.drop(columns=['Name', 'Ticket', 'Cabin'])

    # Save preprocessed data
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"Preprocessed dataset saved at {PROCESSED_DATA_FILE}")

# Function to train model
def train_model():
    df = pd.read_csv(PROCESSED_DATA_FILE)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc}")

    # Save model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved at {MODEL_FILE}")

# Define DAG
with DAG(
    dag_id="train_titanic_dag",
    start_date=datetime(2025, 9, 17),
    schedule="@daily",
    catchup=False,
    tags=["ml", "titanic"]
) as dag:

    download_task = PythonOperator(
        task_id="download_dataset",
        python_callable=download_dataset
    )

    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    # Task dependencies
    download_task >> preprocess_task >> train_task
