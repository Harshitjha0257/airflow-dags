from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Function to load and preprocess the Titanic dataset
def preprocess():
    df = pd.read_csv("/opt/airflow/dags/repo/data/titanic.csv")  # adjust path if needed
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df.fillna(0, inplace=True)
    df.to_csv("/opt/airflow/dags/repo/data/titanic_processed.csv", index=False)

# Function to train the model
def train_model():
    df = pd.read_csv("/opt/airflow/dags/repo/data/titanic_processed.csv")
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc}")
    with open("/opt/airflow/dags/repo/models/titanic_model.pkl", "wb") as f:
        pickle.dump(model, f)

# Define the DAG
with DAG(
    dag_id="train_titanic_dag",
    start_date=datetime(2025, 9, 17),
    schedule="@daily",      # <-- modern Airflow uses 'schedule' instead of 'schedule_interval'
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
