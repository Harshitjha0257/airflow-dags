# train_titanic_dag.py
# Keep only lightweight imports at top
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import urllib.request

# Paths
DATA_DIR = "/tmp/airflow/data"
MODEL_DIR = "/tmp/airflow/models"
RAW_DATA_FILE = os.path.join(DATA_DIR, "titanic.csv")
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "titanic_processed.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "titanic_model")


# Task 1: Download dataset
def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    if not os.path.exists(RAW_DATA_FILE):
        urllib.request.urlretrieve(url, RAW_DATA_FILE)
        print(f"Downloaded dataset to {RAW_DATA_FILE}")
    else:
        print(f"Dataset already exists at {RAW_DATA_FILE}")


# Task 2: Preprocess data
def preprocess():
    import pandas as pd  # heavy import inside function
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(RAW_DATA_FILE)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).fillna(0)
    df.fillna(0, inplace=True)

    df['Title'] = df['Name'].str.extract(r', (\w+)\.')[0]
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss').replace(['Mme'], 'Mrs')
    title_mapping = {t: i for i, t in enumerate(df['Title'].unique())}
    df['Title'] = df['Title'].map(title_mapping)

    df = df.drop(columns=['Name', 'Ticket', 'Cabin'])
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"Preprocessed data saved at {PROCESSED_DATA_FILE}")


# Task 3: Hyperparameter tuning
def hyperparameter_tuning():
    import pandas as pd
    import itertools
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import mlflow
    import mlflow.sklearn

    df = pd.read_csv(PROCESSED_DATA_FILE)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Titanic_Hyperparameter_Tuning")

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, None],
        "min_samples_split": [2, 5]
    }

    best_accuracy, best_model, best_params = 0.0, None, None

    with mlflow.start_run(run_name="titanic_rf_tuning") as parent_run:
        for params in itertools.product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            with mlflow.start_run(nested=True):
                mlflow.sklearn.autolog()
                clf = RandomForestClassifier(**param_dict, random_state=42)
                clf.fit(X_train, y_train)
                acc = accuracy_score(y_test, clf.predict(X_test))

                mlflow.log_params(param_dict)
                mlflow.log_metric("accuracy", acc)

                if acc > best_accuracy:
                    best_accuracy, best_model, best_params = acc, clf, param_dict

        mlflow.sklearn.log_model(best_model, artifact_path="best_model")
        print(f"Best model params: {best_params}, Accuracy: {best_accuracy}")


# ---------------- DAG Definition ----------------
default_args = {"retries": 1}

with DAG(
    dag_id="train_titanic_dag",
    start_date=datetime(2025, 9, 17),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["ml", "titanic"],
) as dag:

    download_task = PythonOperator(
        task_id="download_dataset",
        python_callable=download_dataset,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess,
    )

    tuning_task = PythonOperator(
        task_id="hyperparameter_tuning",
        python_callable=hyperparameter_tuning,
    )

    download_task >> preprocess_task >> tuning_task
