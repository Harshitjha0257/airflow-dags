from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="titanic_train_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    train_default = BashOperator(
        task_id="train_default",
        bash_command="python /usr/local/airflow/dags/train.py"
    )

    train_tuned = BashOperator(
        task_id="train_tuned",
        bash_command="python /usr/local/airflow/dags/train.py --n_estimators 200 --max_depth 10"
    )

    train_default >> train_tuned
