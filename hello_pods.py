from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from datetime import datetime

with DAG(
    dag_id="hello_async_pods",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:

    task = KubernetesPodOperator(
        task_id="run-async-pod",
        name="async-hello",
        namespace="airflow",
        image="python:3.9",
        cmds=["python", "-c"],
        arguments=["print('Hello from a pod!')"],
        is_delete_operator_pod=True,
        get_logs=True,
    )
