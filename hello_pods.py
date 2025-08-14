from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from datetime import datetime

with DAG(
    dag_id="hello_pods",
    start_date=datetime(2025, 8, 14),
    schedule=None,
    catchup=False,
) as dag:

    hello_task = KubernetesPodOperator(
        task_id="hello-task",
        name="hello-pod",
        namespace="airflowdags",
        image="python:3.9-slim",
        cmds=["python", "-c"],
        arguments=["print('Hello from Kubernetes Pod')"],
        is_delete_operator_pod=True,
        kubernetes_conn_id="kubernetes_default"  # Explicit connection
    )

    hello_task
