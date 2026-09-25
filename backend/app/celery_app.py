from celery import Celery
from kombu import Queue

from backend.app.config import CELERY_BROKER_URL, CELERY_QUEUE


celery_app = Celery(
    "pathovision",
    broker=CELERY_BROKER_URL,
    include=["backend.app.tasks"],
)

celery_app.conf.update(
    task_default_queue=CELERY_QUEUE,
    task_default_exchange=CELERY_QUEUE,
    task_default_routing_key=CELERY_QUEUE,
    task_queues=(
        Queue(
            CELERY_QUEUE,
            durable=True,
            queue_arguments={"x-queue-type": "quorum"},
        ),
    ),
    broker_transport_options={"confirm_publish": True},
    task_default_delivery_mode="persistent",
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_ignore_result=True,
    worker_send_task_events=True,
    task_send_sent_event=True,
    task_track_started=True,
    task_time_limit=6 * 60 * 60,
    task_soft_time_limit=5 * 60 * 60,
)
