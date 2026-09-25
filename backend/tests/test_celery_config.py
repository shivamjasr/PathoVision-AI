from backend.app.celery_app import celery_app


def test_analysis_queue_is_durable_and_late_acked():
    assert celery_app.conf.task_default_queue == "analysis"
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True
    assert celery_app.conf.worker_prefetch_multiplier == 1
    assert celery_app.conf.task_ignore_result is True
