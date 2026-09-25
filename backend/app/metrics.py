from prometheus_client import Counter, Histogram


HTTP_REQUESTS_TOTAL = Counter(
    "pathovision_http_requests_total",
    "HTTP requests handled by the API",
    ["method", "path", "status"],
)

HTTP_LATENCY_SECONDS = Histogram(
    "pathovision_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)

JOBS_SUBMITTED_TOTAL = Counter(
    "pathovision_jobs_submitted_total",
    "Analysis jobs submitted",
    ["mode"],
)

JOBS_COMPLETED_TOTAL = Counter(
    "pathovision_jobs_completed_total",
    "Analysis jobs completed",
    ["mode"],
)

JOBS_FAILED_TOTAL = Counter(
    "pathovision_jobs_failed_total",
    "Analysis jobs failed",
    ["mode"],
)
