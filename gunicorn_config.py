# Gunicorn Configuration for Production Deployment
# This provides better performance and stability than Flask's built-in server

import multiprocessing

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = 1  # Single worker to prevent memory issues
worker_class = "sync"
worker_connections = 1000
timeout = 1200  # 20 minutes for long video processing
keepalive = 5
max_requests = 10  # Restart worker after 10 requests to prevent memory leaks
max_requests_jitter = 2

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "videocrafter"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Request limits
limit_request_line = 0
limit_request_fields = 100
limit_request_field_size = 0
