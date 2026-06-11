"""
Serve Label Studio with a CONCURRENCY-CAPABLE server (waitress) instead of the
Django development server.

Why: `label-studio start` uses Django's dev server, which buckles under the
several concurrent requests the LS UI fires while annotating (save + load-next +
timer polling). On SQLite that surfaces as
    "Runtime error: Cannot operate on a closed database."
Waitress is multi-threaded and handles this cleanly. Static files (/static/) are
served by WhiteNoise since we're no longer using the dev server.

Env it honors (set by start_label_studio.ps1):
    LABEL_STUDIO_BASE_DATA_DIR, LABEL_STUDIO_HOST,
    LOCAL_FILES_SERVING_ENABLED, LOCAL_FILES_DOCUMENT_ROOT,
    LABEL_STUDIO_LOCAL_FILES_* (prefixed forms),
    LS_PORT (default 8090), LS_THREADS (default 8)
"""
import os
import sys

import label_studio

# Put Label Studio's package dir on sys.path so 'core.*' imports resolve.
LS_DIR = os.path.dirname(label_studio.__file__)
sys.path.insert(0, LS_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings.label_studio")

import django  # noqa: E402

django.setup()

# Make SQLite robust under the threaded server: WAL lets readers + the writer
# work concurrently, and a long busy_timeout means a thread WAITS for a lock
# instead of erroring ("database is locked" / "closed database"). Applied to
# every connection the moment it opens (one per worker thread).
from django.db.backends.signals import connection_created  # noqa: E402


def _sqlite_pragmas(sender, connection, **kwargs):
    if connection.vendor == "sqlite":
        cur = connection.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA busy_timeout=30000;")
        cur.close()


connection_created.connect(_sqlite_pragmas)

from django.conf import settings  # noqa: E402
from django.core.management import call_command  # noqa: E402
from django.core.wsgi import get_wsgi_application  # noqa: E402
from waitress import serve  # noqa: E402
from whitenoise import WhiteNoise  # noqa: E402

# --- Local-network access -------------------------------------------------- #
# Accept any Host header and trust the configured HOST (the LAN IP set by the
# start script) plus localhost for CSRF, so annotating from another device on
# the network doesn't fail the CSRF/Host checks.
settings.ALLOWED_HOSTS = ["*"]
_origins = set(getattr(settings, "CSRF_TRUSTED_ORIGINS", []) or [])
_host = os.environ.get("LABEL_STUDIO_HOST", "").rstrip("/")
if _host:
    _origins.add(_host)
_lsport = os.environ.get("LS_PORT", "8090")
for _h in ("localhost", "127.0.0.1"):
    _origins.add(f"http://{_h}:{_lsport}")
settings.CSRF_TRUSTED_ORIGINS = sorted(_origins)

# Ensure DB schema is current and static files exist (cheap if already done).
call_command("migrate", "--noinput", verbosity=0)
if not os.path.isdir(settings.STATIC_ROOT) or not os.listdir(settings.STATIC_ROOT):
    call_command("collectstatic", "--noinput", verbosity=0)

application = get_wsgi_application()
application = WhiteNoise(application, root=settings.STATIC_ROOT, prefix=settings.STATIC_URL)

if __name__ == "__main__":
    port = int(os.environ.get("LS_PORT", "8090"))
    threads = int(os.environ.get("LS_THREADS", "8"))
    print(f"Serving Label Studio via waitress on http://localhost:{port} (threads={threads})")
    print(f"  data dir : {os.environ.get('LABEL_STUDIO_BASE_DATA_DIR')}")
    serve(application, host="0.0.0.0", port=port, threads=threads)
