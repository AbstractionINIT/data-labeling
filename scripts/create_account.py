r"""
Create (or update) a Label Studio account + API token WITHOUT the browser, and
write the token into scripts/env.ps1 and scripts/env.sh automatically.

Run it directly (it self-bootstraps Django); set the data dir first so it edits
the right database:

    # PowerShell
    $env:LABEL_STUDIO_BASE_DATA_DIR="D:\annotation\data\.ls-data"
    .\.venv\Scripts\python.exe scripts\create_account.py

Credentials come from env vars (with defaults):
    LS_EMAIL     (default admin@local.dev)
    LS_PASSWORD  (default Annotate123!)
"""
import os
import re
from pathlib import Path

# Self-bootstrap Django so this works as a plain script (not only inside `shell`).
# Label Studio's settings import top-level packages like `core`, so its package
# directory must be on sys.path (this is what the `label-studio` CLI does).
import sys  # noqa: E402

import label_studio  # noqa: E402

sys.path.insert(0, os.path.dirname(label_studio.__file__))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings.label_studio")
import django  # noqa: E402
from django.apps import apps  # noqa: E402

if not apps.ready:
    django.setup()

from django.contrib.auth import get_user_model  # noqa: E402
from organizations.models import Organization, OrganizationMember  # noqa: E402
from rest_framework.authtoken.models import Token  # noqa: E402

EMAIL = os.environ.get("LS_EMAIL", "admin@local.dev")
PASSWORD = os.environ.get("LS_PASSWORD", "Annotate123!")

U = get_user_model()
user = U.objects.filter(email=EMAIL).first()
created = user is None
if created:
    user = U(email=EMAIL, username=EMAIL)
user.username = user.username or EMAIL
user.set_password(PASSWORD)
user.is_active = True
user.is_staff = True
user.save()

# Ensure the user belongs to an organization and has it active (the API needs it).
org = user.active_organization or Organization.objects.first()
if org is None:
    try:
        org = Organization.create_organization(created_by=user, title="Label Studio")
    except Exception:
        org = Organization.objects.create(created_by=user, title="Label Studio")
OrganizationMember.objects.get_or_create(user=user, organization=org)
if user.active_organization_id != org.id:
    user.active_organization = org
    user.save(update_fields=["active_organization"])

# LS 1.23+ disables legacy "Authorization: Token <key>" auth by default; our
# backend + bootstrap rely on it, so enable it for this organization.
try:
    from jwt_auth.models import JWTSettings
    js, _ = JWTSettings.objects.get_or_create(organization=org)
    changed = False
    if hasattr(js, "legacy_api_tokens_enabled") and not js.legacy_api_tokens_enabled:
        js.legacy_api_tokens_enabled = True
        changed = True
    if hasattr(js, "api_tokens_enabled") and not js.api_tokens_enabled:
        js.api_tokens_enabled = True
        changed = True
    if changed:
        js.save()
    print(f"LEGACY_TOKENS_ENABLED org={org.id}")
except Exception as e:
    print(f"WARN_JWT_SETTINGS {e!r}")

# Legacy API token (our backend/bootstrap use 'Authorization: Token <key>').
token, _ = Token.objects.get_or_create(user=user)


def _patch(path: str, pattern: str, replacement: str):
    p = Path(path)
    if not p.exists():
        return
    txt = p.read_text(encoding="utf-8")
    new = re.sub(pattern, replacement, txt)
    if new != txt:
        p.write_text(new, encoding="utf-8")


_patch("scripts/env.ps1",
       r'\$env:LABEL_STUDIO_API_KEY\s*=\s*"[^"]*"',
       f'$env:LABEL_STUDIO_API_KEY = "{token.key}"')
_patch("scripts/env.sh",
       r'export LABEL_STUDIO_API_KEY="[^"]*"',
       f'export LABEL_STUDIO_API_KEY="{token.key}"')

print(f"ACCOUNT_READY email={EMAIL} created={created} org={org.id} token={token.key}")
