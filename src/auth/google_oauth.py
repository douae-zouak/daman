"""
src/auth/google_oauth.py
─────────────────────────
Flux OAuth2 Google via Authlib.
"""

from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from config.settings import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET

config = Config()

oauth = OAuth(config)
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)
