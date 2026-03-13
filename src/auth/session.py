"""
src/auth/session.py
────────────────────
Génération et vérification de JWT pour la session utilisateur.
"""

from datetime import datetime, timedelta
from jose import jwt, JWTError
from fastapi import Request, HTTPException, status
from config.settings import SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRE_DAYS


def create_token(user_data: dict) -> str:
    """Crée un JWT signé contenant les infos de l'utilisateur."""
    payload = {
        "sub":   user_data["email"],
        "name":  user_data["name"],
        "pic":   user_data.get("picture", ""),
        "gid":   user_data["google_id"],
        "exp":   datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Décode et valide un JWT. Lève une HTTPException si invalide."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session invalide ou expirée. Veuillez vous reconnecter.",
        )


def get_current_user(request: Request) -> dict:
    """
    Dépendance FastAPI : lit le JWT depuis le cookie 'session_token'.
    Retourne les données de l'utilisateur connecté.
    """
    token = request.cookies.get("session_token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Non authentifié. Veuillez vous connecter.",
        )
    return decode_token(token)
