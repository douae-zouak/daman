"""
Connexion SQLite + création automatique des tables au démarrage.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config.settings import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # nécessaire pour SQLite + FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db():
    """Crée toutes les tables si elles n'existent pas encore."""
    from src.db import models  # import ici pour éviter les imports circulaires
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dépendance FastAPI pour injecter une session DB dans chaque route."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
