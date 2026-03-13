"""
Modèles SQLAlchemy : User, Document, Alert, ChatMessage.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from src.db.database import Base


class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    email      = Column(String, unique=True, index=True, nullable=False)
    name       = Column(String, nullable=False)
    picture    = Column(String, nullable=True)   # URL photo Google
    google_id  = Column(String, unique=True, nullable=False)
    is_premium = Column(Boolean, default=False)  # True = usage illimité
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="user", cascade="all, delete")
    messages  = relationship("ChatMessage", back_populates="user", cascade="all, delete")


class Document(Base):
    __tablename__ = "documents"

    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=False)
    type             = Column(String, nullable=False)
    nom              = Column(String, nullable=False)
    numero           = Column(String, nullable=True)
    date_expiration  = Column(DateTime, nullable=True)
    fichier_path     = Column(String, nullable=True)
    texte_ocr        = Column(Text, nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow)

    user   = relationship("User", back_populates="documents")
    alerts = relationship("Alert", back_populates="document", cascade="all, delete")


class Alert(Base):
    __tablename__ = "alerts"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    jours_avant = Column(Integer, nullable=False)   # 30, 15, ou 7
    date_envoi  = Column(DateTime, nullable=True)
    envoye      = Column(Boolean, default=False)

    document = relationship("Document", back_populates="alerts")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    role       = Column(String, nullable=False)   # "user" | "assistant"
    content    = Column(Text, nullable=False)
    source     = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="messages")
