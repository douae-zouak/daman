"""
src/alerts/scheduler.py
────────────────────────
Scheduler qui vérifie chaque jour les documents proches de l'expiration
et envoie des emails de rappel J-30, J-15, J-7.
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, date
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from sqlalchemy.orm import Session

from src.db.database import SessionLocal
from src.db.models import Document, Alert, User
from config.settings import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, APP_URL


# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constantes ───────────────────────────────────────────────────────────────

DELAIS: list[int] = [30, 15, 7]   # jours avant expiration


# ── Email ─────────────────────────────────────────────────────────────────────

def send_email(to: str, subject: str, html_body: str) -> bool:
    """
    Envoie un email via SMTP (TLS).
    Retourne True si succès, False sinon.
    """
    if not to or "@" not in to:
        logger.warning("Adresse email invalide : %s", to)
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_USER
        msg["To"]      = to
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, to, msg.as_string())

        logger.info("Email envoyé à %s | Sujet : %s", to, subject)
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error("Échec d'authentification SMTP — vérifiez SMTP_USER / SMTP_PASSWORD")
    except smtplib.SMTPConnectError:
        logger.error("Impossible de se connecter à %s:%s", SMTP_HOST, SMTP_PORT)
    except smtplib.SMTPException as exc:
        logger.error("Erreur SMTP : %s", exc)
    except OSError as exc:
        logger.error("Erreur réseau lors de l'envoi à %s : %s", to, exc)

    return False


# ── Template HTML ─────────────────────────────────────────────────────────────

def build_email_body(
    user_name: str,
    doc_nom: str,
    doc_type: str,
    date_exp: datetime,
    jours: int,
) -> str:
    """Génère le corps HTML de l'email d'alerte."""
    date_str = date_exp.strftime("%d/%m/%Y")
    urgency  = "🔴" if jours <= 7 else ("🟠" if jours <= 15 else "🟡")

    return f"""
    <html>
    <body style="font-family:Arial,sans-serif;color:#333;max-width:600px;margin:auto">

      <div style="background:#1a1a2e;padding:24px;border-radius:12px 12px 0 0">
        <h2 style="color:white;margin:0">🇲🇦 Assistant Administratif Maroc</h2>
      </div>

      <div style="padding:24px;background:#f9f9f9;border-radius:0 0 12px 12px">

        <p>Bonjour <strong>{user_name}</strong>,</p>

        <p>
          {urgency} Votre document <strong>{doc_nom}</strong>
          (<em>{doc_type}</em>)
          expire dans <strong>{jours} jour(s)</strong>.
        </p>

        <div style="background:white;border-left:4px solid #e63946;
                    padding:16px;margin:16px 0;border-radius:4px">
          <p><strong>Date d'expiration :</strong> {date_str}</p>
        </div>

        <p>Pensez à le renouveler pour éviter tout problème administratif.</p>

        <a href="{APP_URL}/dashboard"
           style="display:inline-block;background:#e63946;color:white;
                  padding:12px 24px;border-radius:8px;text-decoration:none;
                  margin-top:8px">
          Voir mes documents →
        </a>

      </div>
    </body>
    </html>
    """


# ── Helpers ───────────────────────────────────────────────────────────────────

def _alert_already_sent(db: Session, document_id: int, jours: int) -> bool:
    """Vérifie si une alerte a déjà été envoyée pour ce document / délai."""
    return (
        db.query(Alert)
        .filter(
            Alert.document_id == document_id,
            Alert.jours_avant  == jours,
            Alert.envoye       == True,
        )
        .first()
        is not None
    )


def _get_user(db: Session, user_id: int) -> Optional[User]:
    """Récupère un utilisateur par son id."""
    return db.query(User).filter(User.id == user_id).first()


def _get_docs_expiring_on(db: Session, target: date) -> list[Document]:
    """Retourne les documents dont la date d'expiration tombe sur `target`."""
    return (
        db.query(Document)
        .filter(
            Document.date_expiration.isnot(None),
            Document.date_expiration >= datetime(target.year, target.month, target.day, 0, 0, 0),
            Document.date_expiration <  datetime(target.year, target.month, target.day, 23, 59, 59),
        )
        .all()
    )


def _record_alert(db: Session, user_id: int, document_id: int, jours: int) -> None:
    """Persiste une alerte en base de données."""
    alert = Alert(
        user_id     = user_id,
        document_id = document_id,
        jours_avant = jours,
        date_envoi  = datetime.utcnow(),
        envoye      = True,
    )
    db.add(alert)
    db.commit()


# ── Job principal ─────────────────────────────────────────────────────────────

def check_and_send_alerts() -> None:
    """
    Job quotidien du scheduler :
    — Récupère les documents qui expirent dans 30, 15 ou 7 jours.
    — Envoie un email si l'alerte n'a pas encore été envoyée.
    """
    logger.info("=== Début vérification des alertes ===")

    db: Session = SessionLocal()
    sent = skipped = errors = 0

    try:
        today = datetime.utcnow().date()

        for jours in DELAIS:
            target = today + timedelta(days=jours)
            docs   = _get_docs_expiring_on(db, target)

            logger.info("J-%d → %d document(s) expirant le %s", jours, len(docs), target)

            for doc in docs:

                # Alerte déjà envoyée ?
                if _alert_already_sent(db, doc.id, jours):
                    logger.debug("Alerte J-%d déjà envoyée pour doc#%d — ignoré", jours, doc.id)
                    skipped += 1
                    continue

                # Utilisateur existe ?
                user = _get_user(db, doc.user_id)
                if not user:
                    logger.warning("Utilisateur introuvable pour doc#%d (user_id=%s)", doc.id, doc.user_id)
                    errors += 1
                    continue

                # Construction et envoi
                html    = build_email_body(user.name, doc.nom, doc.type, doc.date_expiration, jours)
                subject = f"⚠️ Document « {doc.nom} » expire dans {jours} jour(s)"
                success = send_email(to=user.email, subject=subject, html_body=html)

                if success:
                    _record_alert(db, user.id, doc.id, jours)
                    logger.info("✅ Alerte J-%d → %s (doc : %s)", jours, user.email, doc.nom)
                    sent += 1
                else:
                    errors += 1

    except Exception as exc:
        logger.exception("Erreur inattendue dans check_and_send_alerts : %s", exc)

    finally:
        db.close()
        logger.info(
            "=== Fin vérification | envoyées=%d  ignorées=%d  erreurs=%d ===",
            sent, skipped, errors,
        )


# ── Scheduler ─────────────────────────────────────────────────────────────────

def _on_job_event(event) -> None:
    """Callback APScheduler pour journaliser les succès/erreurs de jobs."""
    if event.exception:
        logger.error("Le job '%s' a échoué : %s", event.job_id, event.exception)
    else:
        logger.info("Le job '%s' s'est terminé avec succès", event.job_id)


def start_scheduler() -> BackgroundScheduler:
    """
    Initialise et démarre le scheduler APScheduler.
    Vérification quotidienne à 09h00 UTC.
    Retourne l'instance du scheduler.
    """
    scheduler = BackgroundScheduler(timezone="UTC")

    scheduler.add_job(
        check_and_send_alerts,
        trigger  = "cron",
        hour     = 9,
        minute   = 0,
        id       = "daily_alerts",
        name     = "Vérification quotidienne des alertes",
        replace_existing = True,
    )

    scheduler.add_listener(_on_job_event, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    scheduler.start()
    logger.info("✅ Scheduler démarré — vérification quotidienne à 09h00 UTC")
    return scheduler