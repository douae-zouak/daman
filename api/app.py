"""
api/app.py — Interface FastAPI complète
Routes : auth Google, chatbot RAG, scan documents, alertes
"""

import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session

from src.pipeline import RAGPipeline
from src.db.database import init_db, get_db
from src.db.models import User, Document, ChatMessage, Alert
from src.auth.google_oauth import oauth
from src.auth.session import create_token, get_current_user
from src.alerts.scheduler import start_scheduler
from config.settings import SECRET_KEY, APP_URL

# ─── Constantes ──────────────────────────────────────────────────────────────
FREE_DOC_LIMIT = 3   # nombre max de documents pour les utilisateurs gratuits

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Assistant Administratif Maroc",
    description="Chatbot RAG + scan documents + alertes renouvellement",
    version="2.0.0",
)

# Servir les fichiers statiques (images, etc.) depuis le dossier api/
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__))), name="static")

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pipeline RAG ────────────────────────────────────────────────────────────
pipeline = RAGPipeline()

@app.on_event("startup")
async def startup():
    init_db()
    try:
        pipeline.load()
        print("✅ Pipeline RAG prêt.")
    except Exception as e:
        print(f"❌ ERREUR pipeline : {e}")
        print("👉 Lancez 'python main.py --build' d'abord.")
    start_scheduler()


# ─── Pages HTML ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("api/home.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    token = request.cookies.get("session_token")
    if not token:
        return RedirectResponse("/")
    with open("api/dashboard.html", "r", encoding="utf-8") as f:
        return f.read()


# ─── AUTH GOOGLE OAUTH ───────────────────────────────────────────────────────

@app.get("/auth/login")
async def auth_login(request: Request):
    """Redirige vers la page de connexion Google."""
    redirect_uri = f"{APP_URL}/auth/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback")
async def auth_callback(request: Request, db: Session = Depends(get_db)):
    """Reçoit le token Google, crée/met à jour l'utilisateur en DB, génère un JWT."""
    token     = await oauth.google.authorize_access_token(request)
    user_info = token.get("userinfo")

    if not user_info:
        raise HTTPException(status_code=400, detail="Informations Google introuvables.")

    # Upsert utilisateur en DB
    user = db.query(User).filter(User.google_id == user_info["sub"]).first()
    if not user:
        user = User(
            email=user_info["email"],
            name=user_info["name"],
            picture=user_info.get("picture", ""),
            google_id=user_info["sub"],
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # Génère JWT et le stocke dans un cookie
    jwt_token = create_token({
        "email":     user.email,
        "name":      user.name,
        "picture":   user.picture,
        "google_id": user.google_id,
    })

    response = RedirectResponse("/dashboard")
    response.set_cookie("session_token", jwt_token, httponly=True, max_age=604800)
    return response


@app.get("/auth/logout")
async def auth_logout():
    response = RedirectResponse("/")
    response.delete_cookie("session_token")
    return response


@app.get("/auth/me")
async def auth_me(user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Retourne les infos de l'utilisateur connecté + statut premium."""
    db_user = db.query(User).filter(User.email == user["sub"]).first()
    doc_count = db.query(Document).filter(Document.user_id == db_user.id).count() if db_user else 0
    return {
        "email": user["sub"],
        "name": user["name"],
        "picture": user["pic"],
        "is_premium": db_user.is_premium if db_user else False,
        "doc_count": doc_count,
        "doc_limit": FREE_DOC_LIMIT,
    }


# ─── CHATBOT RAG ─────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str
    langue: str = "fr"


@app.post("/ask")
async def ask(
    request_data: QuestionRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    if not request_data.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    # Utilisateur connecté (optionnel — on sauvegarde si connecté)
    user_id = None
    try:
        user_info = get_current_user(request)
        db_user = db.query(User).filter(User.email == user_info["sub"]).first()
        if db_user:
            user_id = db_user.id
    except Exception:
        pass

    try:
        result = pipeline.query(request_data.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Sauvegarde l'historique en DB si connecté
    if user_id:
        db.add(ChatMessage(user_id=user_id, role="user",      content=request_data.question))
        db.add(ChatMessage(user_id=user_id, role="assistant", content=result["answer"], source=result["source"]))
        db.commit()

    return {
        "question": request_data.question,
        "reponse":  result["answer"],
        "source":   result["source"],
        "metadata": result.get("metadata", {}),
    }


@app.get("/history")
async def get_history(user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Retourne l'historique de chat de l'utilisateur connecté."""
    db_user = db.query(User).filter(User.email == user["sub"]).first()
    if not db_user:
        return []
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == db_user.id)
        .order_by(ChatMessage.created_at)
        .all()
    )
    return [{"role": m.role, "content": m.content, "source": m.source} for m in messages]


# ─── DOCUMENTS / SCAN OCR ────────────────────────────────────────────────────

@app.post("/documents/scan")
async def scan_document(
    file: UploadFile = File(...),
    type_doc: str = Form(""),   # Optionnel : auto-détecté si vide
    nom: str = Form(""),        # Optionnel : auto-généré si vide
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload + OCR d'un document. Détecte automatiquement type et nom si non fournis."""
    from src.scanner.ocr import (
        extract_text, extract_expiry_date, save_upload,
        detect_doc_type, extract_person_name
    )

    db_user = db.query(User).filter(User.email == user["sub"]).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")

    # ── Vérification limite gratuite ──────────────────────────────────────────
    if not db_user.is_premium:
        doc_count = db.query(Document).filter(Document.user_id == db_user.id).count()
        if doc_count >= FREE_DOC_LIMIT:
            raise HTTPException(
                status_code=403,
                detail=f"Limite atteinte ! Vous avez {doc_count}/{FREE_DOC_LIMIT} documents. Passez à la version Premium pour un usage illimité."
            )

    file_bytes = await file.read()
    texte      = extract_text(file_bytes, file.filename)
    date_exp   = extract_expiry_date(texte)
    path       = save_upload(file_bytes, file.filename, db_user.id)

    # ── Auto-détection si champs non fournis ─────────────────────────────────
    auto_detected = {}
    if not type_doc:
        type_doc = detect_doc_type(texte)
        auto_detected["type"] = type_doc

    if not nom:
        person_name = extract_person_name(texte)
        type_labels = {
            "passeport":       "Passeport",
            "cin":             "CIN",
            "carte_grise":     "Carte Grise",
            "permis_conduire": "Permis de conduire",
            "assurance":       "Assurance",
            "acte_naissance":  "Acte de naissance",
            "diplome":         "Diplôme",
            "contrat_bail":    "Contrat de bail",
            "contrat_travail": "Contrat de travail",
            "titre_foncier":   "Titre foncier",
            "autre":           "Document",
        }
        label = type_labels.get(type_doc, "Document")
        nom = f"{label} {person_name}".strip() if person_name else label
        auto_detected["nom"] = nom

    doc = Document(
        user_id=db_user.id,
        type=type_doc,
        nom=nom,
        date_expiration=date_exp,
        fichier_path=path,
        texte_ocr=texte,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    return {
        "id":              doc.id,
        "nom":             doc.nom,
        "type":            doc.type,
        "date_expiration": date_exp.strftime("%d/%m/%Y") if date_exp else None,
        "texte_ocr":       texte[:300] + "..." if len(texte) > 300 else texte,
        "auto_detected":   auto_detected,
    }


@app.get("/documents")
async def list_documents(user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Liste tous les documents de l'utilisateur connecté."""
    db_user = db.query(User).filter(User.email == user["sub"]).first()
    if not db_user:
        return []
    docs = db.query(Document).filter(Document.user_id == db_user.id).all()
    return [
        {
            "id":              d.id,
            "nom":             d.nom,
            "type":            d.type,
            "date_expiration": d.date_expiration.strftime("%d/%m/%Y") if d.date_expiration else None,
            "created_at":      d.created_at.strftime("%d/%m/%Y"),
        }
        for d in docs
    ]


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int, user=Depends(get_current_user), db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user["sub"]).first()
    doc = db.query(Document).filter(Document.id == doc_id, Document.user_id == db_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document introuvable.")
    db.delete(doc)
    db.commit()
    return {"message": "Document supprimé."}


# ─── ALERTES ─────────────────────────────────────────────────────────────────

class ExpirationUpdate(BaseModel):
    date_expiration: str   # format ISO YYYY-MM-DD


@app.post("/alerts/test")
async def test_alerts(user=Depends(get_current_user)):
    """Déclenche manuellement la vérification des alertes (pour tests)."""
    from src.alerts.scheduler import check_and_send_alerts
    check_and_send_alerts()
    return {"message": "Vérification des alertes effectuée."}


@app.put("/documents/{doc_id}/expiration")
async def update_expiration(
    doc_id: int,
    payload: ExpirationUpdate,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Modifie la date d'expiration d'un document (pour tester les alertes)."""
    db_user = db.query(User).filter(User.email == user["sub"]).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")

    doc = db.query(Document).filter(Document.id == doc_id, Document.user_id == db_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document introuvable.")

    try:
        new_date = datetime.strptime(payload.date_expiration, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Format de date invalide. Utilisez YYYY-MM-DD.")

    doc.date_expiration = new_date
    db.commit()
    db.refresh(doc)

    return {
        "id": doc.id,
        "nom": doc.nom,
        "date_expiration": new_date.strftime("%d/%m/%Y"),
        "message": f"Date d'expiration mise à jour → {new_date.strftime('%d/%m/%Y')}",
    }


@app.get("/alerts")
async def list_alerts(user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Liste les alertes envoyées pour l'utilisateur connecté."""
    db_user = db.query(User).filter(User.email == user["sub"]).first()
    if not db_user:
        return []

    alerts = (
        db.query(Alert)
        .filter(Alert.user_id == db_user.id)
        .order_by(Alert.date_envoi.desc())
        .all()
    )
    result = []
    for a in alerts:
        doc = db.query(Document).filter(Document.id == a.document_id).first()
        result.append({
            "id": a.id,
            "document_id": a.document_id,
            "document_nom": doc.nom if doc else "—",
            "jours_avant": a.jours_avant,
            "date_envoi": a.date_envoi.strftime("%d/%m/%Y %H:%M") if a.date_envoi else None,
            "envoye": a.envoye,
        })
    return result


@app.delete("/alerts/{doc_id}")
async def reset_alerts(doc_id: int, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Réinitialise (supprime) les alertes d'un document pour pouvoir re-tester."""
    db_user = db.query(User).filter(User.email == user["sub"]).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")

    doc = db.query(Document).filter(Document.id == doc_id, Document.user_id == db_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document introuvable.")

    deleted = db.query(Alert).filter(Alert.document_id == doc_id).delete()
    db.commit()
    return {"message": f"{deleted} alerte(s) réinitialisée(s) pour « {doc.nom} »."}


# ─── HEALTH ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}