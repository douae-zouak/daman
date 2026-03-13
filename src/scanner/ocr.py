"""
src/scanner/ocr.py
───────────────────
Extraction de texte et de dates d'expiration depuis des images/PDF
en utilisant PaddleOCR (multilingue : français + arabe).
"""

import re
import os
import io
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
from config.settings import UPLOADS_DIR

# ─── Fix pour PaddleX / PaddleOCR >= 2.8 ─────────────────────────────────────
# Désactive la vérification de source qui fait crasher l'appli FastAPI
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


# ─── Initialisation lazy du modèle PaddleOCR ─────────────────────────────────
# On charge le modèle au premier appel pour ne pas bloquer le démarrage du serveur
_ocr = None

def _get_ocr():
    """Charge PaddleOCR à la première utilisation (lazy loading)."""
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        # Disable MKL-DNN and TensorRT to avoid OneDnnContext input Filter crash on Windows CPU
        _ocr = PaddleOCR(use_angle_cls=True, lang="fr", show_log=False, use_mkldnn=False, use_tensorrt=False)
    return _ocr


# ─── Mois (FR) ──────────────────────────────────────────────────────────────
MOIS_FR = {
    "jan": 1, "fév": 2, "feb": 2, "mar": 3, "avr": 4, "apr": 4,
    "mai": 5, "may": 5, "jun": 6, "jui": 7, "jul": 7, "aoû": 8,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "déc": 12, "dec": 12,
}

# Patterns de dates
DATE_PATTERNS = [
    r"(\d{2})[/\-\.](\d{2})[/\-\.](\d{4})",       # DD/MM/YYYY
    r"(\d{2})\s+(\w{3,})\s+(\d{4})",               # DD MON YYYY
    r"(\d{4})[/\-\.](\d{2})[/\-\.](\d{2})",        # YYYY-MM-DD
]


def _image_from_upload(file_bytes: bytes, filename: str) -> Image.Image:
    """Convertit les bytes d'un upload (JPG/PNG/PDF) en image PIL."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        try:
            from pdf2image import convert_from_bytes
            pages = convert_from_bytes(file_bytes, dpi=200)
            return pages[0]
        except ImportError:
            raise RuntimeError(
                "pdf2image n'est pas installé. "
                "Installez-le avec : pip install pdf2image"
            )
    return Image.open(io.BytesIO(file_bytes))


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Lance PaddleOCR sur l'image/PDF et retourne le texte brut extrait.
    """
    # S'assurer que le dossier uploads existe
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    image = _image_from_upload(file_bytes, filename)

    # Convertir en RGB si nécessaire (RGBA, palette, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # PaddleOCR accepte un numpy array directement (pas besoin de fichier temp)
    img_array = np.array(image)

    ocr = _get_ocr()
    result = ocr.ocr(img_array, cls=True)

    lines = []
    if result:
        for page in result:
            if page:
                for line in page:
                    text = line[1][0]       # [bbox, (texte, confiance)]
                    conf = line[1][1]       # score de confiance
                    if conf > 0.15:         # seuil minimum
                        lines.append(text)

    return "\n".join(lines)


def extract_expiry_date(text: str) -> datetime | None:
    """
    Cherche une date d'expiration dans le texte OCR.
    Priorité : 1) Mots-clés expiration, 2) MRZ passeport, 3) Dernière date trouvée.
    """
    # ── 1) Recherche par mots-clés (patterns plus précis) ──────────────────────
    expiry_keywords = [
        r"date\s+d['\'']expir",           # "Date d'expiration" (apostrophe)
        r"date\s+d[ée]xpir",              # "Date d'expir..." sans apostrophe
        r"date\s+of\s+expiry",            # English on passports
        r"expiry\s*[:\-]?",               # "Expiry:"
        r"expiration\s*[:\-]?",
        r"valable?\s+jusqu[\.']",          # "Valable jusqu'au"
        r"expire?\s+le",
        r"valid\s+until",
        r"تاريخ\s+الانتهاء",
        r"صالح\s+حتى",
        r"تاريخ\s+انتهاء\s+الصلاحية",
    ]

    text_lower = text.lower()

    for keyword in expiry_keywords:
        match = re.search(keyword, text_lower, re.IGNORECASE)
        if match:
            # Cherche la date dans les 100 chars qui suivent le mot-clé
            after = text[match.end(): match.end() + 100]
            date = _parse_date(after)
            if date:
                return date

    # ── 2) MRZ passeport (ligne du bas : YYMMDD en position fixe) ──────────────
    mrz_date = _extract_from_mrz(text)
    if mrz_date:
        return mrz_date

    # ── 3) Fallback : dernière date chronologique dans le texte ────────────────
    all_dates = _find_all_dates(text)
    return all_dates[-1] if all_dates else None


def _extract_from_mrz(text: str) -> datetime | None:
    """
    Tente de lire la date d'expiration depuis la zone MRZ d'un passeport.
    Ligne MRZ format TD3 : <numero><nationalite><naissance><sexe><EXPIRATION><...>
    Exemple : SP86153516MAR6404253M**1412071**A400400<<<<<<<<42
                                           ^^^^^^^
                                           AAMMJJ = 14/12/07 → 2014-12-07
    """
    # MRZ ligne 2 (44 chars) : positions 13-18 = date d'expiration YYMMDD
    mrz_pattern = r'[A-Z0-9<]{9}[A-Z]{3}(\d{6})[MF](\d{6})'
    m = re.search(mrz_pattern, text.upper())
    if m:
        expiry_raw = m.group(2)   # YYMMDD
        try:
            yy = int(expiry_raw[0:2])
            mm = int(expiry_raw[2:4])
            dd = int(expiry_raw[4:6])
            year = 2000 + yy if yy < 70 else 1900 + yy
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                return datetime(year, mm, dd)
        except (ValueError, IndexError):
            pass
    return None


def _parse_date(text: str) -> datetime | None:
    """Essaie de parser une date depuis une sous-chaîne."""
    for pattern in DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            continue
        try:
            groups = m.groups()
            if len(groups) == 3:
                a, b, c = groups
                if len(c) == 4:        # DD/MM/YYYY ou DD MON YYYY
                    day, month_raw, year = int(a), b, int(c)
                    month = _resolve_month(month_raw)
                else:                   # YYYY-MM-DD
                    year, month_raw, day = int(a), b, int(c)
                    month = _resolve_month(month_raw)
                if month and 1 <= int(day) <= 31 and 2000 <= year <= 2100:
                    return datetime(year, month, int(day))
        except (ValueError, TypeError):
            continue
    return None


def _resolve_month(raw) -> int | None:
    """Convertit un mois en chiffre ou texte en entier."""
    try:
        m = int(raw)
        return m if 1 <= m <= 12 else None
    except ValueError:
        key = raw[:3].lower()
        return MOIS_FR.get(key)


def _find_all_dates(text: str) -> list[datetime]:
    """Trouve toutes les dates valides dans le texte."""
    dates = []
    for pattern in DATE_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            d = _parse_date(m.group())
            if d and d.year > 2000:
                dates.append(d)
    return sorted(set(dates))


def save_upload(file_bytes: bytes, filename: str, user_id: int) -> str:
    """Sauvegarde le fichier uploadé dans UPLOADS_DIR et retourne le chemin."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    safe_name = f"user_{user_id}_{filename}"
    path = os.path.join(UPLOADS_DIR, safe_name)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path


# ─── Auto-détection du type de document ──────────────────────────────────────

def _normalize(text: str) -> str:
    """Met en majuscules et supprime les accents pour comparaison robuste."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", text.upper())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def detect_doc_type(text: str) -> str:
    """
    Détecte automatiquement le type de document depuis le texte OCR.
    Retourne un code parmi les types supportés.
    """
    norm = _normalize(text)   # sans accents, en majuscules
    orig = text.upper()       # original en majuscules (pour l'arabe)

    def _any(keywords):
        return any(kw in norm for kw in keywords)

    def _any_ar(keywords):
        return any(kw in orig for kw in keywords)

    # ── Passeport ──────────────────────────────────────────────────────────────
    if _any(["PASSEPORT", "PASSPORT"]):
        return "passeport"
    if re.search(r'\bP<[A-Z]{3}', orig):
        return "passeport"

    # ── Carte Nationale d'Identité (CNIE marocaine) ───────────────────────────
    # MRZ CIN commence par "IMA" (Identity Morocco)
    if re.search(r'\bI<MA[A-Z0-9<]', orig):
        return "cin"
    # Mots-clés exacts
    if _any([
        "CARTE NATIONALE", "CARTE D IDENTITE", "IDENTITE NATIONALE",
        "IDENTITY CARD", "CNIE", "CARTE D'IDENTITE",
        "CARTE NATIONALE D IDENTITE", "CARTE NATIONALE D'IDENTITE",
        "NATIONAL IDENTITY", "NATIONAL IDENTITY CARD",
    ]):
        return "cin"
    # Fragments OCR typiques sur la CIN marocaine (texte souvent découpé)
    if _any(["ROYAUME DU MAROC", "KINGDOM OF MOROCCO"]):
        # Si "PASSEPORT" n'est pas dedans, c'est probablement une CIN
        if not _any(["PASSEPORT", "PASSPORT"]):
            return "cin"
    # Regex flexibles pour capter les variations OCR (espaces, erreurs)
    cin_patterns = [
        r'CART\w*\s*NAT\w*',             # "CARTE NATIONALE" fragmenté
        r'CART\w*\s*D\s*ID\w*',           # "CARTE D IDENTITE" fragmenté
        r'IDEN\w*\s*NAT\w*',             # "IDENTITE NATIONALE"
        r'ROY\w*\s*DU\s*MAR\w*',         # "ROYAUME DU MAROC" fragmenté
        r'N\s*°?\s*C\.?\s*I\.?\s*N',     # "N° CIN" ou "N C.I.N"
        r'C\.?\s*N\.?\s*I\.?\s*E',       # "C.N.I.E"
        r'C\.?\s*I\.?\s*N\.?\s*E?',      # "C.I.N" ou "C.I.N.E"
        r'N\s*°?\s*(?:DE\s+)?(?:LA\s+)?CARTE',  # "N° de la carte"
    ]
    for pat in cin_patterns:
        if re.search(pat, norm):
            return "cin"
    # Arabe
    if _any_ar(["بطاقة التعريف الوطنية", "بطاقة الهوية الوطنية",
                "المملكة المغربية",          # "Royaume du Maroc" en arabe
                "بطاقة التعريف",              # forme courte
                "الهوية الوطنية"]):           # "identité nationale"
        return "cin"

    # ── Permis de conduire ─────────────────────────────────────────────────────
    if _any(["PERMIS DE CONDUIRE", "DRIVING LICENCE", "DRIVING LICENSE"]):
        return "permis_conduire"
    if _any_ar(["رخصة القيادة", "رخصة قيادة"]):
        return "permis_conduire"

    # ── Carte Grise / Immatriculation ─────────────────────────────────────────
    if _any(["CARTE GRISE", "IMMATRICULATION", "CERTIFICAT D IMMATRICULATION",
             "CERTIFICAT D'IMMATRICULATION"]):
        return "carte_grise"
    if _any_ar(["بطاقة الرمادية"]):
        return "carte_grise"

    # ── Assurance ─────────────────────────────────────────────────────────────
    if _any(["ASSURANCE", "CNSS", "RAMED", "MUTUELLE",
             "ATTESTATION D ASSURANCE", "POLICE D ASSURANCE"]):
        return "assurance"
    if _any_ar(["التأمين"]):
        return "assurance"

    # ── Acte de naissance ─────────────────────────────────────────────────────
    if _any(["ACTE DE NAISSANCE", "EXTRAIT DE NAISSANCE", "NAISSANCE"]):
        return "acte_naissance"
    if _any_ar(["عقد الازدياد", "شهادة الميلاد"]):
        return "acte_naissance"

    # ── Diplôme / Attestation scolaire ────────────────────────────────────────
    if _any(["DIPLOME", "DIPLOMA", "BACCALAUREAT", "MASTER",
             "ATTESTATION DE REUSSITE", "BULLETIN"]):
        return "diplome"
    if _any_ar(["شهادة", "دبلوم"]):
        return "diplome"

    # ── Contrat de bail ───────────────────────────────────────────────────────
    if _any(["CONTRAT DE BAIL", "CONTRAT DE LOCATION",
             "LOCATAIRE", "BAILLEUR"]):
        return "contrat_bail"
    if _any_ar(["عقد الكراء", "عقد إيجار"]):
        return "contrat_bail"

    # ── Contrat de travail ────────────────────────────────────────────────────
    if _any(["CONTRAT DE TRAVAIL", "CONTRAT D EMBAUCHE", "EMPLOYEUR"]):
        return "contrat_travail"
    if _any_ar(["عقد الشغل", "عقد العمل"]):
        return "contrat_travail"

    # ── Titre foncier ─────────────────────────────────────────────────────────
    if _any(["TITRE FONCIER", "CONSERVATION FONCIERE", "REQUISITION"]):
        return "titre_foncier"
    if _any_ar(["الرسم العقاري", "شهادة الملكية"]):
        return "titre_foncier"

    return "autre"




# ─── Extraction du nom de la personne ────────────────────────────────────────

def extract_person_name(text: str) -> str:
    """
    Tente d'extraire le nom complet de la personne depuis le texte OCR.
    Stratégie : MRZ passeport → nom en majuscules → pattern capitalisation.
    """
    # 1) Depuis la zone MRZ : P<MARNom<<Prénom<<<
    mrz_match = re.search(r'P<[A-Z]{3}([A-Z]+)<<([A-Z<]+)', text.upper())
    if mrz_match:
        last  = mrz_match.group(1).title()
        first = mrz_match.group(2).replace("<", " ").strip().split()[0].title()
        return f"{first} {last}"

    # 2) Lignes tout-en-majuscules qui ressemblent à un nom (2-3 mots, lettres uniquement)
    for line in text.split("\n"):
        line = line.strip()
        words = line.split()
        if (2 <= len(words) <= 3
                and all(re.match(r'^[A-ZÉÈÀÂÊÛÎÔÙÆŒ\-]+$', w) for w in words)
                and len(line) >= 5):
            return line.title()

    # 3) Patterns "Nom/Name : ..." ou "الاسم : ..."
    for pattern in [r"(?:nom|name)\s*[:/]\s*([^\n]{3,40})",
                    r"الاسم\s*[:/]\s*([^\n]{3,40})"]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().title()

    return ""

