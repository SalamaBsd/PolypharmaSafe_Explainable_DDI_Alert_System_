"""
ClearDDI — moteur d'inférence
Charge les artefacts réels du pipeline PolypharmaSafe (XGBoost binaire + multiclasse,
SHAP, mappings nom -> DrugBank ID -> SMILES) et expose les fonctions utilisées par l'API.

Prototype académique d'aide à la décision — non destiné à un usage clinique réel.
"""

from __future__ import annotations

import json
import pickle
import re
import difflib
import warnings
from pathlib import Path
from functools import lru_cache

import numpy as np
import joblib
import shap
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem, Draw

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


# ──────────────────────────────────────────────────────────────────────────
# Chargement des artefacts (une seule fois, au démarrage du serveur)
# ──────────────────────────────────────────────────────────────────────────

def _load_artifacts():
    with open(ARTIFACTS_DIR / "name_to_id.pkl", "rb") as f:
        name_to_id = pickle.load(f)
    with open(ARTIFACTS_DIR / "id_to_smiles.pkl", "rb") as f:
        id_to_smiles = pickle.load(f)
    with open(ARTIFACTS_DIR / "inference_config.json", encoding="utf-8") as f:
        config = json.load(f)
    with open(ARTIFACTS_DIR / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

    binary_model = joblib.load(ARTIFACTS_DIR / "ddi_binary_model.joblib")
    multi_model = joblib.load(ARTIFACTS_DIR / "ddi_multiclass_model.joblib")
    label_encoder = joblib.load(ARTIFACTS_DIR / "ddi_multiclass_label_encoder.joblib")

    return {
        "name_to_id": name_to_id,
        "id_to_smiles": id_to_smiles,
        "config": config,
        "metrics": metrics,
        "binary_model": binary_model,
        "multi_model": multi_model,
        "label_encoder": label_encoder,
    }


_A = _load_artifacts()
NAME_TO_ID: dict[str, str] = _A["name_to_id"]
ID_TO_SMILES: dict[str, str] = _A["id_to_smiles"]
CONFIG: dict = _A["config"]
METRICS: dict = _A["metrics"]
BINARY_MODEL = _A["binary_model"]
MULTI_MODEL = _A["multi_model"]
LABEL_ENCODER = _A["label_encoder"]

FP_RADIUS = CONFIG["fp_radius"]
FP_BITS = CONFIG["fp_bits"]
DECISION_THRESHOLD = CONFIG.get("decision_threshold", 0.5)

_MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
_SHAP_EXPLAINER = shap.TreeExplainer(BINARY_MODEL)

# Liste triée des noms connus, pour la recherche / les suggestions
_ALL_NAMES_SORTED = sorted(NAME_TO_ID.keys())

# Alias courants (français / noms commerciaux fréquents) vers le nom DrugBank
# officiel utilisé comme clé dans name_to_id. Complète la recherche exacte,
# car la base utilise la nomenclature DrugBank (souvent le nom anglais de la
# substance active).
_COMMON_ALIASES = {
    "aspirine": "acetylsalicylic acid",
    "aspirin": "acetylsalicylic acid",
    "acide acetylsalicylique": "acetylsalicylic acid",
    "warfarine": "warfarin",
    "paracetamol": "acetaminophen",
    "paracétamol": "acetaminophen",
    "doliprane": "acetaminophen",
    "tylenol": "acetaminophen",
}


# ──────────────────────────────────────────────────────────────────────────
# Connaissance métier additionnelle : libellés et effets secondaires typiques
# par type clinique. Construite manuellement à partir des 11 classes du
# modèle multiclasse (le texte source DrugBank n'est pas inclus dans le
# bundle de déploiement) — niveau prototype académique, à valider par un
# professionnel de santé avant tout usage réel.
# ──────────────────────────────────────────────────────────────────────────

CLINICAL_TYPE_INFO = {
    "Hémorragique": {
        "label": "Risque hémorragique",
        "description": "Effet additif ou synergique sur l'hémostase, augmentant le risque de saignement.",
        "side_effects": [
            "Risque hémorragique accru",
            "Ecchymoses, saignements de nez",
            "Hémorragie digestive (cas sévères)",
            "Allongement du temps de coagulation",
        ],
    },
    "Cardiaque": {
        "label": "Risque cardiaque (rythme)",
        "description": "Effet potentiel sur la conduction cardiaque ou le rythme (allongement du QTc, arythmies).",
        "side_effects": [
            "Allongement de l'intervalle QTc",
            "Risque de troubles du rythme (tachycardie, bradycardie)",
            "Palpitations",
        ],
    },
    "Cardiovasculaire": {
        "label": "Risque cardiovasculaire (tension)",
        "description": "Interaction affectant la pression artérielle ou la fonction cardiovasculaire globale.",
        "side_effects": [
            "Hypotension ou hypertension",
            "Étourdissements posturaux",
            "Modification de la fréquence cardiaque",
        ],
    },
    "Rénal": {
        "label": "Risque rénal",
        "description": "Effet potentiellement néphrotoxique ou altérant la clairance rénale.",
        "side_effects": [
            "Altération de la fonction rénale",
            "Réduction de l'élimination des toxines",
            "Risque accru chez l'insuffisant rénal",
        ],
    },
    "Hépatique": {
        "label": "Risque hépatique",
        "description": "Effet potentiel sur le métabolisme ou la fonction hépatique.",
        "side_effects": [
            "Élévation des enzymes hépatiques",
            "Risque d'hépatotoxicité",
        ],
    },
    "Métabolique (CYP)": {
        "label": "Interaction métabolique (CYP450)",
        "description": "Inhibition ou induction probable d'une enzyme du cytochrome P450, modifiant les concentrations plasmatiques.",
        "side_effects": [
            "Augmentation ou diminution de la concentration plasmatique d'un des deux médicaments",
            "Risque de surdosage relatif ou de sous-dosage",
            "Effets toxiques liés à l'accumulation",
        ],
    },
    "Excrétion": {
        "label": "Interaction sur l'excrétion",
        "description": "Modification probable de la clairance ou de l'élimination d'un des deux médicaments.",
        "side_effects": [
            "Variation du taux sérique du médicament",
            "Risque d'accumulation ou d'élimination accélérée",
        ],
    },
    "Hématologique": {
        "label": "Risque hématologique",
        "description": "Effet potentiel sur les cellules sanguines ou le transport de l'oxygène.",
        "side_effects": [
            "Risque de méthémoglobinémie",
            "Anomalies de la numération sanguine",
        ],
    },
    "Sédation/SNC": {
        "label": "Sédation / dépression du SNC",
        "description": "Effet dépresseur additif sur le système nerveux central.",
        "side_effects": [
            "Somnolence accrue",
            "Risque de dépression respiratoire (cas sévères)",
            "Altération de la vigilance",
        ],
    },
    "Efficacité": {
        "label": "Modification d'efficacité thérapeutique",
        "description": "Risque de réduction ou de potentialisation de l'effet thérapeutique d'un des deux médicaments.",
        "side_effects": [
            "Perte d'efficacité thérapeutique",
            "Effet thérapeutique exagéré",
        ],
    },
    "Pharmacovigilance (TWOSIDES)": {
        "label": "Signal de pharmacovigilance",
        "description": "Association identifiée par signal statistique de pharmacovigilance (base TWOSIDES), sans mécanisme unique établi.",
        "side_effects": [
            "Profil d'effets indésirables non spécifique",
            "Surveillance clinique recommandée",
        ],
    },
    "Autre": {
        "label": "Mécanisme non catégorisé",
        "description": "Interaction détectée sans correspondance claire avec les catégories cliniques principales.",
        "side_effects": [
            "Mécanisme à investiguer au cas par cas",
        ],
    },
}


# ──────────────────────────────────────────────────────────────────────────
# Recherche de médicament par nom
# ──────────────────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def find_drug(name: str) -> dict | None:
    """Cherche un médicament par nom (insensible à la casse). Retourne None si absent."""
    key = normalize_name(name)
    drug_id = NAME_TO_ID.get(key)
    if drug_id is None:
        alias = _COMMON_ALIASES.get(key)
        if alias:
            drug_id = NAME_TO_ID.get(alias)
            key = alias
    if drug_id is None:
        return None
    return {
        "query": name,
        "matched_name": key,
        "drugbank_id": drug_id,
        "smiles": ID_TO_SMILES[drug_id],
    }


def suggest_names(name: str, n: int = 5) -> list[str]:
    """Suggestions de noms proches (utilisées uniquement dans le message d'erreur)."""
    key = normalize_name(name)
    matches = difflib.get_close_matches(key, _ALL_NAMES_SORTED, n=n, cutoff=0.6)
    if not matches:
        # repli : recherche par sous-chaîne
        matches = [n2 for n2 in _ALL_NAMES_SORTED if key in n2][:n]
    return matches


# ──────────────────────────────────────────────────────────────────────────
# Featurisation
# ──────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=4096)
def _get_fp(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES invalide : {smiles}")
    return _MFPGEN.GetFingerprintAsNumPy(mol).astype(np.int16)


def pair_features(smi_a: str, smi_b: str) -> np.ndarray:
    fa, fb = _get_fp(smi_a), _get_fp(smi_b)
    return np.concatenate([fa + fb, np.abs(fa - fb)]).reshape(1, -1)


def bit_to_fragment(smiles: str, bit_idx: int) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    bit_info: dict = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS, bitInfo=bit_info)
    if bit_idx not in bit_info:
        return None
    atom_idx, rad = bit_info[bit_idx][0]
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
    submol = Chem.PathToSubmol(mol, env, atomMap={})
    frag_smiles = Chem.MolToSmiles(submol)
    return frag_smiles if frag_smiles else None


def molecule_svg(smiles: str, width: int = 280, height: int = 220) -> str:
    """Génère un SVG 2D de la molécule (pour affichage dans l'interface)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    from rdkit.Chem.Draw import rdMolDraw2D

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg


# ──────────────────────────────────────────────────────────────────────────
# Risque & confiance
# ──────────────────────────────────────────────────────────────────────────

def risk_level(proba: float) -> str:
    if proba < 0.35:
        return "faible"
    elif proba < 0.70:
        return "modéré"
    return "élevé"


def confidence_score(proba: float) -> float:
    """Score de confiance du modèle : distance à la frontière de décision (0.5),
    normalisée sur [0,1]. Une proba proche de 0 ou 1 -> confiance élevée ;
    une proba proche de 0.5 -> confiance faible (zone d'incertitude)."""
    return float(abs(proba - 0.5) * 2)


# ──────────────────────────────────────────────────────────────────────────
# Pipeline complet : prédiction + explicabilité
# ──────────────────────────────────────────────────────────────────────────

def analyze_pair(name_a: str, name_b: str, top_k_fragments: int = 6) -> dict:
    drug_a = find_drug(name_a)
    if drug_a is None:
        return {"error": "not_found", "which": "A", "query": name_a,
                "suggestions": suggest_names(name_a)}

    drug_b = find_drug(name_b)
    if drug_b is None:
        return {"error": "not_found", "which": "B", "query": name_b,
                "suggestions": suggest_names(name_b)}

    smi_a, smi_b = drug_a["smiles"], drug_b["smiles"]

    mol_a, mol_b = Chem.MolFromSmiles(smi_a), Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return {"error": "invalid_smiles"}

    feat = pair_features(smi_a, smi_b)

    # --- Prédiction binaire ---
    proba = float(BINARY_MODEL.predict_proba(feat)[0, 1])
    level = risk_level(proba)
    confidence = confidence_score(proba)
    interaction_predicted = proba >= DECISION_THRESHOLD

    # --- Prédiction multiclasse (type clinique) ---
    multi_probs = MULTI_MODEL.predict_proba(feat)[0]
    order = np.argsort(multi_probs)[::-1]
    clinical_ranking = [
        {"type": LABEL_ENCODER.classes_[i], "probability": float(multi_probs[i])}
        for i in order
    ]
    top_clinical_type = clinical_ranking[0]["type"]
    clinical_info = CLINICAL_TYPE_INFO.get(top_clinical_type, {
        "label": top_clinical_type, "description": "", "side_effects": []
    })

    # --- SHAP : fragments responsables ---
    shap_values = _SHAP_EXPLAINER.shap_values(feat)[0]
    top_bits = np.argsort(np.abs(shap_values))[::-1][:top_k_fragments]

    fragments = []
    for bit in top_bits:
        is_sum = bit < FP_BITS
        local_bit = int(bit % FP_BITS)
        block = "Somme (A+B)" if is_sum else "Différence absolue |A−B|"
        frag_a = bit_to_fragment(smi_a, local_bit)
        frag_b = bit_to_fragment(smi_b, local_bit)
        frag = frag_a or frag_b or None
        source_drug = "A" if frag_a else ("B" if frag_b else None)
        fragments.append({
            "bit": local_bit,
            "block": block,
            "shap_value": float(shap_values[bit]),
            "direction": "augmente le risque" if shap_values[bit] > 0 else "réduit le risque",
            "fragment_smiles": frag,
            "source_drug": source_drug,
        })

    # --- Raisons lisibles (synthèse pour le pharmacien) ---
    reasons = build_reasons(fragments, clinical_info, proba)

    return {
        "error": None,
        "drug_a": {"name": name_a.strip(), "drugbank_id": drug_a["drugbank_id"], "smiles": smi_a},
        "drug_b": {"name": name_b.strip(), "drugbank_id": drug_b["drugbank_id"], "smiles": smi_b},
        "prediction": {
            "probability": proba,
            "interaction_predicted": bool(interaction_predicted),
            "risk_level": level,
            "confidence": confidence,
            "decision_threshold": DECISION_THRESHOLD,
        },
        "clinical_type": {
            "top": top_clinical_type,
            "label": clinical_info["label"],
            "description": clinical_info["description"],
            "ranking": clinical_ranking[:5],
        },
        "side_effects": clinical_info["side_effects"],
        "fragments": fragments,
        "reasons": reasons,
        "model_metrics": {
            "cold_start_auc": METRICS.get("cold_start", {}).get("auc"),
            "warm_start_auc": METRICS.get("warm_start", {}).get("auc"),
        },
    }


def build_reasons(fragments: list[dict], clinical_info: dict, proba: float) -> list[dict]:
    """Construit 2-4 raisons lisibles à partir des fragments SHAP et du type clinique."""
    reasons = []

    positive_frags = [f for f in fragments if f["shap_value"] > 0 and f["fragment_smiles"]]
    if positive_frags:
        reasons.append({
            "title": "Sous-structures moléculaires partagées",
            "detail": (
                f"{len(positive_frags)} fragment(s) commun(s) ou similaires identifiés par SHAP "
                f"contribuent positivement au score de risque, suggérant une parenté structurelle "
                f"avec des paires d'interaction connues."
            ),
        })

    if clinical_info.get("description"):
        reasons.append({
            "title": clinical_info["label"],
            "detail": clinical_info["description"],
        })

    if proba >= 0.70:
        reasons.append({
            "title": "Score de probabilité élevé",
            "detail": "La probabilité prédite dépasse largement le seuil de décision, indiquant une similarité forte avec des interactions documentées dans les données d'entraînement.",
        })
    elif proba < 0.35:
        reasons.append({
            "title": "Score de probabilité faible",
            "detail": "Le modèle ne détecte pas de similarité significative avec des paires d'interaction connues.",
        })

    return reasons


def list_sample_names(n: int = 12) -> list[str]:
    """Quelques noms réels présents dans la base, pour peupler des exemples dans l'UI."""
    preferred = [
        "acetylsalicylic acid", "warfarin", "simvastatin", "clarithromycin",
        "furosemide", "digoxin", "fluoxetine", "omeprazole",
    ]
    found = [p for p in preferred if p in NAME_TO_ID]
    return found[:n]
