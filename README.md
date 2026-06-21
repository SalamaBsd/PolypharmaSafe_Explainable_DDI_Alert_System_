# ClearDDI — Explainable DDI Alert System

Prototype académique d'aide à la décision pour la détection d'interactions
médicamenteuses (DDI), avec une couche d'explicabilité (SHAP) destinée au
pharmacien.

**Ce n'est pas un dispositif médical validé.** Les prédictions sont générées
par un modèle expérimental à but pédagogique et ne doivent jamais remplacer
le jugement clinique d'un professionnel de santé.

## Ce que fait l'application

1. Vous saisissez le **nom** de deux médicaments (pas besoin de connaître leur SMILES).
2. L'application retrouve la molécule correspondante dans la base DrugBank
   embarquée (8 282 substances) et calcule sa structure.
3. Un modèle **XGBoost** entraîné sur des fingerprints moléculaires (Morgan,
   rayon 2, 2048 bits) prédit la probabilité d'interaction.
4. Un second modèle XGBoost multi-classe prédit le **type clinique** le plus
   probable de l'interaction (hémorragique, cardiaque, métabolique, etc.).
5. **SHAP** (TreeExplainer) identifie les fragments moléculaires qui
   contribuent le plus à la prédiction, avec remontée vers de vraies
   sous-structures chimiques (RDKit).
6. Tout est assemblé dans une fiche synthétique imprimable.

## Lancer la démo

Prérequis : Python 3.10+ installé sur la machine.

```bash
python3 run.py
```

Le script installe les dépendances manquantes, démarre le serveur, **affiche un QR
code dans le terminal** (et l'enregistre dans `qr_acces.png`), et ouvre
automatiquement votre navigateur sur `http://localhost:8765`.

### Faire essayer l'app à d'autres personnes (QR code)

Par défaut, `run.py` détecte l'adresse IP de votre machine sur le réseau local
et génère un QR code pointant vers `http://<votre-ip>:8765`. **Toute personne
connectée au même réseau Wi-Fi** (salle de classe, soutenance, démo) peut
scanner ce QR code avec son téléphone pour ouvrir l'app directement — aucune
installation requise de leur côté.

```bash
python3 run.py
```

Pour un accès **public, depuis n'importe où sur Internet** (pas seulement le
même Wi-Fi), utilisez un tunnel :

```bash
python3 run.py --public
```

Ce mode nécessite un compte ngrok gratuit (le tunnel anonyme sans compte
n'est plus autorisé par ngrok) :
1. Créez un compte sur https://dashboard.ngrok.com/signup
2. Récupérez votre authtoken sur le tableau de bord
3. Lancez une fois : `python3 -m pyngrok config add-authtoken VOTRE_TOKEN`
4. Relancez `python3 run.py --public`

Si le tunnel ne peut pas s'ouvrir (pas de token, pas de réseau...), le script
revient automatiquement à l'accès réseau local plutôt que de planter.

Autres options : `--port 9000` (changer de port), `--no-browser` (ne pas
ouvrir le navigateur local automatiquement).

Si le navigateur ne s'ouvre pas seul, ouvrez ce lien manuellement.

Pour arrêter : `Ctrl+C` dans le terminal.

### Fiche imprimable / export PDF

Le bouton **« Imprimer / exporter »** (dans la section *Fiche synthétique
pharmacien*) ouvre la boîte de dialogue d'impression du navigateur, qui
permet d'enregistrer directement en PDF. Le document généré est une **fiche
mise en page proprement** (en-tête avec les deux médicaments et le niveau de
risque, sections organisées, mention légale en pied de page) — il ne contient
**ni le bouton d'export, ni le formulaire de saisie, ni le menu de
navigation**, contrairement à une simple capture de la page.


## Structure du projet

```
clearddi-app/
├── run.py                  ← à lancer pour la démo
├── qr_acces.png             ← généré au lancement (QR code d'accès, ignorable/à régénérer)
├── artifacts/               ← vos artefacts de modèle (issus du notebook)
│   ├── ddi_binary_model.joblib
│   ├── ddi_multiclass_model.joblib
│   ├── ddi_multiclass_label_encoder.joblib
│   ├── name_to_id.pkl
│   ├── id_to_smiles.pkl
│   ├── inference_config.json
│   └── metrics.json
├── backend/
│   ├── main.py              ← API FastAPI
│   └── inference.py         ← chargement du modèle, SHAP, logique métier
└── frontend/
    ├── index.html
    ├── styles.css
    ├── app.js
    └── logo.png
```

## Limites connues (à mentionner en soutenance)

- La base de noms est celle de **DrugBank** : les noms doivent correspondre
  à la nomenclature DrugBank (ex. *acetylsalicylic acid*, pas systématiquement
  *aspirine* — quelques alias français courants sont gérés, mais pas tous).
- Le **type clinique documenté** (texte source DrugBank/TWOSIDES) n'est pas
  inclus dans le bundle de déploiement : le type clinique affiché est une
  **prédiction** du second modèle, pas une donnée vérifiée pour toutes les paires.
- Les **effets secondaires potentiels** affichés sont un mapping générique
  construit par catégorie clinique (pas une extraction patient-spécifique).
- L'extension **GNN** explorée dans le notebook n'est pas déployée dans cette
  démo (modèle non sauvegardé dans le bundle) ; l'explicabilité utilise donc
  uniquement SHAP sur le modèle XGBoost.

## Métriques du modèle (issues de votre notebook)

| Protocole | Accuracy | F1 | AUC |
|---|---|---|---|
| Warm-start | 0.892 | 0.895 | 0.958 |
| Cold-start | 0.931 | 0.932 | 0.981 |
