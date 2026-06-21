#!/usr/bin/env python3
"""
ClearDDI — lanceur de démo
Installe les dépendances si besoin, démarre le serveur, affiche un QR code
pour que n'importe qui (sur le même réseau, ou partout avec --public) puisse
ouvrir l'app sur son téléphone, et ouvre le navigateur local.

Usage :
  python3 run.py              → accès réseau local (Wi-Fi de la salle)
  python3 run.py --public     → accès public via un tunnel (n'importe où)
  python3 run.py --port 9000  → changer le port
  python3 run.py --no-browser → ne pas ouvrir le navigateur automatiquement
"""

import argparse
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
QR_PATH = ROOT / "qr_acces.png"

REQUIRED_PACKAGES = [
    "fastapi", "uvicorn", "rdkit", "shap", "xgboost",
    "scikit-learn", "joblib", "pandas", "numpy", "pyarrow", "pydantic",
    "qrcode", "pillow",
]


def ensure_dependencies(need_ngrok: bool):
    print("Vérification des dépendances Python...")
    packages = list(REQUIRED_PACKAGES) + (["pyngrok"] if need_ngrok else [])
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *packages],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        # Systèmes Debian/Ubuntu récents (PEP 668) refusent pip install hors venv.
        # On retente avec --break-system-packages plutôt que de planter au démarrage,
        # sans alarmer l'utilisateur avec le message pip brut si ça se résout seul.
        retry = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages", *packages],
            capture_output=True, text=True, check=False,
        )
        if retry.returncode != 0:
            print(
                "⚠️  Installation automatique des dépendances impossible.\n"
                "   Essayez manuellement :\n"
                f"     {sys.executable} -m pip install --break-system-packages {' '.join(packages)}\n"
                "   ou créez un environnement virtuel (python3 -m venv .venv) avant de relancer."
            )


def get_lan_ip():
    """Adresse IP de cette machine sur le réseau local (celle que les autres
    appareils du même Wi-Fi doivent utiliser pour nous joindre). N'envoie
    aucun paquet : ouvre juste une socket UDP pour lire l'IP de sortie."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return None


def wait_for_server(port, timeout=25.0):
    """Attend que /api/health réponde, au lieu d'un sleep() arbitraire."""
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/api/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def start_public_tunnel(port):
    """Ouvre un tunnel public (ngrok) vers le serveur local. Nécessite un
    compte ngrok gratuit + un authtoken configuré (voir README) — sans quoi
    ngrok refuse désormais les tunnels anonymes."""
    try:
        from pyngrok import ngrok
    except ImportError:
        print("⚠️  pyngrok n'est pas installé — accès public désactivé.")
        return None

    try:
        tunnel = ngrok.connect(port, "http")
        return tunnel.public_url
    except Exception as e:
        print(
            "⚠️  Impossible d'ouvrir un tunnel public via ngrok.\n"
            "   Cause probable : pas d'authtoken configuré.\n"
            "   → Créez un compte gratuit sur https://dashboard.ngrok.com/signup\n"
            "   → Récupérez votre token puis lancez une fois :\n"
            "       python3 -m pyngrok config add-authtoken VOTRE_TOKEN\n"
            f"   Détail technique : {e}\n"
            "   → On continue avec l'accès réseau local à la place."
        )
        return None


def show_qr(url):
    try:
        import qrcode
    except ImportError:
        print("⚠️  Le paquet 'qrcode' n'est pas installé : QR code non généré.")
        return

    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make()

    print("\nScannez ce QR code pour ouvrir ClearDDI :\n")
    qr.print_ascii(invert=True)

    try:
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(QR_PATH)
        print(f"\n(QR code aussi enregistré dans : {QR_PATH})")
    except Exception:
        pass  # l'ASCII ci-dessus suffit si l'image ne peut pas être sauvegardée


def main():
    parser = argparse.ArgumentParser(description="Lanceur de démo ClearDDI")
    parser.add_argument("--port", type=int, default=8765, help="Port du serveur (défaut : 8765)")
    parser.add_argument("--public", action="store_true",
                         help="Rendre l'app accessible publiquement (tunnel ngrok), pas seulement sur le réseau local")
    parser.add_argument("--no-browser", action="store_true",
                         help="Ne pas ouvrir automatiquement le navigateur local")
    args = parser.parse_args()
    port = args.port

    ensure_dependencies(need_ngrok=args.public)

    print(f"Démarrage du serveur ClearDDI sur le port {port} ...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(port)],
        cwd=str(BACKEND_DIR),
    )

    if not wait_for_server(port):
        print("⚠️  Le serveur ne répond pas encore après 25s — vérifiez les erreurs ci-dessus.")

    local_url = f"http://localhost:{port}"
    share_url = local_url
    access_note = "réseau local uniquement"

    if args.public:
        public_url = start_public_tunnel(port)
        if public_url:
            share_url = public_url
            access_note = "accessible depuis n'importe où sur Internet"

    if share_url == local_url:
        lan_ip = get_lan_ip()
        if lan_ip:
            share_url = f"http://{lan_ip}:{port}"
            access_note = "accessible à toute personne sur le même réseau Wi-Fi"
        else:
            access_note = "IP locale non détectée — partagez plutôt l'URL locale manuellement"

    print(f"\n✅ Serveur prêt : {local_url}  (sur cette machine)")
    print(f"📱 À partager ({access_note}) : {share_url}")
    show_qr(share_url)

    if not args.no_browser:
        webbrowser.open(local_url)

    print("\nServeur lancé. Appuyez sur Ctrl+C pour arrêter.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
