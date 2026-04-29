#!/bin/bash

# ═══════════════════════════════════════════════════════════════
#  launch_api.sh — Lancement automatique de l'API Seattle Energy
# ═══════════════════════════════════════════════════════════════

# Couleurs pour l'affichage
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Seattle Energy Predictor — API       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── Dossier du projet ────────────────────────────────────────────
PROJECT_DIR="$HOME/Documents/P6/Projet 6"
VENV_DIR="$PROJECT_DIR/venv_projet6"

# Vérification du dossier
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}❌ Dossier projet introuvable : $PROJECT_DIR${NC}"
    exit 1
fi

cd "$PROJECT_DIR"
echo -e "${GREEN}✓ Dossier projet : $PROJECT_DIR${NC}"

# ── Activation du venv ───────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}❌ Environnement virtuel introuvable : $VENV_DIR${NC}"
    exit 1
fi

source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Environnement virtuel activé${NC}"

# ── Vérification du modèle ───────────────────────────────────────
if [ ! -f "model.pkl" ]; then
    echo -e "${YELLOW}⚠ model.pkl introuvable — génération en cours...${NC}"
    python save_model_pkl.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Échec de la génération du modèle${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Modèle généré avec succès${NC}"
else
    echo -e "${GREEN}✓ Modèle model.pkl trouvé${NC}"
fi

# ── Vérification du service ──────────────────────────────────────
if [ ! -f "service.py" ]; then
    echo -e "${RED}❌ service.py introuvable${NC}"
    exit 1
fi
echo -e "${GREEN}✓ service.py trouvé${NC}"

# ── Lancement de l'API ───────────────────────────────────────────
echo ""
echo -e "${BLUE}🚀 Lancement de l'API sur http://localhost:8000${NC}"
echo -e "${BLUE}📖 Interface Swagger : http://localhost:8000/docs${NC}"
echo -e "${YELLOW}⚠  Pour arrêter : Ctrl + C${NC}"
echo ""

uvicorn service:app --reload --port 8000
