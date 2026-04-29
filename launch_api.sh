#!/bin/bash

# ═══════════════════════════════════════════════════════════════
#  launch_api.sh — Lancement automatique de l'API Seattle Energy
# ═══════════════════════════════════════════════════════════════

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Seattle Energy Predictor — API       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── Dossier du projet ────────────────────────────────────────────
PROJECT_DIR="$HOME/Documents/P6/Projet 6"

if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}❌ Dossier projet introuvable : $PROJECT_DIR${NC}"
    exit 1
fi

cd "$PROJECT_DIR"
echo -e "${GREEN}✓ Dossier projet : $PROJECT_DIR${NC}"

# ── Détection automatique du venv ────────────────────────────────
# Cherche .venv en premier, puis venv_projet6
if [ -d "$PROJECT_DIR/.venv" ]; then
    VENV_DIR="$PROJECT_DIR/.venv"
elif [ -d "$PROJECT_DIR/venv_projet6" ]; then
    VENV_DIR="$PROJECT_DIR/venv_projet6"
else
    echo -e "${RED}❌ Aucun environnement virtuel trouvé (.venv ou venv_projet6)${NC}"
    exit 1
fi

source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Impossible d'activer le venv : $VENV_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Environnement virtuel activé : $VENV_DIR${NC}"

# ── Vérification du modèle ───────────────────────────────────────
if [ ! -f "model.pkl" ]; then
    echo -e "${YELLOW}⚠ model.pkl introuvable — génération en cours...${NC}"
    python3 save_model_pkl.py
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

# On utilise python3 -m uvicorn pour être sûr d'utiliser
# le uvicorn du venv et non celui du système Ubuntu
python3 -m uvicorn service:app --reload --port 8000
