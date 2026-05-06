"""
service.py — FastAPI (VERSION CORRIGÉE sans data leakage)
──────────────────────────────────────────────────────────
Correction : ENERGYSTARScore retiré du schéma d'entrée
  → Ce score nécessite des données de consommation réelle pour être calculé.
    Un propriétaire ne peut pas le fournir avant la mesure de consommation.
    L'inclure constitue du data leakage.

Lancer : python3 -m uvicorn service:app --reload --port 8000
Swagger : http://localhost:8000/docs
"""

from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

# ══════════════════════════════════════════════════════════════════
# SCHÉMA D'ENTRÉE — validation Pydantic (sans ENERGYSTARScore)
# ══════════════════════════════════════════════════════════════════

class BuildingInput(BaseModel):
    NumberofBuildings: int = Field(ge=1, le=100, examples=[1],
        description="Nombre de bâtiments sur la parcelle (1–100)")
    NumberofFloors: int = Field(ge=1, le=150, examples=[5],
        description="Nombre d'étages (1–150)")
    # ENERGYSTARScore supprimé — data leakage
    # Ce score est calculé à partir des données de consommation réelle
    # que l'on cherche justement à prédire.
    YearBuilt: int = Field(ge=1850, le=2016, examples=[1985],
        description="Année de construction (1850–2016)")
    PropertyGFATotal: float = Field(gt=0, le=5_000_000, examples=[50000.0],
        description="Surface totale brute en pieds carrés")
    PropertyGFAParking: float = Field(ge=0, le=2_000_000, examples=[5000.0],
        description="Surface de parking en pieds carrés (0 si aucun)")
    LargestPropertyUseTypeGFA: float = Field(gt=0, le=5_000_000, examples=[45000.0],
        description="Surface de l'usage principal en pieds carrés")
    SecondLargestPropertyUseTypeGFA: float = Field(default=0.0, ge=0, examples=[0.0],
        description="Surface du 2e usage (0 si mono-usage)")
    ThirdLargestPropertyUseTypeGFA: float = Field(default=0.0, ge=0, examples=[0.0],
        description="Surface du 3e usage (0 si mono ou bi-usage)")
    Latitude: float = Field(ge=47.4, le=47.8, examples=[47.61],
        description="Latitude GPS (Seattle : 47.4–47.8)")
    Longitude: float = Field(ge=-122.6, le=-122.1, examples=[-122.33],
        description="Longitude GPS (Seattle : -122.6 à -122.1)")
    CouncilDistrictCode: int = Field(ge=1, le=7, examples=[7],
        description="Arrondissement du conseil municipal (1–7)")
    LargestPropertyUseType: str = Field(examples=["Office"],
        description="Type d'usage principal (ex: Office, Hotel, Retail Store)")

    @field_validator("LargestPropertyUseType")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("LargestPropertyUseType ne peut pas être vide.")
        return v.strip()

    @model_validator(mode="after")
    def check_coherence(self) -> "BuildingInput":
        if self.PropertyGFAParking > self.PropertyGFATotal:
            raise ValueError("PropertyGFAParking ne peut pas dépasser PropertyGFATotal.")
        if self.LargestPropertyUseTypeGFA > self.PropertyGFATotal:
            raise ValueError("LargestPropertyUseTypeGFA ne peut pas dépasser PropertyGFATotal.")
        return self

    class Config:
        extra = "forbid"


# ══════════════════════════════════════════════════════════════════
# SCHÉMA DE SORTIE
# ══════════════════════════════════════════════════════════════════

class PredictionOutput(BaseModel):
    prediction_kBtu: float = Field(description="Consommation prédite en kBtu/an")
    prediction_MWh: float  = Field(description="Consommation prédite en MWh/an")
    model_version: str     = Field(description="Version du modèle")


# ══════════════════════════════════════════════════════════════════
# FEATURES — identique à save_model_pkl.py (sans ENERGYSTARScore)
# ══════════════════════════════════════════════════════════════════

FEATURES_NUM = [
    "NumberofBuildings", "NumberofFloors",
    "BuildingAge", "ParkingRatio", "GFAPerFloor",
    "MainUseRatio", "HasSecondUse", "HasThirdUse",
    "Latitude", "Longitude", "CouncilDistrictCode",
]
FEATURES_CAT = ["LargestPropertyUseType", "EraConstruction"]


# ══════════════════════════════════════════════════════════════════
# CHARGEMENT DU MODÈLE
# ══════════════════════════════════════════════════════════════════

try:
    with open("model.pkl", "rb") as f:
        pipeline = pickle.load(f)
    print("Modèle chargé depuis model.pkl")
except FileNotFoundError:
    pipeline = None
    print("ATTENTION : model.pkl introuvable — lancez d'abord save_model_pkl.py")


# ══════════════════════════════════════════════════════════════════
# APPLICATION FASTAPI
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Seattle Energy Predictor",
    description=(
        "API de prédiction de la consommation énergétique annuelle "
        "des bâtiments non résidentiels de Seattle.\n\n"
        "**Modèle** : Random Forest sans data leakage "
        "(ENERGYSTARScore exclu)\n\n"
        "**Target** : SiteEnergyUseWN(kBtu) — normalisée météo\n\n"
        "**Unités** : kBtu/an et MWh/an"
    ),
    version="2.0.0",
)


def feature_engineering(b: BuildingInput) -> pd.DataFrame:
    """Applique le même feature engineering que dans le notebook."""
    year = b.YearBuilt
    row = {
        "NumberofBuildings":      b.NumberofBuildings,
        "NumberofFloors":         b.NumberofFloors,
        # ENERGYSTARScore absent
        "BuildingAge":            2016 - year,
        "ParkingRatio":           b.PropertyGFAParking / (b.PropertyGFATotal + 1),
        "GFAPerFloor":            b.PropertyGFATotal   / (b.NumberofFloors + 1),
        "MainUseRatio":           b.LargestPropertyUseTypeGFA / (b.PropertyGFATotal + 1),
        "HasSecondUse":           int(b.SecondLargestPropertyUseTypeGFA > 0),
        "HasThirdUse":            int(b.ThirdLargestPropertyUseTypeGFA  > 0),
        "Latitude":               b.Latitude,
        "Longitude":              b.Longitude,
        "CouncilDistrictCode":    b.CouncilDistrictCode,
        "LargestPropertyUseType": b.LargestPropertyUseType,
        "EraConstruction": (
            "avant_1930" if year < 1930 else
            "1930-1960"  if year < 1960 else
            "1960-1980"  if year < 1980 else
            "1980-2000"  if year < 2000 else
            "2000-2010"  if year < 2010 else
            "apres_2010"
        ),
    }
    return pd.DataFrame([row], columns=FEATURES_NUM + FEATURES_CAT)


@app.get("/health")
def health():
    """Vérifie que l'API est en ligne."""
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(building: BuildingInput):
    """
    Prédit la consommation énergétique annuelle d'un bâtiment.

    Variables structurelles uniquement — pas de data leakage :
    - Taille et structure du bâtiment (GFA, étages, parking)
    - Âge et époque de construction
    - Type d'usage principal
    - Localisation géographique
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Lancez d'abord : python3 save_model_pkl.py"
        )
    input_df = feature_engineering(building)
    log_pred = pipeline.predict(input_df)
    kBtu = float(np.expm1(log_pred[0]))
    return PredictionOutput(
        prediction_kBtu=round(kBtu, 2),
        prediction_MWh=round(kBtu / 3412.14, 2),
        model_version="random_forest_v2_sans_energystar",
    )
