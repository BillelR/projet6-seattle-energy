"""
save_model_pkl.py — VERSION FINALE CORRIGÉE
────────────────────────────────────────────
Corrections appliquées :
  1. Filtre Outlier : garder uniquement NaN (bâtiments normaux)
  2. Filtre ComplianceStatus : garder uniquement 'Compliant'
  3. Target : SiteEnergyUseWN(kBtu) — normalisée météo (correction tuteur)

Lancer UNE SEULE FOIS dans le dossier du projet :
    python3 save_model_pkl.py
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ── 1. Chargement ─────────────────────────────────────────────────
df = pd.read_csv("2016_Building_Energy_Benchmarking.csv")

# ── 2. Filtrage NonResidential ────────────────────────────────────
df = df[df["BuildingType"] == "NonResidential"].copy()
print(f"Après NonResidential       : {len(df)} bâtiments")

# ── 3. Filtre Outlier (NaN = bâtiment normal → garder) ───────────
df = df[df["Outlier"].isna()].copy()
print(f"Après filtre Outlier       : {len(df)} bâtiments")

# ── 4. Filtre ComplianceStatus (Compliant uniquement) ────────────
df = df[df["ComplianceStatus"] == "Compliant"].copy()
print(f"Après filtre Compliant     : {len(df)} bâtiments")

# ── 5. Suppression colonnes inutiles / data leakage ──────────────
cols_drop = [
    "OSEBuildingID", "DataYear", "TaxParcelIdentificationNumber",
    "PropertyName", "Address", "City", "State", "ZipCode",
    "ComplianceStatus", "DefaultData", "BuildingType",
    "Comments", "Outlier", "YearsENERGYSTARCertified",
    "SiteEnergyUse(kBtu)",           # remplacée par la version WN
    "SiteEUI(kBtu/sf)", "SiteEUIWN(kBtu/sf)",
    "SourceEUI(kBtu/sf)", "SourceEUIWN(kBtu/sf)",
    "SteamUse(kBtu)", "Electricity(kWh)", "Electricity(kBtu)",
    "NaturalGas(therms)", "NaturalGas(kBtu)",
    "GHGEmissionsIntensity",
]

# Sauvegarder la target AVANT suppression
target_series = df["SiteEnergyUseWN(kBtu)"].copy()
df = df.drop(columns=[c for c in cols_drop if c in df.columns])

# ── 6. Nettoyage valeurs manquantes ──────────────────────────────
df["ENERGYSTARScore"] = df["ENERGYSTARScore"].fillna(df["ENERGYSTARScore"].median())
df["SecondLargestPropertyUseTypeGFA"] = df["SecondLargestPropertyUseTypeGFA"].fillna(0)
df["ThirdLargestPropertyUseTypeGFA"]  = df["ThirdLargestPropertyUseTypeGFA"].fillna(0)
df = df.dropna(subset=["YearBuilt", "PropertyGFATotal",
                        "NumberofFloors", "LargestPropertyUseType"])
df = df[df["PropertyGFATotal"] > 0]

# ── 7. Feature Engineering ────────────────────────────────────────
df["BuildingAge"]  = 2016 - df["YearBuilt"]
df["ParkingRatio"] = df["PropertyGFAParking"] / (df["PropertyGFATotal"] + 1)
df["GFAPerFloor"]  = df["PropertyGFATotal"]   / (df["NumberofFloors"] + 1)
df["MainUseRatio"] = df["LargestPropertyUseTypeGFA"] / (df["PropertyGFATotal"] + 1)
df["HasSecondUse"] = (df["SecondLargestPropertyUseTypeGFA"] > 0).astype(int)
df["HasThirdUse"]  = (df["ThirdLargestPropertyUseTypeGFA"]  > 0).astype(int)

bins   = [0, 1930, 1960, 1980, 2000, 2010, 2017]
labels = ["avant_1930","1930-1960","1960-1980","1980-2000","2000-2010","apres_2010"]
df["EraConstruction"] = pd.cut(df["YearBuilt"], bins=bins, labels=labels, right=False)

# ── 8. DataFrame de modélisation ─────────────────────────────────
FEATURES_NUM = [
    "NumberofBuildings", "NumberofFloors", "ENERGYSTARScore",
    "BuildingAge", "ParkingRatio", "GFAPerFloor",
    "MainUseRatio", "HasSecondUse", "HasThirdUse",
    "Latitude", "Longitude", "CouncilDistrictCode",
]
FEATURES_CAT = ["LargestPropertyUseType", "EraConstruction"]
TARGET = "SiteEnergy UseWN(kBtu)"   # ← target corrigée (Weather Normalized)

df[TARGET] = target_series[df.index]
df_model = df[FEATURES_NUM + FEATURES_CAT + [TARGET]].copy()
df_model = df_model.dropna(subset=[TARGET])
df_model = df_model[df_model[TARGET] > 0]

X = df_model[FEATURES_NUM + FEATURES_CAT].copy()
y = np.log1p(df_model[TARGET])

# Suppression outliers IQR 5%-95%
Q1, Q3 = y.quantile(0.05), y.quantile(0.95)
mask = (y >= Q1 - 1.5*(Q3-Q1)) & (y <= Q3 + 1.5*(Q3-Q1))
X, y = X[mask], y[mask]
print(f"Bâtiments pour modélisation : {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 9. Pipeline sklearn ───────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                          max_categories=12), FEATURES_CAT),
])

pipeline = Pipeline([
    ("pre",   preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )),
])

pipeline.fit(X_train, y_train)
print(f"R² Train : {r2_score(y_train, pipeline.predict(X_train)):.4f}")
print(f"R² Test  : {r2_score(y_test,  pipeline.predict(X_test)):.4f}")

# ── 10. Sauvegarde pickle ─────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nModèle sauvegardé : model.pkl")
print("Lance maintenant : python3 -m uvicorn service:app --reload --port 8000")
