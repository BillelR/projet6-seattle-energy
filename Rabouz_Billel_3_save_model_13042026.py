"""
save_model.py
─────────────
Étape 1 : Entraîne le pipeline final et le sauvegarde dans le registre BentoML.
À exécuter UNE SEULE FOIS depuis le terminal, dans le même dossier que le CSV.

    python save_model.py
"""

import pandas as pd
import numpy as np
import bentoml

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# ── 1. Chargement & nettoyage (identique au notebook) ─────────────────────────
df = pd.read_csv("2016_Building_Energy_Benchmarking.csv")

# Filtrage NonResidential uniquement
df = df[df["BuildingType"] == "NonResidential"].copy()

# Suppression colonnes inutiles / data leakage
cols_drop = [
    "OSEBuildingID", "DataYear", "TaxParcelIdentificationNumber",
    "PropertyName", "Address", "City", "State", "ZipCode",
    "ComplianceStatus", "DefaultData", "Comments", "Outlier",
    "YearsENERGYSTARCertified", "BuildingType",
    "SiteEUI(kBtu/sf)", "SiteEUIWN(kBtu/sf)",
    "SourceEUI(kBtu/sf)", "SourceEUIWN(kBtu/sf)",
    "SiteEnergyUse(kBtu)",
    "SteamUse(kBtu)", "Electricity(kWh)", "Electricity(kBtu)",
    "NaturalGas(therms)", "NaturalGas(kBtu)",
    "GHGEmissionsIntensity",
]
df = df.drop(columns=[c for c in cols_drop if c in df.columns])

# Nettoyage valeurs manquantes
df = df.dropna(subset=["SiteEnergyUseWN(kBtu)",
                        "YearBuilt", "PropertyGFATotal",
                        "NumberofFloors", "LargestPropertyUseType"])
df = df[df["PropertyGFATotal"] > 0]
df = df[df["SiteEnergyUseWN(kBtu)"] > 0]

df["SecondLargestPropertyUseTypeGFA"] = df["SecondLargestPropertyUseTypeGFA"].fillna(0)
df["ThirdLargestPropertyUseTypeGFA"]  = df["ThirdLargestPropertyUseTypeGFA"].fillna(0)

# ── 2. Feature Engineering ─────────────────────────────────────────────────────
df["BuildingAge"]   = 2016 - df["YearBuilt"]
df["ParkingRatio"]  = df["PropertyGFAParking"] / (df["PropertyGFATotal"] + 1)
df["GFAPerFloor"]   = df["PropertyGFATotal"]   / (df["NumberofFloors"] + 1)
df["MainUseRatio"]  = df["LargestPropertyUseTypeGFA"] / (df["PropertyGFATotal"] + 1)
df["HasSecondUse"]  = (df["SecondLargestPropertyUseTypeGFA"] > 0).astype(int)
df["HasThirdUse"]   = (df["ThirdLargestPropertyUseTypeGFA"]  > 0).astype(int)

bins   = [0, 1930, 1960, 1980, 2000, 2010, 2017]
labels = ["avant_1930","1930-1960","1960-1980","1980-2000","2000-2010","apres_2010"]
df["EraConstruction"] = pd.cut(df["YearBuilt"], bins=bins, labels=labels, right=False)

# ── 3. Préparation X / y ───────────────────────────────────────────────────────
FEATURES_NUM = [
    "NumberofBuildings", "NumberofFloors",
    "BuildingAge", "ParkingRatio", "GFAPerFloor",
    "MainUseRatio", "HasSecondUse", "HasThirdUse",
    "Latitude", "Longitude", "CouncilDistrictCode",
]
FEATURES_CAT = ["LargestPropertyUseType", "EraConstruction"]
TARGET = "SiteEnergyUseWN(kBtu)"

X = df[FEATURES_NUM + FEATURES_CAT].copy()
y = np.log1p(df[TARGET])

# Suppression outliers IQR 5%-95%
Q1, Q3 = y.quantile(0.05), y.quantile(0.95)
mask = (y >= Q1 - 1.5*(Q3-Q1)) & (y <= Q3 + 1.5*(Q3-Q1))
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 4. Pipeline sklearn ────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                          max_categories=12), FEATURES_CAT),
])

pipeline = Pipeline([
    ("pre",   preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )),
])

pipeline.fit(X_train, y_train)

from sklearn.metrics import r2_score
print(f"R² Train : {r2_score(y_train, pipeline.predict(X_train)):.4f}")
print(f"R² Test  : {r2_score(y_test,  pipeline.predict(X_test)):.4f}")

# ── 5. Sauvegarde dans BentoML ─────────────────────────────────────────────────
# custom_objects permet de récupérer les noms de features au moment de l'inférence
saved_model = bentoml.sklearn.save_model(
    "seattle_energy_model",
    pipeline,
    custom_objects={
        "features_num": FEATURES_NUM,
        "features_cat": FEATURES_CAT,
    },
    signatures={"predict": {"batchable": False}},
)

print(f"\nModèle sauvegardé : {saved_model.tag}")
print("Pour vérifier : bentoml models list")
