"""
test_api.py
───────────
Script de test de l'API FastAPI en local.
L'API doit être démarrée avant de lancer ce script :

    python3 -m uvicorn service:app --reload --port 8000

Puis dans un autre terminal :

    python3 test_api.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_predict(label: str, payload: dict) -> None:
    """Envoie une requête et affiche le résultat."""
    print(f"\n{'─'*55}")
    print(f"TEST : {label}")
    print(f"{'─'*55}")
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        if r.status_code == 200:
            result = r.json()
            print(f"  Statut         : {r.status_code} OK")
            print(f"  Consommation   : {result['prediction_kBtu']:,.0f} kBtu/an")
            print(f"  Equivalent     : {result['prediction_MWh']:,.1f} MWh/an")
            print(f"  Modèle utilisé : {result['model_version']}")
        else:
            print(f"  Statut : {r.status_code}")
            print(f"  Erreur : {r.text[:300]}")
    except requests.exceptions.ConnectionError:
        print("  ERREUR : l'API n'est pas démarrée.")
        print("  Lancez : python3 -m uvicorn service:app --reload --port 8000")


# ── Test 1 : Grand immeuble de bureaux ────────────────────────────
test_predict(
    "Grand immeuble de bureaux (Office, 12 étages, 1995)",
    {
        "NumberofBuildings": 1,
        "NumberofFloors": 12,
        "ENERGYSTARScore": 72.0,
        "YearBuilt": 1995,
        "PropertyGFATotal": 150000.0,
        "PropertyGFAParking": 20000.0,
        "LargestPropertyUseTypeGFA": 130000.0,
        "SecondLargestPropertyUseTypeGFA": 0.0,
        "ThirdLargestPropertyUseTypeGFA": 0.0,
        "Latitude": 47.6062,
        "Longitude": -122.3321,
        "CouncilDistrictCode": 7,
        "LargestPropertyUseType": "Office",
    },
)

# ── Test 2 : Petit bâtiment commercial ancien ─────────────────────
test_predict(
    "Petit bâtiment commercial ancien (Retail, 2 étages, 1935)",
    {
        "NumberofBuildings": 1,
        "NumberofFloors": 2,
        "ENERGYSTARScore": 45.0,
        "YearBuilt": 1935,
        "PropertyGFATotal": 8000.0,
        "PropertyGFAParking": 0.0,
        "LargestPropertyUseTypeGFA": 8000.0,
        "SecondLargestPropertyUseTypeGFA": 0.0,
        "ThirdLargestPropertyUseTypeGFA": 0.0,
        "Latitude": 47.5900,
        "Longitude": -122.3200,
        "CouncilDistrictCode": 2,
        "LargestPropertyUseType": "Retail Store",
    },
)

# ── Test 3 : Hôtel ────────────────────────────────────────────────
test_predict(
    "Hôtel (Hotel, 15 étages, 2003)",
    {
        "NumberofBuildings": 1,
        "NumberofFloors": 15,
        "ENERGYSTARScore": 55.0,
        "YearBuilt": 2003,
        "PropertyGFATotal": 200000.0,
        "PropertyGFAParking": 30000.0,
        "LargestPropertyUseTypeGFA": 170000.0,
        "SecondLargestPropertyUseTypeGFA": 0.0,
        "ThirdLargestPropertyUseTypeGFA": 0.0,
        "Latitude": 47.6100,
        "Longitude": -122.3350,
        "CouncilDistrictCode": 7,
        "LargestPropertyUseType": "Hotel",
    },
)

# ── Test 4 : Champ inconnu (doit retourner erreur 422) ────────────
print(f"\n{'─'*55}")
print("TEST : Champ inconnu (doit échouer — 422)")
print(f"{'─'*55}")
r = requests.post(f"{BASE_URL}/predict", json={
    "NumberofBuildings": 1,
    "NumberofFloors": 5,
    "ENERGYSTARScore": 60.0,
    "YearBuilt": 2000,
    "PropertyGFATotal": 50000.0,
    "PropertyGFAParking": 5000.0,
    "LargestPropertyUseTypeGFA": 45000.0,
    "Latitude": 47.61,
    "Longitude": -122.33,
    "CouncilDistrictCode": 7,
    "LargestPropertyUseType": "Office",
    "champ_inconnu": "valeur_interdite",
}, timeout=10)
print(f"  Statut attendu : 422 | Statut reçu : {r.status_code}")
if r.status_code == 422:
    print("  Validation OK — champ inconnu bien rejeté")

# ── Test 5 : Parking > surface totale (doit retourner erreur 422) ──
print(f"\n{'─'*55}")
print("TEST : Parking > Surface totale (doit échouer — 422)")
print(f"{'─'*55}")
r = requests.post(f"{BASE_URL}/predict", json={
    "NumberofBuildings": 1,
    "NumberofFloors": 3,
    "ENERGYSTARScore": 60.0,
    "YearBuilt": 2000,
    "PropertyGFATotal": 10000.0,
    "PropertyGFAParking": 99999.0,
    "LargestPropertyUseTypeGFA": 9000.0,
    "Latitude": 47.61,
    "Longitude": -122.33,
    "CouncilDistrictCode": 7,
    "LargestPropertyUseType": "Office",
}, timeout=10)
print(f"  Statut attendu : 422 | Statut reçu : {r.status_code}")
if r.status_code == 422:
    print("  Validation OK — incohérence bien rejetée")

# ── Test 6 : Santé de l'API (GET) ────────────────────────────────
print(f"\n{'─'*55}")
print("TEST : Endpoint GET /health")
print(f"{'─'*55}")
r = requests.get(f"{BASE_URL}/health", timeout=5)  # GET et non POST
print(f"  Statut : {r.status_code}")
print(f"  Réponse : {r.json()}")

print(f"\n{'═'*55}")
print("Tous les tests terminés.")
print(f"{'═'*55}\n")
