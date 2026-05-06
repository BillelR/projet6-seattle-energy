"""
test_api.py — VERSION CORRIGÉE
───────────────────────────────
Corrections :
  - ENERGYSTARScore retiré de tous les payloads (data leakage)
  - /health appelé en GET (et non POST)
  - Port harmonisé à 8000
  - Commande de lancement corrigée (FastAPI)

Lancer l'API d'abord :
    python3 -m uvicorn service:app --reload --port 8000

Puis dans un autre terminal :
    python3 test_api.py
"""

import requests

BASE_URL = "http://localhost:8000"


def test_predict(label: str, payload: dict) -> None:
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
        # ENERGYSTARScore absent — data leakage
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

# ── Test 4 : Même bâtiment 1990 vs 1995 (vérif cohérence) ────────
print(f"\n{'─'*55}")
print("TEST COHÉRENCE : 1990 vs 1995 (le plus récent doit consommer moins)")
print(f"{'─'*55}")
base = {
    "NumberofBuildings": 1, "NumberofFloors": 5,
    "PropertyGFATotal": 50000.0, "PropertyGFAParking": 5000.0,
    "LargestPropertyUseTypeGFA": 45000.0,
    "SecondLargestPropertyUseTypeGFA": 0.0,
    "ThirdLargestPropertyUseTypeGFA": 0.0,
    "Latitude": 47.61, "Longitude": -122.33,
    "CouncilDistrictCode": 7, "LargestPropertyUseType": "Office",
}
for year in [1990, 1995]:
    payload = {**base, "YearBuilt": year}
    r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
    if r.status_code == 200:
        kBtu = r.json()["prediction_kBtu"]
        print(f"  YearBuilt={year} → {kBtu:,.0f} kBtu/an")

# ── Test 5 : Champ inconnu (doit retourner 422) ───────────────────
print(f"\n{'─'*55}")
print("TEST : Champ inconnu (doit échouer — 422)")
print(f"{'─'*55}")
r = requests.post(f"{BASE_URL}/predict", json={
    "NumberofBuildings": 1, "NumberofFloors": 5,
    "YearBuilt": 2000, "PropertyGFATotal": 50000.0,
    "PropertyGFAParking": 5000.0, "LargestPropertyUseTypeGFA": 45000.0,
    "Latitude": 47.61, "Longitude": -122.33,
    "CouncilDistrictCode": 7, "LargestPropertyUseType": "Office",
    "champ_inconnu": "interdit",
}, timeout=10)
print(f"  Statut attendu : 422 | Reçu : {r.status_code}")

# ── Test 6 : Santé de l'API (GET) ────────────────────────────────
print(f"\n{'─'*55}")
print("TEST : GET /health")
print(f"{'─'*55}")
r = requests.get(f"{BASE_URL}/health", timeout=5)  # GET et non POST
print(f"  Statut : {r.status_code}")
print(f"  Réponse : {r.json()}")

print(f"\n{'═'*55}")
print("Tous les tests terminés.")
print(f"{'═'*55}\n")
