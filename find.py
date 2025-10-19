
import requests
import math
import csv
from datetime import datetime

# --- Predefined location and search radius ---``
CENTER_LAT = 22.5592613
CENTER_LON = 88.3861048
RADIUS_M = 2000   # default 2 km


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


CATEGORIES = {
    "1": ("amenity", "hospital"),
    "2": ("amenity", "pharmacy"),
    "3": ("amenity", "restaurant"),
    "4": ("shop", "supermarket"),
    "5": ("shop", "clothes"),
}

def haversine_m(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two lat/lon points."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def build_overpass_query(lat, lon, radius_m, key, value):
    return f"""
    [out:json][timeout:25];
    (
      node["{key}"="{value}"](around:{radius_m},{lat},{lon});
      way["{key}"="{value}"](around:{radius_m},{lat},{lon});
      relation["{key}"="{value}"](around:{radius_m},{lat},{lon});
    );
    out center tags;
    """

def run_overpass(query):
    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=60)
    resp.raise_for_status()
    return resp.json()

def extract_places(osm_json, center_lat, center_lon, key):
    places = []
    for el in osm_json.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name") or tags.get(key) or "UNKNOWN"
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            c = el.get("center")
            if not c:
                continue
            lat, lon = c["lat"], c["lon"]
        dist = haversine_m(center_lat, center_lon, lat, lon)
        places.append({
            "osm_id": f"{el['type']}/{el['id']}",
            "name": name,
            "tag_value": tags.get(key, ""),
            "lat": lat,
            "lon": lon,
            "distance_m": round(dist, 1)
        })
    return sorted(places, key=lambda x: x["distance_m"])

def save_csv(places, filename):
    fields = ["osm_id","name","tag_value","lat","lon","distance_m"]
    with open(filename,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(places)

def main():
    # --- Choose category ---
    print("Select what you want to search near the point:")
    for k, v in CATEGORIES.items():
        print(f"{k}. {v[1].capitalize()}")
    choice = input("Enter choice number: ").strip()

    if choice not in CATEGORIES:
        print("Invalid choice. Exiting.")
        return

    key, value = CATEGORIES[choice]

    print(f"\nSearching for {value} around ({CENTER_LAT}, {CENTER_LON}), radius={RADIUS_M}m ...")

    query = build_overpass_query(CENTER_LAT, CENTER_LON, RADIUS_M, key, value)
    try:
        data = run_overpass(query)
    except Exception as e:
        print("Error:", e)
        return

    places = extract_places(data, CENTER_LAT, CENTER_LON, key)
    print(f"Found {len(places)} {value}(s).\n")

    for i, s in enumerate(places[:50], 1):
        print(f"{i:2d}. {s['name']} | {s['tag_value'] or '-'} | {s['distance_m']} m | ({s['lat']:.6f},{s['lon']:.6f})")

    out = f"{value}_{RADIUS_M}m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_csv(places, out)
    print(f"\nSaved full list to {out}")

if __name__=="__main__":
    main()
