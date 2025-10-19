import requests
import math
import csv
from datetime import datetime

# --- Predefined location ---
CENTER_LAT = 22.5592613
CENTER_LON = 88.3861048
RADIUS_M = 2000   # 2 km

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

CATEGORIES = {
    "1": ("amenity", "hospital"),
    "2": ("amenity", "pharmacy"),
    "3": ("amenity", "restaurant"),
    "4": ("shop", "supermarket"),
    "5": ("shop", "clothes"),
}

def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def bearing_deg(lat1, lon1, lat2, lon2):
    """Bearing in degrees from point 1 to 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    x = math.sin(delta_lon) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(delta_lon)
    brng = math.atan2(x, y)
    return (math.degrees(brng) + 360) % 360

def bearing_to_direction(bearing):
    """Convert bearing to approximate direction."""
    if 337.5 <= bearing or bearing < 22.5:
        return "N / Straight"
    elif 22.5 <= bearing < 67.5:
        return "NE / Slight Right"
    elif 67.5 <= bearing < 112.5:
        return "E / Right"
    elif 112.5 <= bearing < 157.5:
        return "SE / Sharp Right"
    elif 157.5 <= bearing < 202.5:
        return "S / Behind"
    elif 202.5 <= bearing < 247.5:
        return "SW / Sharp Left"
    elif 247.5 <= bearing < 292.5:
        return "W / Left"
    elif 292.5 <= bearing < 337.5:
        return "NW / Slight Left"
    else:
        return "-"

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
        brng = bearing_deg(center_lat, center_lon, lat, lon)
        direction = bearing_to_direction(brng)
        places.append({
            "osm_id": f"{el['type']}/{el['id']}",
            "name": name,
            "tag_value": tags.get(key, ""),
            "lat": lat,
            "lon": lon,
            "distance_m": round(dist,1),
            "bearing_deg": round(brng,1),
            "direction": direction
        })
    return sorted(places, key=lambda x: x["distance_m"])

def save_csv(places, filename):
    fields = ["osm_id","name","tag_value","lat","lon","distance_m","bearing_deg","direction"]
    with open(filename,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(places)

def main():
    print("Select what you want to search near the point:")
    for k, v in CATEGORIES.items():
        print(f"{k}. {v[1].capitalize()}")
    choice = input("Enter choice number: ").strip()
    if choice not in CATEGORIES:
        print("Invalid choice. Exiting.")
        return

    key, value = CATEGORIES[choice]
    print(f"\nSearching for {value} around ({CENTER_LAT}, {CENTER_LON}), radius={RADIUS_M} m ...")

    query = build_overpass_query(CENTER_LAT, CENTER_LON, RADIUS_M, key, value)
    try:
        data = run_overpass(query)
    except Exception as e:
        print("Error:", e)
        return

    places = extract_places(data, CENTER_LAT, CENTER_LON, key)
    print(f"Found {len(places)} {value}(s).\n")

    for i, s in enumerate(places[:20],1):
        print(f"{i:2d}. {s['name']} | {s['tag_value'] or '-'} | Distance: {s['distance_m']} m | Direction: {s['direction']} | ({s['lat']:.6f},{s['lon']:.6f})")

    if places:
        nearest = places[0]
        print(f"\nNearest {value}: {nearest['name']} ({nearest['distance_m']} m away, Direction: {nearest['direction']})")

    # Save CSV
    out = f"{value}_{RADIUS_M}m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_csv(places, out)
    print(f"\n📂 Saved full list to {out}")

if __name__=="__main__":
    main()
