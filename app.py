import requests

# -----------------------------
# 1️⃣ Google API Key (restricted key use করুন)
# -----------------------------
API_KEY = "AIzaSyBeu3KiIFwrKzpesYjKbNPCB-00w2ImUAk"

# -----------------------------
# 2️⃣ Predefined location
# -----------------------------
latitude = 22.5577851
longitude = 88.3948815

# -----------------------------
# 3️⃣ Search parameters
# -----------------------------
initial_radius = 2000  # 2 km
max_radius = 5000      # 5 km
keywords = ["hospital", "pharmacy"]

# -----------------------------
# 4️⃣ Function to get nearby places
# -----------------------------
def get_nearby_places(lat, lng, keyword, radius):
    url = (
        f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={lat},{lng}&radius={radius}&keyword={keyword}&key={API_KEY}"
    )
    response = requests.get(url)
    data = response.json()
    return data

# -----------------------------
# 5️⃣ Terminal Output
# -----------------------------
if __name__ == "__main__":
    for keyword in keywords:
        radius = initial_radius
        while radius <= max_radius:
            print(f"\n🔎 Searching nearby {keyword}s within {radius} meters...")
            data = get_nearby_places(latitude, longitude, keyword, radius)
            status = data.get("status")
            
            if status != "OK" and status != "ZERO_RESULTS":
                print(f"❌ Error: {status}")
                break
            
            results = data.get("results", [])
            if results:
                for i, place in enumerate(results, start=1):
                    name = place.get("name")
                    address = place.get("vicinity")
                    lat = place["geometry"]["location"]["lat"]
                    lng = place["geometry"]["location"]["lng"]
                    print(f"{i}. {name} ({address}) -> Lat: {lat}, Lng: {lng}")
                break  # results found, no need to increase radius
            else:
                if radius == max_radius:
                    print("No results found within maximum radius.")
                    break
                radius += 1000  # increase radius by 1 km
