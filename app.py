from flask import Flask, render_template, send_file, Response, request, jsonify
import psycopg2
import io
import os
import tempfile
from datetime import datetime, timedelta
from geopy.distance import geodesic
import folium
from folium.plugins import MarkerCluster
from gradio_client import Client, handle_file

app = Flask(__name__)

# Database connection details
DB_NAME = "images_aa6r"
DB_USER = "images_aa6r_user"
DB_PASSWORD = "nSdWmdxrcu8DCKINowfffad5KeL6Ukrk"
DB_HOST = "dpg-cvohgfumcj7s7384tbdg-a.oregon-postgres.render.com"
DB_PORT = "5432"

# ✅ Preload Gradio client once
client = Client("RohithAttoli/cotton-server")

def get_db_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return conn

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/tested_samples')
def tested_samples():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, prediction, confidence FROM image_results;")
    rows = cursor.fetchall()
    conn.close()
    return render_template('tested_samples.html', data=rows)

@app.route('/image/<int:image_id>')
def get_image(image_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT image FROM image_results WHERE id = %s;", (image_id,))
    result = cursor.fetchone()
    conn.close()

    if result and result[0]:
        return Response(result[0], mimetype='image/jpeg')
    return "Image not found", 404

# Scan image page
@app.route('/scan-image-page')
def scan_image_page():
    return render_template('scan_image.html')

@app.route("/scan-image", methods=["POST"])
def scan_image():
    image = request.files["image"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
        image.save(temp.name)
        result = client.predict(image=handle_file(temp.name), api_name="/predict")

    return jsonify({"output": result})

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image.save(temp.name)
            try:
                result = client.predict(image=handle_file(temp.name), api_name="/predict")
                print("✅ Prediction:", result)
                return jsonify({"result": result})
            except Exception as e:
                print("❌ Prediction error:", e)
                return jsonify({"error": "Prediction failed"}), 500
            finally:
                os.unlink(temp.name)
    elif request.json and 'message' in request.json:
        print(f"⚠️ {request.json['message']}")
        return jsonify({"status": "offline message received"}), 200
    else:
        return jsonify({"error": "No image or message"}), 400

# -------- Farm History Section --------

def fetch_images():
    conn = get_db_connection()
    cursor = conn.cursor()

    current_time = datetime.now()
    one_week_ago = current_time - timedelta(weeks=1)
    one_week_ago_str = one_week_ago.strftime('%Y-%m-%d %H:%M:%S')

    query = """
        SELECT image, latitude, longitude, timestamp, prediction
        FROM image_results
        WHERE timestamp >= %s
        AND CAST(prediction AS INTEGER) != 2
    """
    cursor.execute(query, (one_week_ago,))
    result = cursor.fetchall()

    cursor.close()
    conn.close()

    image_data = []
    for row in result:
        image_data.append({
            "image": row[0],
            "latitude": row[1],
            "longitude": row[2],
            "timestamp": row[3],
            "prediction": row[4]
        })
    return image_data

def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).meters

def create_cluster_map(image_data, radius=5, count_threshold=5):
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    marker_cluster = MarkerCluster().add_to(m)

    for i, data1 in enumerate(image_data):
        if data1["latitude"] is None or data1["longitude"] is None:
            continue

        nearby_count = 0
        for j, data2 in enumerate(image_data):
            if i == j:
                continue
            if data2["latitude"] is None or data2["longitude"] is None:
                continue

            distance = calculate_distance(
                (data1["latitude"], data1["longitude"]),
                (data2["latitude"], data2["longitude"])
            )

            if distance <= radius and data2["prediction"] != "2":
                nearby_count += 1

        if nearby_count >= count_threshold:
            folium.CircleMarker(
                location=[data1["latitude"], data1["longitude"]],
                radius=10,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.8
            ).add_to(marker_cluster)

    map_path = "static/cluster_map_with_danger.html"
    m.save(map_path)
    return map_path

@app.route('/field_status')
def field_status():
    image_data = fetch_images()
    create_cluster_map(image_data)
    return render_template('field_status.html')

# ----------------------------------------

if __name__ == '__main__':
    app.run(debug=True)
