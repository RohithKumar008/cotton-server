from flask import Flask, render_template, send_file, Response
import psycopg2
import io
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")
# Replace with your actual credentials
DB_NAME = "images_aa6r"
DB_USER = "images_aa6r_user"
DB_PASSWORD = "nSdWmdxrcu8DCKINowfffad5KeL6Ukrk"
DB_HOST = "dpg-cvohgfumcj7s7384tbdg-a.oregon-postgres.render.com"
DB_PORT = "5432"

def get_db_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return conn

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

from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import tempfile

# Scan image page
@app.route('/scan-image-page')
def scan_image_page():
    return render_template('scan_image.html')

# API endpoint for processing the image
@app.route("/scan-image", methods=["POST"])
def scan_image():
    image = request.files["image"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
        image.save(temp.name)
        client = Client("RohithAttoli/cotton-server")
        result = client.predict(image=handle_file(temp.name), api_name="/predict")
if __name__ == "__main__":
    app.run(debug=True)
