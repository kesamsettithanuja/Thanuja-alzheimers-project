from flask import Flask, render_template, request, redirect, session
import numpy as np
import os
import sqlite3
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------- APP CONFIG ----------------
app = Flask(__name__)
app.secret_key = "alz_secret_key"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- DATABASE ----------------
def db():
    return sqlite3.connect("users.db")

# Create users table
con = db()
con.execute("CREATE TABLE IF NOT EXISTS users (u TEXT, p TEXT)")
con.commit()
con.close()

# ---------------- LOAD MODEL ----------------
model = load_model("model/alzheimer_model.h5")
print("✅ Model Loaded Successfully")

with open("model/class_indices.json", "r") as f:
    class_indices = json.load(f)

classes = [None] * len(class_indices)

for class_name, index in class_indices.items():
    classes[index] = class_name

print("Classes:", classes)

# ---------------- ROUTES ----------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["POST"])
def register():
    u = request.form["username"]
    p = request.form["password"]

    con = db()
    con.execute("INSERT INTO users VALUES (?, ?)", (u, p))
    con.commit()
    con.close()

    return redirect("/")


@app.route("/login", methods=["POST"])
def login():
    u = request.form["username"]
    p = request.form["password"]

    con = db()
    user = con.execute(
        "SELECT * FROM users WHERE u=? AND p=?",
        (u, p)
    ).fetchone()
    con.close()

    if user:
        session["user"] = u
        return redirect("/dashboard")

    return "❌ Invalid Login Details"


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")

    return render_template("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect("/")

    file = request.files["image"]

    if file.filename == "":
        return "Please select an image"

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    # Image Preprocessing
    img = tf.keras.preprocessing.image.load_img(
        path,
        target_size=(224, 224)
    )

    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img, verbose=0)

    index = np.argmax(pred)
    confidence = float(np.max(pred)) * 100
    class_name = classes[index]

    print("Prediction:", pred)
    print("Index:", index)
    print("Class:", class_name)
    print("Confidence:", confidence)

    if confidence < 60:
        message = "⚠️ Prediction Not Confident"
        status = "warning"

    elif class_name == "Non_Demented":
        message = "✅ No Alzheimer's Disease Detected"
        status = "safe"

    else:
        message = f"🧠 {class_name} Detected"
        status = "disease"

    return render_template(
        "dashboard.html",
        image_path=path,
        message=message,
        confidence=round(confidence, 2),
        status=status
    )


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)