import os
import time
import threading
from flask import Flask, request, jsonify, render_template

from sound_combined import classify_combined
from describe import describe_animal
from image_gen import generate_image

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'data'
app.config['IMAGE_OUTPUT'] = 'static/generated/images'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGE_OUTPUT'], exist_ok=True)

# GLOBAL SINGLE-JOB STATE
progress = {"percent": 0, "stage": "Idle"}
results = None
processing_thread = None



# -------------------------
# Utilities
# -------------------------
def set_progress(pct, text):
    progress["percent"] = pct
    progress["stage"] = text
    print(f"[PROGRESS] {pct}% - {text}")



# -------------------------
# Processing Worker
# -------------------------
def do_processing(file_path):
    global results
    results = None

    try:
        # -------- Stage 1: Analyze Audio --------
        set_progress(5, "Analyzing Audio…")
        result = classify_combined(file_path)
        animal, confidence = result["animal"]
        time.sleep(0.5)

        set_progress(25, "Audio Analyzed")

        # -------- Stage 2: Description --------
        set_progress(35, "Creating Description…")
        description = describe_animal(animal)
        time.sleep(0.5)

        set_progress(50, "Description Generated")

        # -------- Stage 3: Image Generation --------
        set_progress(60, "Generating Image…")
        image = generate_image(animal)
        filename = f"{animal.lower()}_{int(time.time())}.png"
        image_path = os.path.join(app.config["IMAGE_OUTPUT"], filename)
        image.save(image_path)

        set_progress(100, "Complete")

        # Store results
        results = {
            "animal": animal,
            "confidence": confidence,
            "description": description,
            "image_url": "/" + image_path
        }

    finally:
        try:
            os.remove(file_path)
        except:
            pass



# -------------------------
# Routes
# -------------------------
@app.route("/")
def upload_page():
    return render_template("upload.html")


@app.route("/start-process", methods=["POST"])
def start_process():
    global processing_thread, progress

    f = request.files["file"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
    f.save(file_path)

    progress = {"percent": 0, "stage": "Analyzing Audio…"}  # reset

    processing_thread = threading.Thread(target=do_processing, args=(file_path,), daemon=True)
    processing_thread.start()

    return jsonify({"status": "started"})


@app.route("/progress")
def get_progress():
    return jsonify(progress)


@app.route("/results")
def show_results():
    global results
    if results is None:
        return "Still processing", 202
    return render_template("results.html", **results)


if __name__ == "__main__":
    app.run(debug=True)
