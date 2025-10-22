
# 🐾 AnimalPRF (Animal Pre-Trained Roar Finder)

AnimalPRF is an AI-powered Flask web application that identifies animal sounds from uploaded audio clips.  
It uses **YAMNet** (a pretrained TensorFlow model) and a fine-tuned **EfficientNet-B0** CNN to classify sounds from rural animals such as cats, dogs, birds, foxes, wolves, elk, deer, and bears.

---

## 🚀 Features
- Upload or record an animal sound and receive a predicted species.
- Flask-based web interface for simple use and team collaboration.
- Modular design for integrating multiple AI models.
- Uses open datasets (ESC-50, Xeno-Canto) for training and evaluation.

---

## 🧠 Tech Stack
- **Frontend:** HTML5 + CSS (Flask templates)
- **Backend:** Flask (Python)
- **AI Models:** TensorFlow + TensorFlow Hub (YAMNet, EfficientNet-B0)
- **Libraries:** Librosa, NumPy, Matplotlib
- **Version Control:** Git & GitHub

---

## 🧩 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/CSC-5240-AI-Final-Project/AnimalPRF.git
cd AnimalPRF
```

### 2. Create and Activate a Virtual Environment
#### Windows:
```bash
py -m venv venv
venv\Scripts\activate
```

#### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
py app.py
```
Once running, open your browser and visit:  [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Team Members

-   **Brian Kemp** – bkemp42@tntech.edu
-   **Marim Elhanafy** – mmelhanafy42@tntech.edu
-   **Ryan Thornton** – rthornton42@tntech.edu