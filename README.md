
# AnimalVision (AI-Powered Animal Sound Visualization System)

**AnimalVision** is an AI-powered Flask web application that transforms animal sounds into visual and descriptive representations.  
It uses pretrained AI models to classify animal sounds, generate brief descriptions, and create representative images — combining audio, language, and vision into a single experience.

---

## Features
- Upload an animal sound file (e.g., `.wav`) and receive:
  - The predicted animal name  
  - A short AI-generated description  
  - An AI-generated image of the animal  
- Flask-based web interface for easy interaction  
- Uses multiple **pretrained AI models** (no training required)  
- Modular design for future expansion or model swaps  

---

## Tech Stack
- **Frontend:** HTML5 + CSS (Flask templates)  
- **Backend:** Flask (Python)  
- **Sound Classification:** TensorFlow Hub (YAMNet / PANNs / Wav2Vec2)  
- **Description Generation:** DeepAI Text Generation API  
- **Image Rendering:** Stable Diffusion / DALL·E / DeepAI Text-to-Image API  
- **Libraries:** Librosa, NumPy, Matplotlib, TensorFlow  
- **Version Control:** Git & GitHub  


---

## Installation & Setup

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