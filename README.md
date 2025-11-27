
# AnimalPRF (Animal Pre-defined Roar Finder - AI-Powered Animal Sound Visualization System)

**AnimalPRF** is an AI-powered Flask web application that transforms animal sounds into visual and descriptive representations.  
It uses pretrained AI models to classify animal sounds, generate brief descriptions, and create representative images — combining audio, language, and vision into a single experience.

---

## Features
- Upload an audio file (e.g., `.wav`, `.mp3`)
- The system automatically:
  - **Classifies the animal sound** using a YAMNet + AST ensemble
  - **Generates a short factual description** using a local TinyLlama LLM
  - **Creates a photorealistic AI image** using Stable Diffusion XL Turbo
- Real-time progress UI:
  - Dynamic progress bar
  - Hourglass loading animation
  - Status messages for each pipeline stage
- Automatic cleanup of temporary audio/image files
- Fully local models (no external API calls)

---

## Tech Stack
### Frontend
- HTML5 / CSS3  
- Custom styled UI with progress bar and animated hourglass  
- JavaScript EventSource (SSE) for real-time progress updates  

### Backend (Flask)
- Python 3.10+  
- Multithreaded pipeline processing (sound → text → image)  
- SSE progress updates  
- Automatic model lazy-loading for speed  
- Temporary file management and cleanup  

### Models Used
#### 🔊 Sound Classification
- **YAMNet** (TensorFlow Hub)  
- **AST — Audio Spectrogram Transformer** (MIT / HuggingFace)  
Used together for robust animal vocalization recognition.

#### 📝 Description Generation
- **TinyLlama 1.1B Chat**  
  - Fully local  
  - Small enough to run on CPU  
  - Generates clean, coherent 3–5 sentence descriptions based on Wikipedia information  

#### 🖼️ Image Generation
- **Stable Diffusion XL Turbo** (stabilityai/sdxl-turbo)  
  - Fast diffusion variant  
  - Photorealistic images  
  - Runs locally through HuggingFace diffusers 

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