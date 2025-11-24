from flask import Flask, render_template, request
import os

app = Flask(__name__, template_folder='app/templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    filename = os.path.join('data', file.filename)
    file.save(filename)
    return f"File {file.filename} uploaded successfully!"

if __name__ == '__main__':
    app.run(debug=True)
