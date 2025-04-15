from flask import Flask, request, jsonify, render_template
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_url_path='/static', static_folder='static')


# Update this with your Gemini 1.5-compatible API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"

def call_gemini_vision(base64_image):
    headers = {
        "Content-Type": "application/json"
    }

    prompt_text = "This is a hand sign from Indian Sign Language (ISL). Identify the corresponding English alphabet (A-Z). Respond only with the alphabet."

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt_text},
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": base64_image
                    }
                }
            ]
        }]
    }

    response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        except:
            return "Unexpected response format"
    else:
        print("Gemini error:", response.text)
        return "Gemini API error"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        image_data = data.get('image', '')

        if not image_data or ',' not in image_data:
            return jsonify({'prediction': 'Empty image data received'}), 400

        base64_image = image_data.split(',')[1]
        prediction = call_gemini_vision(base64_image)

        return jsonify({'prediction': prediction})
    
    except Exception as e:
        print("Server error:", e)
        return jsonify({'prediction': 'Server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
 