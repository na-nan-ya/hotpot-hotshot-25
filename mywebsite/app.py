from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS module

app = Flask(__name__)

# Enable CORS for all origins and preflight requests (OPTIONS)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

@app.route('/process_text', methods=['POST', 'OPTIONS'])
def process_text():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'message': 'CORS preflight successful'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response

    # Handle actual POST request
    text = request.json.get('text')

    if text and "yes" in text.lower():
        return jsonify({"result": True})
    else:
        return jsonify({"result": False})

if __name__ == '__main__':
    app.run(debug=True, port= 5000)
