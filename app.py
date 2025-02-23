from flask import Flask, render_template, request, jsonify
from classifier import use_classifier

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()  # Receive JSON data
    name = data.get('name', 'Guest')  # Extract 'name'
    
    file_path = "my_file.txt"

    with open(file_path, "w") as file:
        file.write(name)

    stuff = use_classifier("my_file.txt")
    return jsonify({"message": f"Hello, {name}! Data received. {stuff}"})  # Send response as JSON

if __name__ == '__main__':
    app.run(debug=True)
