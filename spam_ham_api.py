import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from flask import Flask, request, jsonify

# Load the model and vectorizer
bnb_loaded = joblib.load('bernoulli_nb_model.pkl')
tfidf_loaded = joblib.load('tfidf_vectorizer.pkl')

# Initialize PorterStemmer
ps = PorterStemmer()

# Function to preprocess and transform the input message
def transform_Msg(Msg):
    Msg = Msg.lower()
    Msg = nltk.word_tokenize(Msg)
    y = [i for i in Msg if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Initialize Flask app
app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return jsonify({"status": online})

# Route to handle form submission and make prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        new_data = request.form['message']
    else:
        new_data = request.args.get('message')

    if not new_data:
        return jsonify({"error": "No message provided"}), 400

    new_data_transformed = tfidf_loaded.transform([transform_Msg(new_data)]).toarray()
    prediction = bnb_loaded.predict(new_data_transformed)

    result = "Ham" if prediction == 0 else "Spam"

    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)