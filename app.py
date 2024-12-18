# Main entry point for our project
import pickle
from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

#Flask app - starting point of our api
app = Flask(__name__)

nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


def predict_spam(message):
    #Preprocess the message 
    transform_sms = transform_text(message)
    print("this is transform message", transform_sms)

    #Vectorise the preprocessed message
    # Wrap the transformed message in a list
    vector_input = tfidf.transform([transform_sms])
    print("this is  vector input",vector_input)

    #Predict using ML model
    result = model.predict(vector_input)[0]

    return result


@app.route('/') #homepage
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST']) # predict route
def predict():
    if request.method == 'POST':
        input_sms = request.form['message']
        result = predict_spam(input_sms)
        print(result)
        return render_template('index.html', result = result)

if __name__ == '__main__':
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    app.run(host='0.0.0.0') # localhost ip address = 0.0.0.0:5000