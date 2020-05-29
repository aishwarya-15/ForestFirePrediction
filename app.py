from flask import Flask, request, render_template
from flask_wtf import FlaskForm
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def home():
    return render_template('forest.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(final)
    prediction = model.predict(final)


    if prediction == 1:
        return render_template('forest.html',pred='Your Forest is in Danger.')
    else:
        return render_template('forest.html',pred='Your Forest is safe.')

if __name__ == '__main__':
    app.run(debug=True)
