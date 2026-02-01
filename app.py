from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("titanic_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pclass = int(request.form['pclass'])
    sex = 1 if request.form['sex'] == 'female' else 0
    age = float(request.form['age'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    fare = float(request.form['fare'])
    embarked_map = {"C": 1, "Q": 2, "S": 0}
    embarked = embarked_map[request.form['embarked']]

    # Match training features: [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    prediction = model.predict(features)

    result = "✅ Survived" if prediction[0] == 1 else "❌ Did not survive"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)