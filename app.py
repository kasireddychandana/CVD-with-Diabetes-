from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import pickle
import numpy as np

app = Flask(__name__)

# Database setup
db = sqlite3.connect('chd_users.db', check_same_thread=False)
cursor = db.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )
''')
db.commit()

# Load ML models and threshold
with open('saved/xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('saved/cat_model.pkl', 'rb') as f:
    cat_model = pickle.load(f)

with open('saved/meta_model.pkl', 'rb') as f:
    meta_model = pickle.load(f)

with open('saved/best_threshold.txt', 'r') as f:
    best_threshold = float(f.read().strip())

print("✅ Models and threshold loaded")

# Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/registration', methods=['GET', 'POST'])
def registration():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']

        if password != confirmpassword:
            return render_template('registration.html', msg='Passwords do not match!')

        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            return render_template('registration.html', msg='Email already registered!')

        cursor.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                       (name, email, password))
        db.commit()
        return render_template('registration.html', msg='User registered successfully!')
    return render_template('registration.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        if user and user[3] == password:
            return redirect(url_for('prediction'))
        return render_template('login.html', msg='Invalid credentials')
    return render_template('login.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            # Extract and convert input values
            glucose = float(request.form['glucose'])
            diabetes = float(request.form['diabetes'])
            age = float(request.form['age'])
            bmi = float(request.form['bmi'])
            sysBP = float(request.form['sysBP'])
            diaBP = float(request.form['diaBP'])
            totChol = float(request.form['totChol'])
            cigsPerDay = float(request.form['cigsPerDay'])

            # Create input array
            new_patient = np.array([[glucose, diabetes, age, bmi, sysBP, diaBP, totChol, cigsPerDay]])

            # Predict probabilities using base models
            xgb_prob = xgb_model.predict_proba(new_patient)[:, 1]
            cat_prob = cat_model.predict_proba(new_patient)[:, 1]

            # Predict with meta-model
            meta_input = np.column_stack((xgb_prob, cat_prob))
            final_prob = meta_model.predict_proba(meta_input)[:, 1]
            prediction = int(final_prob[0] >= best_threshold)

            result = 'High Risk (CHD)' if prediction == 1 else 'Low Risk (No CHD)'
            prob = round(final_prob[0] * 100, 2)

            return render_template('prediction.html', result=result, prob=prob)

        except Exception as e:
            return render_template('prediction.html', error=f'Error: {str(e)}')

    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
