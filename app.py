from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import numpy as np
import io
import csv

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'  # Change this!

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database model for users
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Database model for predictions
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    tenure = db.Column(db.Integer)
    monthly_charges = db.Column(db.Float)
    total_charges = db.Column(db.Float)
    contract = db.Column(db.String(50))
    payment_method = db.Column(db.String(50))
    internet_service = db.Column(db.String(50))
    senior_citizen = db.Column(db.Integer)
    partner = db.Column(db.Integer)
    dependents = db.Column(db.Integer)
    prediction = db.Column(db.Integer)  # 0 or 1
    probability = db.Column(db.Float)

# Load model
model = joblib.load('churn_model.pkl')

# Create tables (if not exist)
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')  # landing page

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('signup.html', error='Username already exists')
        
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        session['user_id'] = new_user.id
        session['username'] = new_user.username
        return redirect(url_for('dashboard'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    prediction = None
    probability = None
    pred_class = None
    
    if request.method == 'POST':
        input_data = {
            'tenure': int(request.form['tenure']),
            'monthly_charges': float(request.form['monthly_charges']),
            'total_charges': float(request.form['total_charges']),
            'contract': request.form['contract'],
            'payment_method': request.form['payment_method'],
            'internet_service': request.form['internet_service'],
            'senior_citizen': int(request.form['senior_citizen']),
            'partner': int(request.form['partner']),
            'dependents': int(request.form['dependents'])
        }
        
        input_df = pd.DataFrame([input_data])
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        pred_class = 'Churn' if pred == 1 else 'No Churn'
        probability = prob[1] if pred == 1 else prob[0]
        
        new_pred = Prediction(
            user_id=session['user_id'],
            tenure=input_data['tenure'],
            monthly_charges=input_data['monthly_charges'],
            total_charges=input_data['total_charges'],
            contract=input_data['contract'],
            payment_method=input_data['payment_method'],
            internet_service=input_data['internet_service'],
            senior_citizen=input_data['senior_citizen'],
            partner=input_data['partner'],
            dependents=input_data['dependents'],
            prediction=int(pred),
            probability=probability
        )
        db.session.add(new_pred)
        db.session.commit()
    
    recent_preds = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.timestamp.desc()).limit(10).all()
    
    return render_template('dashboard.html', 
                         prediction=pred_class, 
                         probability=probability,
                         recent_preds=recent_preds)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.timestamp.desc()).all()
    return render_template('history.html', predictions=predictions)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/chart-data')
def chart_data():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user_preds = Prediction.query.filter_by(user_id=session['user_id']).all()
    total = len(user_preds)
    churn_count = sum(1 for p in user_preds if p.prediction == 1)
    no_churn_count = total - churn_count
    probs = [p.probability for p in user_preds]
    
    return jsonify({
        'churn_pie': {
            'labels': ['Churn', 'No Churn'],
            'values': [churn_count, no_churn_count]
        },
        'prob_dist': probs
    })

@app.route('/export')
def export():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    predictions = Prediction.query.filter_by(user_id=session['user_id']).all()
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Timestamp', 'Tenure', 'Monthly Charges', 'Total Charges', 
                     'Contract', 'Payment Method', 'Internet Service', 'Senior Citizen',
                     'Partner', 'Dependents', 'Prediction', 'Probability'])
    
    for p in predictions:
        writer.writerow([p.id, p.timestamp, p.tenure, p.monthly_charges, p.total_charges,
                         p.contract, p.payment_method, p.internet_service, p.senior_citizen,
                         p.partner, p.dependents, p.prediction, p.probability])
    
    output.seek(0)
    return Response(output, mimetype='text/csv', 
                    headers={"Content-Disposition": "attachment;filename=predictions.csv"})

if __name__ == '__main__':
    app.run(debug=True)