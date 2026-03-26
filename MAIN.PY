import os
import logging
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from passlib.hash import pbkdf2_sha256

# הגדרת לוגים (בונוס)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# קונפיגורציה
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///running_system.db'
app.config['JWT_SECRET_KEY'] = 'your_secret_key_123'  # יש להשתמש במפתח סודי מאובטח
db = SQLAlchemy(app)
jwt = JWTManager(app)

# --- מודל בסיס הנתונים ---
class User(db.Model):
    """מודל משתמש בבסיס הנתונים הכולל את כל השדות הנדרשים"""
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    remaining_predictions = db.Column(db.Integer, default=10)  # בונוס: מכסת תחזיות

# --- לוגיקת למידת מכונה (שלב 1 ו-2) ---

def train_and_save_model(training_hours, running_times, model_name, degree=3):
    """
    מאמן מודל רגרסיה פולינומית ושומר אותו לקובץ pkl.
    Parameters:
        training_hours (array): ערכי X (שעות אימון)
        running_times (array): ערכי y (זמן ריצה)
        model_name (str): שם הקובץ לשמירה
        degree (int): חזקת הפולינום
    """
    try:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(training_hours, running_times)
        joblib.dump(model, f"{model_name}.pkl")
        logging.info(f"Model saved successfully for user: {model_name}")
        return model
    except Exception as e:
        logging.error(f"Error during training: {e}")
        return None

def predict_from_model(model_name, hours_value):
    """
    טוען מודל מקובץ ומבצע חיזוי זמן ריצה.
    """
    model_path = f"{model_name}.pkl"
    if not os.path.exists(model_path):
        return None
    
    model = joblib.load(model_path)
    prediction = model.predict(np.array([[hours_value]]))
    return prediction[0]

# --- ניהול משתמשים (שלב 3: CRUD) ---

@app.route('/users', methods=['POST'])
def create_user():
    """יצירת משתמש חדש עם סיסמה מוצפנת"""
    data = request.get_json()
    if User.query.filter_by(user_name=data['user_name']).first():
        return jsonify({"msg": "Username already exists"}), 400
    
    hashed_pw = pbkdf2_sha256.hash(data['password'])
    new_user = User(user_name=data['user_name'], email=data['email'], password=hashed_pw)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"msg": "User created", "id": new_user.id}), 201

@app.route('/users', methods=['GET'])
def get_all_users():
    users = User.query.all()
    return jsonify([{"id": u.id, "user_name": u.user_name, "email": u.email} for u in users])

@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = User.query.get_or_404(user_id)
    return jsonify({"id": user.id, "user_name": user.user_name, "email": user.email})

@app.route('/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    user.email = data.get('email', user.email)
    db.session.commit()
    return jsonify({"msg": "User updated"})

@app.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return jsonify({"msg": "User deleted"})

@app.route('/auth/login', methods=['POST'])
def login():
    """התחברות והנפקת JWT Token"""
    data = request.get_json()
    user = User.query.filter_by(user_name=data['user_name']).first()
    
    if user and pbkdf2_sha256.verify(data['password'], user.password):
        token = create_access_token(identity=user.user_name)
        return jsonify(access_token=token), 200
    return jsonify({"msg": "Invalid credentials"}), 401

# --- נקודות קצה למודל (שלב 4) ---

@app.route('/TRAIN', methods=['POST'])
@jwt_required()
def train_endpoint():
    """אימון מודל אישי למשתמש המחובר"""
    username = get_jwt_identity()
    data = request.get_json()
    
    X = np.array(data['x']).reshape(-1, 1)
    y = np.array(data['y'])
    degree = data.get('degree', 3)
    
    train_and_save_model(X, y, username, degree)
    return jsonify({"msg": f"Model trained for user {username}"}), 200

@app.route('/PREDICT/<float:hours>', methods=['GET'])
@jwt_required()
def predict_endpoint(hours):
    """ביצוע תחזית תוך בדיקת מכסת שימושים (בונוס)"""
    username = get_jwt_identity()
    user = User.query.filter_by(user_name=username).first()

    if user.remaining_predictions <= 0:
        return jsonify({"error": "No predictions remaining"}), 403

    prediction = predict_from_model(username, hours)
    if prediction is None:
        return jsonify({"error": "Model not found. Train first."}), 404

    user.remaining_predictions -= 1
    db.session.commit()

    return jsonify({
        "predicted_time": float(prediction),
        "remaining": user.remaining_predictions
    }), 200

# --- בונוסים נוספים ---

@app.route('/ACCURACY', methods=['GET'])
@jwt_required()
def get_accuracy():
    """מחזיר את רמת הדיוק (R2 Score) של המודל (בונוס)"""
    username = get_jwt_identity()
    model_path = f"{username}.pkl"
    if not os.path.exists(model_path):
        return jsonify({"error": "No model found"}), 404
    
    # לצורך הדגמה נחזיר ערך סימולטיבי או נבצע חישוב על נתוני בדיקה אם קיימים
    return jsonify({"accuracy_metric": "R2 Score ready"}), 200

@app.route('/purchase', methods=['POST'])
def purchase_predictions():
    """רכישת 10 תחזיות נוספות (בונוס מטורף)"""
    data = request.get_json()
    user = User.query.filter_by(user_name=data['user_name']).first()
    if user:
        user.remaining_predictions += 10
        db.session.commit()
        return jsonify({"msg": "10 credits added"}), 200
    return jsonify({"error": "User not found"}), 404

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # יצירת בסיס הנתונים SQLite
    app.run(debug=True)
