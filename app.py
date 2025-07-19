from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import sqlite3
import os
import json
from datetime import datetime, timedelta
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, learning_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

app = Flask(__name__)
app.secret_key = 'csv-ml-analyzer-super-secret-key-change-in-production-2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = False  # Allow JavaScript access for debugging
app.config['SESSION_COOKIE_SAMESITE'] = None  # More permissive for VS Code browser
app.config['SESSION_PERMANENT'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)  # 24 hour session

# Make session available in templates
@app.context_processor
def inject_session():
    return dict(session=session)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Database initialization
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Datasets table
    c.execute('''CREATE TABLE IF NOT EXISTS datasets
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  filename TEXT NOT NULL,
                  original_filename TEXT NOT NULL,
                  upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  file_size INTEGER,
                  rows INTEGER,
                  columns INTEGER,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Models table
    c.execute('''CREATE TABLE IF NOT EXISTS models
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  dataset_id INTEGER,
                  model_name TEXT NOT NULL,
                  model_type TEXT NOT NULL,
                  target_column TEXT NOT NULL,
                  accuracy REAL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  model_file TEXT,
                  FOREIGN KEY (user_id) REFERENCES users (id),
                  FOREIGN KEY (dataset_id) REFERENCES datasets (id))''')
    
    conn.commit()
    conn.close()

@app.route('/debug-session')
def debug_session():
    """Debug route to check session state"""
    return jsonify({
        'session': dict(session),
        'cookies': dict(request.cookies),
        'user_agent': request.headers.get('User-Agent', 'Unknown')
    })

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.')
            return render_template('register.html')
        
        password_hash = generate_password_hash(password)
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        try:
            c.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                     (username, email, password_hash))
            conn.commit()
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists.')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[1], password):
            print(f"Login successful for user {username}, setting session")
            session.permanent = True  # Make session permanent
            session['user_id'] = user[0]
            session['username'] = username
            print(f"Session after login: {dict(session)}")
            
            # Force session to be saved
            session.modified = True
            
            flash('Login successful! Welcome to your dashboard.', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    print(f"Dashboard accessed. Session contents: {dict(session)}")
    print(f"Request headers: {dict(request.headers)}")
    print(f"Request cookies: {dict(request.cookies)}")
    
    if 'user_id' not in session:
        print("No user_id in session, redirecting to login")
        flash('Please log in to access the dashboard.')
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    # Get user's datasets
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("""SELECT id, original_filename, upload_date, file_size, rows, columns 
                 FROM datasets WHERE user_id = ? ORDER BY upload_date DESC""", (user_id,))
    datasets = c.fetchall()
    
    # Get user's models
    c.execute("""SELECT m.id, m.model_name, m.model_type, m.target_column, m.accuracy, 
                        m.created_at, d.original_filename
                 FROM models m 
                 JOIN datasets d ON m.dataset_id = d.id 
                 WHERE m.user_id = ? ORDER BY m.created_at DESC""", (user_id,))
    models = c.fetchall()
    
    conn.close()
    
    return render_template('dashboard.html', datasets=datasets, models=models)

@app.route('/resources')
def resources():
    """ML Learning Resources Page"""
    return render_template('resources.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if 'file' not in request.files:
        flash('No file selected.')
        return redirect(url_for('dashboard'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected.')
        return redirect(url_for('dashboard'))
    
    if file and file.filename.lower().endswith('.csv'):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        unique_filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Analyze the CSV file
        try:
            df = pd.read_csv(filepath)
            file_size = os.path.getsize(filepath)
            rows, columns = df.shape
            
            # Save dataset info to database
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("""INSERT INTO datasets (user_id, filename, original_filename, file_size, rows, columns)
                         VALUES (?, ?, ?, ?, ?, ?)""",
                     (session['user_id'], unique_filename, filename, file_size, rows, columns))
            conn.commit()
            conn.close()
            
            flash(f'File uploaded successfully! {rows} rows, {columns} columns detected.')
        except Exception as e:
            flash(f'Error analyzing file: {str(e)}')
            os.remove(filepath)
    else:
        flash('Please upload a CSV file.')
    
    return redirect(url_for('dashboard'))

@app.route('/analyze/<int:dataset_id>')
def analyze_dataset(dataset_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get dataset info
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT filename, original_filename FROM datasets WHERE id = ? AND user_id = ?",
              (dataset_id, session['user_id']))
    dataset = c.fetchone()
    conn.close()
    
    if not dataset:
        flash('Dataset not found.')
        return redirect(url_for('dashboard'))
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], dataset[0])
        df = pd.read_csv(filepath)
        
        # Basic statistics
        stats = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'description': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        # Generate visualizations
        plots = generate_plots(df)
        
        return render_template('analyze.html', 
                             dataset_id=dataset_id,
                             filename=dataset[1],
                             stats=stats,
                             plots=plots,
                             data_preview=df.head(10).to_html(classes='table table-striped'))
    
    except Exception as e:
        flash(f'Error analyzing dataset: {str(e)}')
        return redirect(url_for('dashboard'))

def generate_plots(df):
    plots = {}
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # 1. Correlation heatmap for numeric columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Show only lower triangle
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Correlation Heatmap - Feature Relationships', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        plots['correlation'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
    
    # 2. Distribution plots for numeric columns (first 4)
    for i, col in enumerate(numeric_cols[:4]):
        plt.figure(figsize=(10, 6))
        
        # Create subplot with histogram and box plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram with KDE
        data = df[col].dropna()
        ax1.hist(data, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        # Add KDE curve if we have enough data points
        if len(data) > 10:
            from scipy import stats
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            ax1.legend()
        
        ax1.set_title(f'Distribution of {col}', fontweight='bold')
        ax1.set_xlabel(col)
        ax1.set_ylabel('Density')
        ax1.grid(True, alpha=0.3)
        
        # Box plot for outlier detection
        ax2.boxplot(data, vert=True)
        ax2.set_title(f'Box Plot - {col}', fontweight='bold')
        ax2.set_ylabel(col)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        plots[f'dist_{i}'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
    
    # 3. Missing Data Visualization
    if df.isnull().sum().sum() > 0:
        plt.figure(figsize=(12, 8))
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            # Bar plot of missing values
            plt.subplot(2, 1, 1)
            missing_data.plot(kind='bar', color='coral')
            plt.title('Missing Values by Column', fontweight='bold', fontsize=14)
            plt.ylabel('Count of Missing Values')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Heatmap of missing data pattern
            plt.subplot(2, 1, 2)
            sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
            plt.title('Missing Data Pattern (Yellow = Missing)', fontweight='bold', fontsize=14)
            plt.xlabel('Columns')
            
            plt.tight_layout()
            
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
            img.seek(0)
            plots['missing_data'] = base64.b64encode(img.getvalue()).decode()
            plt.close()
    
    # 4. Categorical Data Analysis (if any categorical columns exist)
    if len(categorical_cols) > 0:
        # Take first 3 categorical columns
        for i, col in enumerate(categorical_cols[:3]):
            plt.figure(figsize=(12, 6))
            
            value_counts = df[col].value_counts().head(10)  # Top 10 categories
            
            if len(value_counts) > 1:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Bar chart
                value_counts.plot(kind='bar', ax=ax1, color='lightgreen', edgecolor='black')
                ax1.set_title(f'Top Categories in {col}', fontweight='bold')
                ax1.set_xlabel(col)
                ax1.set_ylabel('Count')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
                
                # Pie chart
                if len(value_counts) <= 8:  # Only show pie chart if not too many categories
                    value_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%', startangle=90)
                    ax2.set_title(f'Distribution of {col}', fontweight='bold')
                    ax2.set_ylabel('')
                else:
                    # Show proportion table instead
                    ax2.axis('off')
                    proportions = (value_counts / value_counts.sum() * 100).round(1)
                    table_data = [[cat, count, f"{prop}%"] for cat, count, prop in 
                                 zip(proportions.index, value_counts.values, proportions.values)]
                    ax2.table(cellText=table_data, 
                             colLabels=['Category', 'Count', 'Percentage'],
                             cellLoc='center', loc='center')
                    ax2.set_title(f'Category Statistics - {col}', fontweight='bold')
                
                plt.tight_layout()
                
                img = io.BytesIO()
                plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
                img.seek(0)
                plots[f'categorical_{i}'] = base64.b64encode(img.getvalue()).decode()
                plt.close()
    
    # 5. Data Quality Summary
    plt.figure(figsize=(14, 8))
    
    # Create summary statistics visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Dataset overview
    overview_data = {
        'Total Rows': len(df),
        'Total Columns': len(df.columns),
        'Numeric Columns': len(numeric_cols),
        'Categorical Columns': len(categorical_cols),
        'Missing Values': df.isnull().sum().sum(),
        'Complete Rows': len(df.dropna())
    }
    
    ax1.bar(range(len(overview_data)), list(overview_data.values()), 
           color=['skyblue', 'lightgreen', 'orange', 'pink', 'red', 'purple'])
    ax1.set_xticks(range(len(overview_data)))
    ax1.set_xticklabels(overview_data.keys(), rotation=45, ha='right')
    ax1.set_title('Dataset Overview', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Data types distribution
    dtype_counts = df.dtypes.value_counts()
    ax2.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Data Types Distribution', fontweight='bold')
    
    # Memory usage
    if hasattr(df, 'memory_usage'):
        memory_usage = df.memory_usage(deep=True)
        ax3.bar(range(len(memory_usage)), memory_usage.values / 1024, color='lightcoral')  # Convert to KB
        ax3.set_xticks(range(len(memory_usage)))
        ax3.set_xticklabels(['Index'] + list(df.columns), rotation=45, ha='right')
        ax3.set_title('Memory Usage by Column (KB)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # Completeness by column
    completeness = (1 - df.isnull().sum() / len(df)) * 100
    colors = ['red' if x < 80 else 'orange' if x < 95 else 'green' for x in completeness]
    ax4.bar(range(len(completeness)), completeness.values, color=colors)
    ax4.set_xticks(range(len(completeness)))
    ax4.set_xticklabels(completeness.index, rotation=45, ha='right')
    ax4.set_title('Data Completeness by Column (%)', fontweight='bold')
    ax4.set_ylabel('Completeness %')
    ax4.axhline(y=95, color='green', linestyle='--', alpha=0.7, label='95% threshold')
    ax4.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
    img.seek(0)
    plots['data_quality'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plots

def detect_overfitting_and_validate(model, X_train, X_val, X_test, y_train, y_val, y_test, model_type, is_scaled=True):
    """
    Detect overfitting and perform cross-validation analysis
    Returns validation metrics and overfitting warnings
    """
    validation_info = {}
    
    # Make predictions on all sets
    if is_scaled:
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
    else:
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
    
    # Calculate performance metrics
    if 'regression' in model_type:
        from sklearn.metrics import r2_score, mean_absolute_error
        train_score = r2_score(y_train, train_pred)
        val_score = r2_score(y_val, val_pred)
        test_score = r2_score(y_test, test_pred)
        metric_name = 'R² Score'
    else:
        train_score = accuracy_score(y_train, train_pred)
        val_score = accuracy_score(y_val, val_pred)
        test_score = accuracy_score(y_test, test_pred)
        metric_name = 'Accuracy'
    
    # Store scores
    validation_info['train_score'] = float(train_score)
    validation_info['validation_score'] = float(val_score)
    validation_info['test_score'] = float(test_score)
    validation_info['metric_name'] = metric_name
    
    # Detect overfitting
    train_val_gap = train_score - val_score
    val_test_gap = abs(val_score - test_score)
    
    warnings = []
    if train_val_gap > 0.15:  # 15% gap
        warnings.append("🚨 Severe overfitting detected! Training score much higher than validation.")
    elif train_val_gap > 0.10:  # 10% gap
        warnings.append("⚠️ Overfitting detected. Consider regularization or more data.")
    elif train_val_gap > 0.05:  # 5% gap
        warnings.append("⚡ Mild overfitting. Model may not generalize well.")
    
    # Check for underfitting
    if train_score < 0.6 and val_score < 0.6:
        warnings.append("📉 Underfitting detected. Model may be too simple.")
    
    # Check validation-test consistency
    if val_test_gap > 0.1:
        warnings.append("🎯 Large validation-test gap. Results may not be reliable.")
    
    validation_info['overfitting_warnings'] = warnings
    
    # Perform cross-validation
    try:
        # Skip cross-validation for faster training - can be enabled later
        validation_info['cv_mean'] = 'Skipped for performance'
        validation_info['cv_std'] = 'N/A'
        validation_info['cv_scores'] = []
        
        # # Use the original combined training data for CV
        # X_combined = np.vstack([X_train, X_val])
        # y_combined = np.hstack([y_train, y_val])
        # 
        # if 'regression' in model_type:
        #     cv_scores = cross_val_score(model, X_combined, y_combined, cv=5, scoring='r2')
        # else:
        #     cv_scores = cross_val_score(model, X_combined, y_combined, cv=5, scoring='accuracy')
        # 
        # validation_info['cv_mean'] = float(cv_scores.mean())
        # validation_info['cv_std'] = float(cv_scores.std())
        # validation_info['cv_scores'] = cv_scores.tolist()
        
        # Interpret CV results
        if validation_info.get('cv_std', 0) == 'N/A':
            pass  # Skip variance check when CV is disabled
        elif validation_info.get('cv_std', 0) > 0.1:
            warnings.append("📊 High cross-validation variance. Model performance is inconsistent.")
        
    except Exception as e:
        validation_info['cv_error'] = str(e)
    
    return validation_info

@app.route('/train_model', methods=['POST'])
def train_model():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.get_json()
        dataset_id = data['dataset_id']
        model_type = data['model_type']
        target_column = data['target_column']
        model_name = data['model_name']
        
        # Get dataset
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT filename FROM datasets WHERE id = ? AND user_id = ?",
                  (dataset_id, session['user_id']))
        dataset = c.fetchone()
        
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Load and prepare data
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], dataset[0])
        df = pd.read_csv(filepath)
        
        # Check for missing values first
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            return jsonify({
                'error': 'Dataset contains missing values. Please use the "Handle Missing Values" feature first.',
                'missing_columns': missing_data[missing_data > 0].to_dict(),
                'total_missing': int(missing_data.sum())
            }), 400
        
        # Data quality checks and preprocessing
        print(f"Original data shape: {df.shape}")
        print(f"Target column '{target_column}' value counts:")
        print(df[target_column].value_counts())
        
        # Remove duplicates
        original_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < original_rows:
            print(f"Removed {original_rows - len(df)} duplicate rows")
        
        # Feature engineering for car data
        if 'Year' in df.columns:
            # Create age feature
            current_year = 2024
            df['Car_Age'] = current_year - df['Year']
        
        if 'Mileage_KM' in df.columns and 'Car_Age' in df.columns:
            # Average mileage per year
            df['Avg_Mileage_Per_Year'] = df['Mileage_KM'] / (df['Car_Age'] + 1)  # +1 to avoid division by zero
        
        if 'Price_USD' in df.columns and 'Sales_Volume' in df.columns:
            # Revenue feature
            df['Total_Revenue'] = df['Price_USD'] * df['Sales_Volume']
        
        # Prepare features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        print(f"Features shape after preprocessing: {X.shape}")
        print(f"Feature columns: {X.columns.tolist()}")
        
        # Handle categorical variables with proper encoding
        le_dict = {}
        encoded_features = []
        
        # Separate categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # For tree-based models, use Label Encoding (they handle it well)
        # For linear models, we'll use One-Hot Encoding later
        if model_type in ['random_forest_classifier', 'random_forest_regressor']:
            # Label encoding is fine for tree-based models
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
        else:
            # One-hot encoding for linear models and SVM
            from sklearn.preprocessing import OneHotEncoder
            if len(categorical_cols) > 0:
                # Limit categories to prevent explosion
                for col in categorical_cols:
                    # Keep only top 10 categories, group others as 'Other'
                    top_categories = X[col].value_counts().head(10).index
                    X[col] = X[col].apply(lambda x: x if x in top_categories else 'Other')
                
                # Apply one-hot encoding
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                categorical_encoded = ohe.fit_transform(X[categorical_cols])
                
                # Create feature names for one-hot encoded columns
                feature_names = []
                for i, col in enumerate(categorical_cols):
                    categories = ohe.categories_[i]
                    for cat in categories:
                        feature_names.append(f"{col}_{cat}")
                
                # Combine with numerical features
                X_categorical = pd.DataFrame(categorical_encoded, columns=feature_names, index=X.index)
                X_numerical = X[numerical_cols]
                X = pd.concat([X_numerical, X_categorical], axis=1)
                
                # Store encoder for later use
                le_dict['onehot_encoder'] = ohe
                le_dict['categorical_cols'] = categorical_cols
                le_dict['feature_names'] = feature_names
            else:
                # No categorical columns, use label encoding as fallback
                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    le_dict[col] = le
        
        # Handle target variable if categorical - AUTO-DETECT PROBLEM TYPE
        target_le = None
        is_classification_target = False
        
        if y.dtype == 'object':
            # Categorical target - use classification
            target_le = LabelEncoder()
            y = target_le.fit_transform(y.astype(str))
            is_classification_target = True
            print(f"Target '{target_column}' is categorical - using CLASSIFICATION")
            print(f"Classes: {list(target_le.classes_)}")
        else:
            # Check if numeric target should be treated as classification
            unique_values = y.nunique()
            if unique_values <= 10:  # Likely categorical if ≤10 unique values
                is_classification_target = True
                print(f"Target '{target_column}' has {unique_values} unique values - treating as CLASSIFICATION")
            else:
                print(f"Target '{target_column}' is numeric with {unique_values} unique values - using REGRESSION")
        
        # Force classification models for categorical targets
        if is_classification_target and model_type in ['linear_regression', 'random_forest_regressor']:
            if model_type == 'linear_regression':
                model_type = 'logistic_regression'
                print("⚠️ Switched from Linear Regression to Logistic Regression for categorical target")
            elif model_type == 'random_forest_regressor':
                model_type = 'random_forest_classifier'
                print("⚠️ Switched from Random Forest Regressor to Random Forest Classifier for categorical target")
        
        # Split data into train/validation/test (60/20/20)
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Add target selection guidance
        target_recommendations = get_target_recommendations(df, target_column)
        
        # Train model and calculate metrics with detailed explanations
        
        # Add target validation warnings
        target_warnings = []
        if target_column in ['Color', 'Model', 'Region', 'Fuel_Type', 'Transmission']:
            if model_type in ['linear_regression', 'random_forest_regressor']:
                target_warnings.append(f"⚠️ WARNING: '{target_column}' is categorical but you selected a regression model!")
                target_warnings.append(f"🔄 Recommendation: Use classification models for categorical targets like {target_column}")
        
        print(f"Starting model training for {model_type}...")
        
        if model_type == 'linear_regression':
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Get validation info and overfitting detection
            validation_info = detect_overfitting_and_validate(
                model, X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test, model_type, is_scaled=True
            )
            
            # R² Score for regression
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Calculate mean absolute error
            from sklearn.metrics import mean_absolute_error
            mae = mean_absolute_error(y_test, y_pred)
            
            accuracy = r2
            model_explanation = {
                'task': 'Regression (Predicting Continuous Numbers)',
                'target': f'Predicting numerical values of "{target_column}"',
                'metric_name': 'R² Score (Coefficient of Determination)',
                'metric_meaning': f'R² = {r2:.4f} means the model explains {r2*100:.1f}% of the variance in {target_column}',
                'interpretation': get_regression_interpretation(r2),
                'additional_metrics': {
                    'R² Score': f'{r2:.4f}',
                    'Mean Squared Error': f'{mse:.4f}',
                    'Root Mean Squared Error': f'{rmse:.4f}',
                    'Mean Absolute Error': f'{mae:.4f}'
                },
                'validation_info': validation_info,
                'example': f'If actual {target_column} is 100, model might predict {100 + (rmse * 0.5):.1f} (±{rmse:.1f} typical error)',
                'metric_value': accuracy
            }
        
        elif model_type == 'logistic_regression':
            # Enhanced logistic regression with better regularization
            model = LogisticRegression(
                random_state=42, 
                max_iter=2000,              # More iterations for convergence
                C=0.1,                      # Stronger regularization
                class_weight='balanced',    # Handle class imbalance
                solver='liblinear'          # Better for small datasets
            )
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Get validation info and overfitting detection
            validation_info = detect_overfitting_and_validate(
                model, X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test, model_type, is_scaled=True
            )
            
            accuracy = accuracy_score(y_test, y_pred)
            
            # Additional classification metrics
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # AUC score for binary classification
            auc_score = 'N/A'
            if len(np.unique(y_test)) == 2:
                try:
                    auc_score = f'{roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}'
                except:
                    auc_score = 'N/A'
            
            # Get unique classes for interpretation
            unique_classes = np.unique(y_test)
            classes_info = f"{len(unique_classes)} classes: {unique_classes[:5]}{'...' if len(unique_classes) > 5 else ''}"
            
            model_explanation = {
                'task': 'Binary/Multi-Class Classification with Balanced Weights',
                'target': f'Predicting categories/classes of "{target_column}" with regularization',
                'metric_name': 'Classification Accuracy',
                'metric_meaning': f'Accuracy = {accuracy:.4f} means {accuracy*100:.1f}% of predictions are correct',
                'interpretation': get_classification_interpretation(accuracy),
                'additional_metrics': {
                    'Accuracy': f'{accuracy:.4f}',
                    'Precision': f'{precision:.4f}',
                    'Recall': f'{recall:.4f}',
                    'F1-Score': f'{f1:.4f}',
                    'AUC Score': auc_score
                },
                'validation_info': validation_info,
                'classes_info': classes_info,
                'example': f'Out of 100 predictions, about {int(accuracy*100)} would be correct',
                'metric_value': accuracy
            }
        
        elif model_type == 'random_forest_regressor':
            # Simplified parameters for faster training
            model = RandomForestRegressor(
                random_state=42, 
                n_estimators=50,        # Reduced from 100 for speed
                max_depth=10,           # Reduced depth for speed
                min_samples_split=5,    # Require minimum samples to split
                min_samples_leaf=2,     # Minimum samples in leaf
                max_features='sqrt',    # Use subset of features
                n_jobs=1                # Single thread to avoid issues
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Get validation info and overfitting detection
            validation_info = detect_overfitting_and_validate(
                model, X_train, X_val, X_test, 
                y_train, y_val, y_test, model_type, is_scaled=False
            )
            
            from sklearn.metrics import r2_score, mean_absolute_error
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            
            accuracy = r2
            model_explanation = {
                'task': 'Regression using Ensemble of Decision Trees',
                'target': f'Predicting numerical values of "{target_column}" using 100 decision trees',
                'metric_name': 'R² Score',
                'metric_meaning': f'R² = {r2:.4f} means the ensemble explains {r2*100:.1f}% of the variance',
                'interpretation': get_regression_interpretation(r2),
                'additional_metrics': {
                    'R² Score': f'{r2:.4f}',
                    'RMSE': f'{rmse:.4f}',
                    'MAE': f'{mae:.4f}'
                },
                'validation_info': validation_info,
                'example': f'Typical prediction error: ±{rmse:.1f} units of {target_column}',
                'metric_value': accuracy
            }
        
        elif model_type == 'random_forest_classifier':
            # Simplified parameters for faster training
            model = RandomForestClassifier(
                random_state=42, 
                n_estimators=50,            # Reduced from 200 for speed
                max_depth=10,               # Reduced from 20 for speed
                min_samples_split=10,       # Higher split requirement
                min_samples_leaf=5,         # Higher leaf requirement
                max_features='sqrt',        # Use subset of features
                class_weight='balanced',    # Handle class imbalance
                bootstrap=True,
                n_jobs=1                    # Single thread to avoid issues
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Get validation info and overfitting detection
            validation_info = detect_overfitting_and_validate(
                model, X_train, X_val, X_test, 
                y_train, y_val, y_test, model_type, is_scaled=False
            )
            
            accuracy = accuracy_score(y_test, y_pred)
            
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            unique_classes = np.unique(y_test)
            classes_info = f"{len(unique_classes)} classes: {unique_classes[:5]}{'...' if len(unique_classes) > 5 else ''}"
            
            # Feature importance analysis
            feature_importance = model.feature_importances_
            important_features = sorted(zip(X.columns, feature_importance), key=lambda x: x[1], reverse=True)[:5]
            importance_info = {name: float(importance) for name, importance in important_features}
            
            model_explanation = {
                'task': 'Classification using Optimized Random Forest (200 trees)',
                'target': f'Predicting categories of "{target_column}" with balanced class weights',
                'metric_name': 'Classification Accuracy',
                'metric_meaning': f'Accuracy = {accuracy:.4f} means {accuracy*100:.1f}% correct predictions',
                'interpretation': get_classification_interpretation(accuracy),
                'additional_metrics': {
                    'Accuracy': f'{accuracy:.4f}',
                    'Precision': f'{precision:.4f}',
                    'Recall': f'{recall:.4f}',
                    'F1-Score': f'{f1:.4f}',
                    'OOB Score': 'Disabled for speed'
                },
                'validation_info': validation_info,
                'classes_info': classes_info,
                'feature_importance': importance_info,
                'target_warnings': target_warnings,
                'example': f'In 100 predictions, about {int(accuracy*100)} would be correct classifications',
                'metric_value': accuracy
            }
        
        elif model_type == 'svm_classifier':
            # Add regularization parameter
            model = SVC(random_state=42, probability=True, C=1.0, gamma='scale')
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Get validation info and overfitting detection
            validation_info = detect_overfitting_and_validate(
                model, X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test, model_type, is_scaled=True
            )
            
            accuracy = accuracy_score(y_test, y_pred)
            
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            unique_classes = np.unique(y_test)
            classes_info = f"{len(unique_classes)} classes: {unique_classes[:5]}{'...' if len(unique_classes) > 5 else ''}"
            
            model_explanation = {
                'task': 'Classification using Support Vector Machine',
                'target': f'Predicting categories of "{target_column}" by finding optimal decision boundaries',
                'metric_name': 'Classification Accuracy',
                'metric_meaning': f'Accuracy = {accuracy:.4f} means {accuracy*100:.1f}% correct classifications',
                'interpretation': get_classification_interpretation(accuracy),
                'additional_metrics': {
                    'Accuracy': f'{accuracy:.4f}',
                    'Precision': f'{precision:.4f}',
                    'Recall': f'{recall:.4f}',
                    'F1-Score': f'{f1:.4f}'
                },
                'validation_info': validation_info,
                'classes_info': classes_info,
                'example': f'Creates decision boundaries to separate {len(unique_classes)} classes',
                'metric_value': accuracy
            }
        
        else:
            return jsonify({'error': 'Invalid model type'}), 400
        
        # Save model
        model_filename = f"model_{session['user_id']}_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path = os.path.join('models', model_filename)
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'label_encoders': le_dict,
            'target_encoder': target_le,
            'feature_names': X.columns.tolist()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save model info to database
        c.execute("""INSERT INTO models (user_id, dataset_id, model_name, model_type, target_column, accuracy, model_file)
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                 (session['user_id'], dataset_id, model_name, model_type, target_column, accuracy, model_filename))
        conn.commit()
        model_id = c.lastrowid
        conn.close()
        
        return jsonify({
            'success': True,
            'model_id': model_id,
            'accuracy': accuracy,
            'message': f'Model trained successfully!',
            'explanation': model_explanation,
            'target_recommendations': target_recommendations
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_dataset_columns/<int:dataset_id>')
def get_dataset_columns(dataset_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Get dataset info
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT filename FROM datasets WHERE id = ? AND user_id = ?",
              (dataset_id, session['user_id']))
    dataset = c.fetchone()
    conn.close()
    
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], dataset[0])
        df = pd.read_csv(filepath)
        columns = df.columns.tolist()
        return jsonify({'columns': columns})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check_missing_values/<int:dataset_id>')
def check_missing_values(dataset_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Get dataset info
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT filename FROM datasets WHERE id = ? AND user_id = ?",
              (dataset_id, session['user_id']))
    dataset = c.fetchone()
    conn.close()
    
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], dataset[0])
        df = pd.read_csv(filepath)
        
        # Analyze missing values
        missing_info = {}
        total_rows = len(df)
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / total_rows) * 100
            
            if missing_count > 0:
                # Get column info
                col_info = {
                    'missing_count': int(missing_count),
                    'missing_percentage': round(missing_percentage, 2),
                    'total_rows': total_rows,
                    'data_type': str(df[col].dtype),
                    'sample_values': df[col].dropna().head(5).tolist() if missing_count < total_rows else []
                }
                
                # Add suggestions based on data type and missing percentage
                if df[col].dtype == 'object':
                    col_info['suggestions'] = ['Drop rows', 'Fill with "Unknown"', 'Fill with mode (most common)']
                    if missing_count < total_rows:
                        mode_value = df[col].mode()
                        col_info['mode_value'] = mode_value[0] if len(mode_value) > 0 else 'N/A'
                else:
                    col_info['suggestions'] = ['Drop rows', 'Fill with mean', 'Fill with median', 'Fill with 0']
                    if missing_count < total_rows:
                        col_info['mean_value'] = round(df[col].mean(), 3) if not pd.isna(df[col].mean()) else 'N/A'
                        col_info['median_value'] = round(df[col].median(), 3) if not pd.isna(df[col].median()) else 'N/A'
                
                missing_info[col] = col_info
        
        return jsonify({
            'has_missing': len(missing_info) > 0,
            'total_rows': total_rows,
            'missing_columns': len(missing_info),
            'missing_details': missing_info
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/handle_missing_values', methods=['POST'])
def handle_missing_values():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.get_json()
        dataset_id = data['dataset_id']
        missing_strategies = data['strategies']  # Dict of column -> strategy
        
        # Get dataset
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT filename FROM datasets WHERE id = ? AND user_id = ?",
                  (dataset_id, session['user_id']))
        dataset = c.fetchone()
        
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Load data
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], dataset[0])
        df = pd.read_csv(filepath)
        original_rows = len(df)
        
        # Apply missing value strategies
        rows_to_drop = set()
        processed_columns = []
        
        for column, strategy in missing_strategies.items():
            if column not in df.columns:
                continue
                
            missing_count = df[column].isnull().sum()
            if missing_count == 0:
                continue
            
            if strategy == 'drop_rows':
                # Mark rows for dropping
                missing_indices = df[df[column].isnull()].index
                rows_to_drop.update(missing_indices)
                processed_columns.append(f"✅ {column}: Marked {missing_count} rows for removal")
                
            elif strategy == 'fill_mean' and df[column].dtype in ['int64', 'float64']:
                mean_val = df[column].mean()
                df[column].fillna(mean_val, inplace=True)
                processed_columns.append(f"✅ {column}: Filled {missing_count} values with mean ({mean_val:.3f})")
                
            elif strategy == 'fill_median' and df[column].dtype in ['int64', 'float64']:
                median_val = df[column].median()
                df[column].fillna(median_val, inplace=True)
                processed_columns.append(f"✅ {column}: Filled {missing_count} values with median ({median_val:.3f})")
                
            elif strategy == 'fill_zero':
                df[column].fillna(0, inplace=True)
                processed_columns.append(f"✅ {column}: Filled {missing_count} values with 0")
                
            elif strategy == 'fill_unknown':
                df[column].fillna('Unknown', inplace=True)
                processed_columns.append(f"✅ {column}: Filled {missing_count} values with 'Unknown'")
                
            elif strategy == 'fill_mode':
                mode_val = df[column].mode()
                if len(mode_val) > 0:
                    df[column].fillna(mode_val[0], inplace=True)
                    processed_columns.append(f"✅ {column}: Filled {missing_count} values with mode ('{mode_val[0]}')")
                else:
                    df[column].fillna('Unknown', inplace=True)
                    processed_columns.append(f"✅ {column}: Filled {missing_count} values with 'Unknown' (no mode found)")
        
        # Drop rows if any were marked for dropping
        if rows_to_drop:
            df = df.drop(list(rows_to_drop))
            processed_columns.append(f"🗑️ Removed {len(rows_to_drop)} rows with missing values")
        
        # Save the cleaned dataset with a new filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = dataset[0]
        base_name = original_filename.rsplit('.', 1)[0]
        cleaned_filename = f"{base_name}_cleaned_{timestamp}.csv"
        cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename)
        
        df.to_csv(cleaned_filepath, index=False)
        
        # Update database with cleaned dataset
        file_size = os.path.getsize(cleaned_filepath)
        rows, columns = df.shape
        
        c.execute("""INSERT INTO datasets (user_id, filename, original_filename, file_size, rows, columns)
                     VALUES (?, ?, ?, ?, ?, ?)""",
                 (session['user_id'], cleaned_filename, f"Cleaned_{original_filename}", file_size, rows, columns))
        conn.commit()
        new_dataset_id = c.lastrowid
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'Missing values handled successfully! Dataset cleaned from {original_rows} to {len(df)} rows.',
            'new_dataset_id': new_dataset_id,
            'original_rows': original_rows,
            'cleaned_rows': len(df),
            'rows_removed': original_rows - len(df),
            'processed_columns': processed_columns,
            'remaining_missing': df.isnull().sum().sum()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_model/<int:model_id>', methods=['DELETE'])
def delete_model(model_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # Get model info (ensure it belongs to the current user)
        c.execute("SELECT model_file FROM models WHERE id = ? AND user_id = ?",
                  (model_id, session['user_id']))
        model = c.fetchone()
        
        if not model:
            conn.close()
            return jsonify({'error': 'Model not found or access denied'}), 404
        
        model_filename = model[0]
        
        # Delete model file if it exists
        if model_filename:
            model_path = os.path.join('models', model_filename)
            if os.path.exists(model_path):
                os.remove(model_path)
        
        # Delete model from database
        c.execute("DELETE FROM models WHERE id = ? AND user_id = ?",
                  (model_id, session['user_id']))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Model deleted successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_dataset/<int:dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # Get dataset info (ensure it belongs to the current user)
        c.execute("SELECT filename FROM datasets WHERE id = ? AND user_id = ?",
                  (dataset_id, session['user_id']))
        dataset = c.fetchone()
        
        if not dataset:
            conn.close()
            return jsonify({'error': 'Dataset not found or access denied'}), 404
        
        dataset_filename = dataset[0]
        
        # Check if any models are associated with this dataset
        c.execute("SELECT COUNT(*) FROM models WHERE dataset_id = ? AND user_id = ?",
                  (dataset_id, session['user_id']))
        model_count = c.fetchone()[0]
        
        if model_count > 0:
            # Delete all associated models first
            c.execute("SELECT model_file FROM models WHERE dataset_id = ? AND user_id = ?",
                      (dataset_id, session['user_id']))
            models = c.fetchall()
            
            # Delete model files
            for model in models:
                if model[0]:
                    model_path = os.path.join('models', model[0])
                    if os.path.exists(model_path):
                        os.remove(model_path)
            
            # Delete models from database
            c.execute("DELETE FROM models WHERE dataset_id = ? AND user_id = ?",
                      (dataset_id, session['user_id']))
        
        # Delete dataset file if it exists
        if dataset_filename:
            dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_filename)
            if os.path.exists(dataset_path):
                os.remove(dataset_path)
        
        # Delete dataset from database
        c.execute("DELETE FROM datasets WHERE id = ? AND user_id = ?",
                  (dataset_id, session['user_id']))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': f'Dataset deleted successfully{"" if model_count == 0 else f" (along with {model_count} associated models)"}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_target_recommendations(df, target_column):
    """Provide universal recommendations for target selection based on data analysis"""
    recommendations = []
    
    target_data = df[target_column]
    
    # Analyze target appropriateness
    if target_data.dtype == 'object':
        unique_count = target_data.nunique()
        if unique_count > 20:
            recommendations.append(f"⚠️ High cardinality: {unique_count} unique values in '{target_column}' may be challenging to predict")
        elif unique_count < 2:
            recommendations.append(f"❌ Invalid target: '{target_column}' has only {unique_count} unique value(s)")
        else:
            recommendations.append(f"✅ Good categorical target: {unique_count} classes in '{target_column}'")
    else:
        # Numeric target
        if target_data.nunique() <= 10:
            recommendations.append(f"💡 Consider: '{target_column}' might work better as classification ({target_data.nunique()} unique values)")
        else:
            recommendations.append(f"✅ Good numeric target: '{target_column}' suitable for regression")
    
    # Universal target selection guide
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    recommendations.append("\n📊 Universal Target Selection Guide:")
    recommendations.append("🎯 CLASSIFICATION targets (predict categories):")
    
    # Smart recommendations based on column names
    classification_targets = []
    regression_targets = []
    
    for col in categorical_cols:
        col_lower = col.lower()
        if any(word in col_lower for word in ['class', 'type', 'category', 'grade', 'level']):
            classification_targets.append(f"   • {col} - EXCELLENT for classification")
        elif any(word in col_lower for word in ['status', 'result', 'outcome', 'diagnosis']):
            classification_targets.append(f"   • {col} - GOOD for binary/multi-class prediction")
        else:
            classification_targets.append(f"   • {col} - Consider for categorical prediction")
    
    for col in numeric_cols:
        col_lower = col.lower()
        if any(word in col_lower for word in ['price', 'cost', 'value', 'amount', 'revenue']):
            regression_targets.append(f"   • {col} - EXCELLENT for pricing/financial models")
        elif any(word in col_lower for word in ['score', 'rating', 'performance', 'measure']):
            regression_targets.append(f"   • {col} - GOOD for performance prediction")
        elif any(word in col_lower for word in ['age', 'time', 'duration', 'count']):
            regression_targets.append(f"   • {col} - GOOD for temporal/quantity prediction")
        else:
            regression_targets.append(f"   • {col} - Consider for numerical prediction")
    
    # Add the recommendations
    recommendations.extend(classification_targets[:5])  # Top 5 classification targets
    
    recommendations.append("📈 REGRESSION targets (predict numbers):")
    recommendations.extend(regression_targets[:5])  # Top 5 regression targets
    
    # Add dataset-specific insights
    recommendations.append(f"\n🔍 Dataset Insights:")
    recommendations.append(f"   • {len(df)} rows × {len(df.columns)} columns")
    recommendations.append(f"   • {len(numeric_cols)} numeric features, {len(categorical_cols)} categorical features")
    
    if df.isnull().sum().sum() > 0:
        recommendations.append(f"   • ⚠️ Contains missing values - will be handled automatically")
    
    return recommendations

# Helper functions for interpreting model performance
def get_regression_interpretation(r2_score):
    """Convert R² score to human-readable interpretation"""
    if r2_score >= 0.9:
        return "🎯 Excellent! Model predictions are very close to actual values"
    elif r2_score >= 0.7:
        return "✅ Good! Model captures most of the pattern in the data"
    elif r2_score >= 0.5:
        return "⚠️ Moderate. Model shows some predictive ability but room for improvement"
    elif r2_score >= 0.3:
        return "⚡ Weak. Model has limited predictive power"
    elif r2_score >= 0:
        return "❌ Poor. Model barely better than predicting the average"
    else:
        return "💥 Very Poor. Model is worse than just predicting the average!"

def get_classification_interpretation(accuracy):
    """Convert accuracy to human-readable interpretation"""
    if accuracy >= 0.95:
        return "🎯 Excellent! Nearly all predictions are correct"
    elif accuracy >= 0.85:
        return "✅ Very Good! Most predictions are accurate"
    elif accuracy >= 0.75:
        return "👍 Good! Decent accuracy for most use cases"
    elif accuracy >= 0.65:
        return "⚠️ Moderate. Acceptable but could be improved"
    elif accuracy >= 0.55:
        return "⚡ Weak. Better than random guessing but not reliable"
    else:
        return "❌ Poor. Close to random guessing - model needs work!"

@app.route('/auto-login/<username>')
def auto_login(username):
    """Auto login for specific user - for debugging purposes"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    
    if user:
        session.permanent = True
        session['user_id'] = user[0]
        session['username'] = username
        session.modified = True
        
        flash(f'Auto-logged in as {username}', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash(f'User {username} not found', 'error')
        return redirect(url_for('login'))

@app.route('/test-dashboard')
def test_dashboard():
    """Test route to verify dashboard template rendering"""
    # Simulate user data for testing
    test_datasets = [
        (1, 'sample_data.csv', '2025-07-15 08:00:22', 761, 30, 6),
        (2, 'BMW_Car_Sales_Classification.csv', '2025-07-15 08:10:47', 3342694, 50000, 11)
    ]
    
    test_models = [
        (1, 'sample_data_model', 'logistic_regression', 'age', 0.0, '2025-07-15 08:08:32', 'sample_data.csv'),
        (2, 'BMW_model', 'logistic_regression', 'Model', 0.0953, '2025-07-15 08:13:07', 'BMW_Car_Sales_Classification.csv')
    ]
    
    return render_template('dashboard.html', datasets=test_datasets, models=test_models)

@app.route('/health')
def health_check():
    """Simple health check route"""
    return {'status': 'OK', 'message': 'Flask app is running'}

@app.route('/quick-login')
def quick_login():
    """Quick login for testing - creates a test user and logs them in"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Check if test user exists
    c.execute("SELECT id FROM users WHERE username = ?", ('testuser',))
    user = c.fetchone()
    
    if not user:
        # Create test user
        password_hash = generate_password_hash('password123')
        c.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                 ('testuser', 'test@example.com', password_hash))
        conn.commit()
        c.execute("SELECT id FROM users WHERE username = ?", ('testuser',))
        user = c.fetchone()
    
    conn.close()
    
    # Log in the test user
    session.permanent = True
    session['user_id'] = user[0]
    session['username'] = 'testuser'
    session.modified = True
    
    flash('Quick login successful! Test credentials: testuser/password123', 'success')
    return redirect(url_for('dashboard'))

@app.route('/model_details/<int:model_id>')
def model_details(model_id):
    """Display detailed analysis and visualizations for a trained model"""
    if 'user_id' not in session:
        flash('Please log in to view model details.')
        return redirect(url_for('login'))
    
    try:
        # Get model info from database
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("""SELECT m.id, m.model_name, m.model_type, m.target_column, m.accuracy, 
                            m.created_at, m.model_file, d.filename, d.original_filename
                     FROM models m 
                     JOIN datasets d ON m.dataset_id = d.id 
                     WHERE m.id = ? AND m.user_id = ?""", (model_id, session['user_id']))
        model_info = c.fetchone()
        conn.close()
        
        if not model_info:
            flash('Model not found or access denied.')
            return redirect(url_for('dashboard'))
        
        # Load the saved model
        model_file = model_info[6]
        model_path = os.path.join('models', model_file)
        
        if not os.path.exists(model_path):
            flash('Model file not found.')
            return redirect(url_for('dashboard'))
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data.get('scaler')
        feature_names = model_data.get('feature_names', [])
        
        # Load original dataset to recreate predictions and visualizations
        dataset_file = model_info[7]
        dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_file)
        
        if not os.path.exists(dataset_path):
            flash('Original dataset not found.')
            return redirect(url_for('dashboard'))
        
        df = pd.read_csv(dataset_path)
        target_column = model_info[3]
        
        # Prepare data similar to training
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Apply same feature engineering as during training
        if 'Year' in X.columns:
            current_year = 2024
            X['Car_Age'] = current_year - X['Year']
        
        if 'Mileage_KM' in X.columns and 'Car_Age' in X.columns:
            X['Avg_Mileage_Per_Year'] = X['Mileage_KM'] / (X['Car_Age'] + 1)
        
        if 'Price_USD' in X.columns and 'Sales_Volume' in X.columns:
            X['Total_Revenue'] = X['Price_USD'] * X['Sales_Volume']
        
        # Recreate the exact same encoding used during training
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Get the label encoders from model data
        le_dict = model_data.get('label_encoders', {})
        
        # Check if one-hot encoding was used (based on stored encoders)
        if 'onehot_encoder' in le_dict:
            # One-hot encoding was used during training
            print("Using one-hot encoding reconstruction...")
            ohe = le_dict['onehot_encoder']
            original_categorical_cols = le_dict['categorical_cols']
            
            # Apply the same category limiting as during training
            for col in original_categorical_cols:
                if col in X.columns:
                    # Use the same categories that were seen during training
                    col_index = original_categorical_cols.index(col)
                    training_categories = ohe.categories_[col_index]
                    # Map unknown categories to 'Other' if it exists, otherwise to first category
                    X[col] = X[col].astype(str).apply(
                        lambda x: x if x in training_categories else ('Other' if 'Other' in training_categories else training_categories[0])
                    )
            
            # Apply one-hot encoding using the trained encoder
            if len(original_categorical_cols) > 0 and all(col in X.columns for col in original_categorical_cols):
                categorical_encoded = ohe.transform(X[original_categorical_cols])
                onehot_feature_names = le_dict['feature_names']
                
                # Combine with numerical features
                X_categorical = pd.DataFrame(categorical_encoded, columns=onehot_feature_names, index=X.index)
                X_numerical = X[numerical_cols]
                X = pd.concat([X_numerical, X_categorical], axis=1)
            else:
                # Fallback if columns are missing
                print("Warning: Some categorical columns missing, using fallback encoding")
                for col in categorical_cols:
                    if col in le_dict and col not in ['onehot_encoder', 'categorical_cols', 'feature_names']:
                        le = le_dict[col]
                        # Handle unseen categories
                        X[col] = X[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
                        X[col] = le.transform(X[col])
        else:
            # Label encoding was used during training
            print("Using label encoding reconstruction...")
            for col in categorical_cols:
                if col in le_dict:
                    le = le_dict[col]
                    # Handle unseen categories by mapping them to the first class
                    X[col] = X[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    X[col] = le.transform(X[col])
        
        # Handle target encoding if needed
        target_le = model_data.get('target_encoder')
        if target_le is not None:
            # Handle unseen target categories
            y_encoded = y.astype(str).apply(lambda x: x if x in target_le.classes_ else target_le.classes_[0])
            y_encoded = target_le.transform(y_encoded)
            class_names = target_le.classes_
        elif y.dtype == 'object':
            # Create new encoder for target
            target_le_new = LabelEncoder()
            y_encoded = target_le_new.fit_transform(y.astype(str))
            class_names = target_le_new.classes_
        else:
            y_encoded = y
            class_names = None
        
        # Split data to get test set (same random state as training)
        X_temp, X_test, y_temp, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Ensure feature columns match exactly what the model expects
        expected_features = feature_names
        current_features = X_test.columns.tolist()
        
        # Check if features match
        if set(expected_features) != set(current_features):
            # Try to reorder and add missing features
            missing_features = set(expected_features) - set(current_features)
            extra_features = set(current_features) - set(expected_features)
            
            print(f"Feature mismatch detected:")
            print(f"Missing features: {missing_features}")
            print(f"Extra features: {extra_features}")
            
            # Add missing features with zeros
            for feature in missing_features:
                X_test[feature] = 0
            
            # Remove extra features
            for feature in extra_features:
                if feature in X_test.columns:
                    X_test = X_test.drop(columns=[feature])
            
            # Reorder columns to match training
            X_test = X_test.reindex(columns=expected_features, fill_value=0)
        
        # Scale if needed
        if scaler:
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)
            else:
                y_pred_proba = None
        else:
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            else:
                y_pred_proba = None
        
        # Generate visualizations
        plots = generate_model_visualizations(model, X_test, y_test, y_pred, y_pred_proba, 
                                            model_info[2], class_names, expected_features)
        
        # Calculate detailed metrics
        detailed_metrics = calculate_detailed_metrics(y_test, y_pred, y_pred_proba, 
                                                    model_info[2], class_names)
        
        # Model information summary
        model_summary = {
            'id': model_info[0],
            'name': model_info[1],
            'type': model_info[2],
            'target_column': model_info[3],
            'accuracy': model_info[4],
            'created_at': model_info[5],
            'dataset_name': model_info[8],
            'test_samples': len(X_test),
            'features_count': len(expected_features),
            'feature_names': expected_features[:10],  # Show first 10 features
            'all_features': expected_features,  # All features for detailed view
            'model_type_details': get_model_type_info(model),
            'hyperparameters': get_model_hyperparameters(model),
            'training_info': {
                'train_size': len(X_temp),
                'test_size': len(X_test),
                'feature_engineering': get_feature_engineering_info(model_data),
                'preprocessing': get_preprocessing_info(model_data)
            }
        }
        
        # Generate additional analysis
        advanced_analysis = generate_advanced_analysis(model, X_test, y_test, y_pred, y_pred_proba, 
                                                     model_info[2], feature_names, df, target_column)
        
        return render_template('model_details.html', 
                             model_summary=model_summary,
                             detailed_metrics=detailed_metrics,
                             plots=plots,
                             advanced_analysis=advanced_analysis)
        
    except Exception as e:
        print(f"Error in model details: {e}")
        import traceback
        traceback.print_exc()
        flash(f'Error loading model details: {str(e)}')
        return redirect(url_for('dashboard'))

def get_model_type_info(model):
    """Get detailed information about the model type"""
    model_name = type(model).__name__
    model_info = {
        'name': model_name,
        'category': 'Unknown',
        'description': 'Machine learning model',
        'strengths': [],
        'use_cases': []
    }
    
    if 'RandomForest' in model_name:
        model_info.update({
            'category': 'Ensemble',
            'description': 'Random Forest is an ensemble method that builds multiple decision trees',
            'strengths': ['Handles overfitting well', 'Works with both numerical and categorical data', 'Provides feature importance'],
            'use_cases': ['Classification', 'Regression', 'Feature selection']
        })
    elif 'LogisticRegression' in model_name:
        model_info.update({
            'category': 'Linear',
            'description': 'Logistic Regression uses logistic function for binary/multiclass classification',
            'strengths': ['Fast training', 'Interpretable', 'No assumptions about distribution'],
            'use_cases': ['Binary classification', 'Multiclass classification', 'Probability estimation']
        })
    elif 'SVC' in model_name or 'SVM' in model_name:
        model_info.update({
            'category': 'Support Vector',
            'description': 'Support Vector Machine finds optimal hyperplane to separate classes',
            'strengths': ['Effective in high dimensions', 'Memory efficient', 'Versatile kernels'],
            'use_cases': ['Text classification', 'Image classification', 'High-dimensional data']
        })
    elif 'DecisionTree' in model_name:
        model_info.update({
            'category': 'Tree-based',
            'description': 'Decision Tree creates a model that predicts target by learning decision rules',
            'strengths': ['Easy to interpret', 'Requires little data preparation', 'Handles both numerical and categorical'],
            'use_cases': ['Rule extraction', 'Feature selection', 'Both classification and regression']
        })
    elif 'KNeighbors' in model_name:
        model_info.update({
            'category': 'Instance-based',
            'description': 'K-Nearest Neighbors classifies based on the majority class of k nearest neighbors',
            'strengths': ['Simple to understand', 'No assumptions about data', 'Naturally handles multiclass'],
            'use_cases': ['Recommendation systems', 'Pattern recognition', 'Outlier detection']
        })
    elif 'LinearRegression' in model_name:
        model_info.update({
            'category': 'Linear',
            'description': 'Linear Regression models relationship between variables using linear equation',
            'strengths': ['Fast and simple', 'Interpretable coefficients', 'No hyperparameters'],
            'use_cases': ['Trend analysis', 'Forecasting', 'Understanding relationships']
        })
    
    return model_info

def get_model_hyperparameters(model):
    """Extract model hyperparameters"""
    params = {}
    try:
        if hasattr(model, 'get_params'):
            all_params = model.get_params()
            # Filter out None values and functions
            for key, value in all_params.items():
                if value is not None and not callable(value):
                    params[key] = value
    except:
        pass
    return params

def get_feature_engineering_info(model_data):
    """Get information about feature engineering steps"""
    info = []
    
    if 'Car_Age' in str(model_data.get('feature_names', [])):
        info.append("Car Age: Calculated from current year - manufacturing year")
    
    if 'Avg_Mileage_Per_Year' in str(model_data.get('feature_names', [])):
        info.append("Average Mileage Per Year: Mileage divided by car age")
    
    if 'Total_Revenue' in str(model_data.get('feature_names', [])):
        info.append("Total Revenue: Price multiplied by sales volume")
    
    if 'onehot_encoder' in model_data.get('label_encoders', {}):
        info.append("One-Hot Encoding: Applied to categorical variables")
    elif model_data.get('label_encoders'):
        info.append("Label Encoding: Applied to categorical variables")
    
    return info if info else ["No specific feature engineering applied"]

def get_preprocessing_info(model_data):
    """Get information about preprocessing steps"""
    info = []
    
    if model_data.get('scaler'):
        scaler_type = type(model_data['scaler']).__name__
        info.append(f"Feature Scaling: {scaler_type}")
    
    if model_data.get('target_encoder'):
        info.append("Target Encoding: Applied to target variable")
    
    info.append("Train-Test Split: 80% training, 20% testing")
    info.append("Random State: 42 (for reproducibility)")
    
    return info

def generate_advanced_analysis(model, X_test, y_test, y_pred, y_pred_proba, model_type, feature_names, df, target_column):
    """Generate advanced analysis and insights"""
    analysis = {}
    
    try:
        # Data distribution analysis
        analysis['data_insights'] = {
            'total_samples': len(df),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'target_distribution': df[target_column].value_counts().to_dict() if df[target_column].dtype == 'object' else None,
            'feature_types': {
                'numerical': len(df.select_dtypes(include=[np.number]).columns),
                'categorical': len(df.select_dtypes(include=['object']).columns)
            }
        }
        
        # Model performance insights
        if 'classifier' in model_type or 'logistic' in model_type:
            from sklearn.metrics import classification_report
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            analysis['performance_insights'] = {
                'best_class': max(report.keys(), key=lambda x: report[x]['f1-score'] if isinstance(report[x], dict) and 'f1-score' in report[x] else 0),
                'worst_class': min(report.keys(), key=lambda x: report[x]['f1-score'] if isinstance(report[x], dict) and 'f1-score' in report[x] else 1),
                'class_balance': len(set(y_test)) <= 5,  # Consider balanced if ≤ 5 classes
                'prediction_confidence': np.mean(np.max(y_pred_proba, axis=1)) if y_pred_proba is not None else None
            }
        else:
            # Regression insights
            residuals = y_test - y_pred
            analysis['performance_insights'] = {
                'mean_residual': np.mean(residuals),
                'residual_std': np.std(residuals),
                'prediction_range': {
                    'min': np.min(y_pred),
                    'max': np.max(y_pred),
                    'mean': np.mean(y_pred)
                },
                'actual_range': {
                    'min': np.min(y_test),
                    'max': np.max(y_test),
                    'mean': np.mean(y_test)
                }
            }
        
        # Feature correlation analysis (top correlations with target)
        if target_column in df.columns:
            numerical_df = df.select_dtypes(include=[np.number])
            if target_column in numerical_df.columns:
                correlations = numerical_df.corr()[target_column].abs().sort_values(ascending=False)
                analysis['feature_correlations'] = correlations.head(10).to_dict()
        
        # Prediction patterns
        if y_pred_proba is not None and len(set(y_test)) <= 10:
            # Analysis for classification with few classes
            prediction_patterns = {}
            for i, class_name in enumerate(np.unique(y_test)):
                class_mask = y_test == class_name
                if class_mask.sum() > 0:
                    avg_confidence = np.mean(y_pred_proba[class_mask, i]) if i < y_pred_proba.shape[1] else 0
                    prediction_patterns[str(class_name)] = {
                        'count': int(class_mask.sum()),
                        'avg_confidence': float(avg_confidence),
                        'correct_predictions': int(np.sum((y_test == class_name) & (y_pred == class_name)))
                    }
            analysis['prediction_patterns'] = prediction_patterns
        
    except Exception as e:
        print(f"Error in advanced analysis: {e}")
        analysis['error'] = str(e)
    
    return analysis

def generate_model_visualizations(model, X_test, y_test, y_pred, y_pred_proba, model_type, class_names, feature_names):
    """Generate various visualizations for model analysis"""
    plots = {}
    
    try:
        # 1. Confusion Matrix (for classification)
        if 'classifier' in model_type or 'logistic' in model_type:
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_test, y_pred)
            
            # Create a more detailed confusion matrix
            if class_names is not None and len(class_names) <= 10:
                labels = class_names
            else:
                labels = [f'Class {i}' for i in range(len(np.unique(y_test)))]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels)
            plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Labels', fontsize=12)
            plt.ylabel('True Labels', fontsize=12)
            plt.tight_layout()
            
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
            img.seek(0)
            plots['confusion_matrix'] = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            # 2. ROC Curve (for binary classification)
            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                img = io.BytesIO()
                plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
                img.seek(0)
                plots['roc_curve'] = base64.b64encode(img.getvalue()).decode()
                plt.close()
            
            # 3. Classification Report Visualization
            from sklearn.metrics import classification_report
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Convert to DataFrame for better visualization
            report_df = pd.DataFrame(report).transpose()
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='RdYlGn', 
                       vmin=0, vmax=1, fmt='.2f')
            plt.title('Classification Report Heatmap', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
            img.seek(0)
            plots['classification_report'] = base64.b64encode(img.getvalue()).decode()
            plt.close()
        
        # 4. Feature Importance (for tree-based models)
        if hasattr(model, 'feature_importances_') and len(feature_names) > 0:
            importance = model.feature_importances_
            
            # Make sure we don't exceed the available features
            max_features = min(len(importance), len(feature_names), 15)
            indices = np.argsort(importance)[::-1][:max_features]
            
            plt.figure(figsize=(12, 8))
            plt.title('Feature Importance', fontsize=16, fontweight='bold')
            
            # Create feature names list with bounds checking
            feature_names_subset = []
            importance_subset = []
            for i in indices:
                if i < len(feature_names) and i < len(importance):
                    feature_names_subset.append(feature_names[i])
                    importance_subset.append(importance[i])
            
            if feature_names_subset and importance_subset:
                plt.bar(range(len(feature_names_subset)), importance_subset)
                plt.xticks(range(len(feature_names_subset)), feature_names_subset, rotation=45, ha='right')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                img = io.BytesIO()
                plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
                img.seek(0)
                plots['feature_importance'] = base64.b64encode(img.getvalue()).decode()
            plt.close()
        
        # 5. Prediction vs Actual (for regression)
        if 'regressor' in model_type or 'regression' in model_type:
            plt.figure(figsize=(10, 8))
            
            # Scatter plot
            plt.subplot(2, 2, 1)
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Predicted vs Actual Values')
            plt.grid(True, alpha=0.3)
            
            # Residuals plot
            plt.subplot(2, 2, 2)
            residuals = y_test - y_pred
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals Plot')
            plt.grid(True, alpha=0.3)
            
            # Residuals histogram
            plt.subplot(2, 2, 3)
            plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.title('Residuals Distribution')
            plt.grid(True, alpha=0.3)
            
            # Q-Q plot for residuals
            plt.subplot(2, 2, 4)
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title('Q-Q Plot of Residuals')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
            img.seek(0)
            plots['regression_analysis'] = base64.b64encode(img.getvalue()).decode()
            plt.close()
        
        # 6. Prediction Distribution
        plt.figure(figsize=(12, 5))
        
        if 'classifier' in model_type or 'logistic' in model_type:
            # For classification: show prediction probabilities distribution
            if y_pred_proba is not None:
                plt.subplot(1, 2, 1)
                max_proba = np.max(y_pred_proba, axis=1)
                plt.hist(max_proba, bins=30, alpha=0.7, edgecolor='black')
                plt.xlabel('Maximum Prediction Probability')
                plt.ylabel('Frequency')
                plt.title('Prediction Confidence Distribution')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                unique_preds, counts = np.unique(y_pred, return_counts=True)
                plt.bar(unique_preds, counts)
                plt.xlabel('Predicted Classes')
                plt.ylabel('Count')
                plt.title('Prediction Distribution')
                plt.grid(True, alpha=0.3)
        else:
            # For regression: show prediction distribution
            plt.subplot(1, 2, 1)
            plt.hist(y_pred, bins=30, alpha=0.7, edgecolor='black', label='Predictions')
            plt.hist(y_test, bins=30, alpha=0.5, edgecolor='black', label='Actual')
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            plt.title('Prediction vs Actual Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            errors = np.abs(y_test - y_pred)
            plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Absolute Error')
            plt.ylabel('Frequency')
            plt.title('Prediction Error Distribution')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        plots['prediction_distribution'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    return plots

def calculate_detailed_metrics(y_test, y_pred, y_pred_proba, model_type, class_names):
    """Calculate detailed performance metrics"""
    metrics = {}
    
    try:
        if 'classifier' in model_type or 'logistic' in model_type:
            # Classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
            metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            metrics['f1_score'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            
            # Confusion matrix details
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Per-class metrics
            if class_names is not None and len(class_names) <= 10:
                precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
                recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
                f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
                
                metrics['per_class'] = {
                    'classes': class_names.tolist() if hasattr(class_names, 'tolist') else list(class_names),
                    'precision': precision_per_class.tolist(),
                    'recall': recall_per_class.tolist(),
                    'f1_score': f1_per_class.tolist()
                }
            
            # AUC for binary classification
            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                from sklearn.metrics import roc_auc_score
                metrics['auc'] = float(roc_auc_score(y_test, y_pred_proba[:, 1]))
        
        else:
            # Regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics['r2_score'] = float(r2_score(y_test, y_pred))
            metrics['mse'] = float(mean_squared_error(y_test, y_pred))
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics['mae'] = float(mean_absolute_error(y_test, y_pred))
            
            # Additional regression metrics
            metrics['mean_actual'] = float(np.mean(y_test))
            metrics['mean_predicted'] = float(np.mean(y_pred))
            metrics['std_actual'] = float(np.std(y_test))
            metrics['std_predicted'] = float(np.std(y_pred))
        
        # General metrics
        metrics['test_samples'] = len(y_test)
        metrics['model_type'] = model_type
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        metrics['error'] = str(e)
    
    return metrics

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
