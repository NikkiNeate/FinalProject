# CSV ML Analyzer

A comprehensive web application for CSV file analysis and machine learning model training. This application allows users to upload CSV files, perform data analysis, visualize data, and train various machine learning models with an intuitive web interface.

## Features

###  User Authentication
- User registration and login system
- Secure password hashing
- Session management

###  Data Analysis
- CSV file upload and validation
- Automatic data type detection
- Statistical summary generation
- Missing value analysis
- Data visualization with charts and graphs

###  Machine Learning
- Multiple ML model support:
  - Linear Regression
  - Logistic Regression
  - Random Forest (Classifier & Regressor)
  - Support Vector Machine (SVM)
- Automated data preprocessing
- Model training and evaluation
- Accuracy metrics and performance tracking

###  Dashboard
- User-friendly dashboard
- Dataset management
- Model tracking and comparison
- Interactive visualizations

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **download the project**
   

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your web browser and go to: `http://localhost:5000`

## Usage

### Getting Started
1. **Register** for a new account or **login** with existing credentials
2. **Upload** your CSV files using the dashboard
3. **Analyze** your data to understand its structure and characteristics
4. **Train** machine learning models on your data
5. **View** results and model performance metrics

### Supported File Formats
- CSV files (up to 16MB)
- Headers should be in the first row
- Mixed data types are supported (numeric, categorical, text)

### Model Types and Use Cases

| Model Type | Best For | Output Type |
|------------|----------|-------------|
| Linear Regression | Predicting continuous values with linear relationships | Continuous |
| Logistic Regression | Binary classification problems | Binary (0/1) |
| Random Forest Regressor | Complex continuous prediction problems | Continuous |
| Random Forest Classifier | Multi-class classification | Categorical |
| SVM Classifier | Classification with complex decision boundaries | Categorical |

## Project Structure

```
Final Project/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── users.db              # SQLite database (created automatically)
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   └── analyze.html
├── static/               # CSS and static files
│   └── style.css
├── uploads/              # User uploaded files (created automatically)
└── models/               # Trained model files (created automatically)
```

## Technology Stack

### Backend
- **Flask** - Web framework
- **SQLite** - Database
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **Matplotlib/Seaborn** - Data visualization

### Frontend
- **HTML5/CSS3** - Structure and styling
- **Bootstrap 5** - Responsive design
- **JavaScript** - Interactive functionality
- **Font Awesome** - Icons

## Security Features

- Password hashing using Werkzeug
- Session management
- File upload validation
- SQL injection prevention
- XSS protection

## Configuration

### Database
The application uses SQLite by default. The database file (`users.db`) will be created automatically when you first run the application.

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Activate your virtual environment if using one

2. **File upload errors**
   - Check file size (max 16MB)
   - Ensure file is in CSV format
   - Verify file permissions

3. **Model training fails**
   - Ensure target column is selected
   - Check for missing values in your data
   - Verify data types are appropriate for the model


## Future Enhancements

- [ ] Neural network models
- [ ] Advanced data preprocessing options
- [ ] Model comparison tools
- [ ] Data export functionality
- [ ] API endpoints for programmatic access
- [ ] Docker containerization
- [ ] Real-time model monitoring

---

**Built with ❤️ for data enthusiasts and machine learning practitioners**
