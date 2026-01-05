# Kickstarter Campaign Success Prediction

This project showcases my skills in machine learning, specifically classification and clustering techniques applied to real-world data.

## Project Overview

Built a machine learning model to predict whether Kickstarter campaigns will succeed or fail using historical campaign data from 200,000+ projects.

## What I Did

**Data Searching**
- https://webrobots.io/kickstarter-datasets/ found it here 

**Data Preprocessing**
- Loaded and cleaned 205,530 Kickstarter campaigns
- Filtered to successful/failed campaigns only (~186,000 records)
- Handled missing values and removed duplicates

**Feature Engineering**
- Created text features (blurb length)
- Extracted category information from JSON
- Built time-based features (campaign duration, prep time, launch timing)
- Transformed goal amounts (raw and log-transformed)
- Added media features (video presence)

**Models Trained**
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors

**Model Optimization**
- Used GridSearchCV for hyperparameter tuning
- Applied StandardScaler for feature normalization
- Performed train/test split validation

## Results

**Best Model: Random Forest Classifier**
- Test Accuracy: 80.3%
- Precision: 84.5%
- F1-Score: 87.1%
- Successfully predicts campaign outcomes with strong performance

## Technologies Used

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- Jupyter Notebook

## Skills Demonstrated

- Binary classification
- Feature engineering
- Model evaluation and selection
- Hyperparameter tuning
- Handling imbalanced datasets
- Data preprocessing pipelines

## How to Run

1. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

2. Download Kickstarter dataset (place in project folder)

3. Run the notebook:
```bash
jupyter notebook kickstarter_prediction.ipynb
```

## Author

Matthieu Lafont  
Master of Management in Analytics - McGill University
