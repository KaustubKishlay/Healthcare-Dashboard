# Healthcare Analytics Dashboard

A Streamlit app for breast cancer diagnostic analytics and predictive modeling.

## Overview

This dashboard uses the Breast Cancer Wisconsin dataset to provide:
- Data overview with class distribution and quality metrics
- Exploratory analysis with distributions, correlations, and scatter plots
- A Random Forest predictive model for malignant vs benign diagnosis
- Business insights and recommendations for clinical decision support

## Features

- Interactive Streamlit interface with sidebar navigation
- Plotly charts for visual analytics
- Model performance metrics including accuracy, ROC curve, and confusion matrix
- Top predictive feature importance and risk factor analysis

## Requirements

Install Python dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Run the app

From the project folder, start Streamlit:

```bash
streamlit run app.py
```

Then open the local URL provided by Streamlit in your browser.

## Project Structure

- `app.py` — main Streamlit application
- `requirements.txt` — Python package dependencies
- `screenshots/` — optional screenshot assets

## Notes

- The app loads the built-in `sklearn.datasets.load_breast_cancer` dataset
- No external dataset download is required
- The Random Forest model is retrained when the test split size changes

## License

This project is provided as-is for demonstration and analytics purposes.
