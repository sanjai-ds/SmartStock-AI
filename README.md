# SmartStock-AI
### AI-Based Demand Forecasting and Intelligent Inventory Decision Support System

**SmartStock-AI** is a deep learning-based demand forecasting and inventory decision-support system designed to help identify future product demand, monitor stockout risks and support inventory replenishment decisions.

The project combines historical sales analysis, feature engineering, PyTorch forecasting models and an interactive Streamlit dashboard to transform inventory data into meaningful business insights.

---

## 1. Project Overview

Inventory management is an important challenge for businesses. Excess inventory can increase storage costs, while insufficient stock can lead to stockouts and missed sales opportunities.

SmartStock-AI addresses this challenge by:

- Analysing historical sales and inventory data.
- Engineering time-series features to capture demand patterns.
- Training and comparing multiple forecasting models.
- Predicting future product demand.
- Identifying potential inventory risks.
- Generating reorder recommendations.
- Visualising results through an interactive dashboard.

## 2. Key Features

- **Demand Forecasting:** Predicts future product demand using deep learning.
- **Feature Engineering:** Uses lag features and rolling statistical features.
- **Model Comparison:** Evaluates Naive Baseline, MLP, LSTM and Adaptive/Hybrid approaches.
- **Model Evaluation:** Uses MAE, RMSE, WAPE and sMAPE.
- **Inventory Risk Analysis:** Classifies inventory records into High, Medium and Low risk.
- **Reorder Recommendations:** Supports REORDER and NO REORDER decisions.
- **Interactive Dashboard:** Presents forecasting and inventory insights using Streamlit.

## 3. System Workflow

```text
Historical Sales & Inventory Data
              |
              v
       Data Preprocessing
              |
              v
        Feature Engineering
              |
              v
      Model Training & Testing
              |
              v
       Forecasting Models
              |
       +------+------+------+
       |      |      |      |
       v      v      v      v
     Naive   MLP    LSTM  Hybrid
              |
              v
       Model Evaluation
              |
              v
       Demand Prediction
              |
              v
      Inventory Risk Analysis
              |
              v
      Reorder Recommendations
              |
              v
       Streamlit Dashboard
```

## 4. Models Implemented

| Model | Description |
|---|---|
| Naive Baseline | Provides a reference forecasting approach. |
| PyTorch MLP | Learns nonlinear relationships from engineered features. |
| PyTorch LSTM | Learns patterns from sequential demand information. |
| Adaptive/Hybrid | Provides an alternative forecasting approach for comparison. |

The PyTorch MLP was selected as the primary model based on its MAE and sMAPE performance. The Hybrid model achieved better RMSE and WAPE.

## 5. Model Performance

The following are the reported test results:

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| PyTorch MLP | 23.65 | 31.14 | 39.14% |
| Adaptive/Hybrid | 24.21 | 29.42 | 38.43% |
| PyTorch LSTM | 25.05 | 30.74 | 39.77% |
| Naive Baseline | 32.41 | 42.78 | 53.63% |

### Evaluation Metrics

- **MAE:** Mean Absolute Error.
- **RMSE:** Root Mean Squared Error.
- **WAPE:** Weighted Absolute Percentage Error.
- **sMAPE:** Symmetric Mean Absolute Percentage Error.

## 6. Streamlit Dashboard

The project includes an interactive dashboard developed using Streamlit.

### Dashboard Overview


### Priority Products
![Priority Products](screenshots/priority_products.png)

### Demand Forecasting
![Demand Forecasting](screenshots/demand_forecasting.png)

### Products with Highest Predicted Demand
![Top Demand Products](screenshots/top_demand_products.png)

### Model Evaluation and Feature Importance
![Model Evaluation](screenshots/model_evaluation.png)

### Inventory Monitoring
![Inventory Records](screenshots/inventory_records.png)

### Replenishment Recommendations
![Replenishment Queue](screenshots/replenishment_queue.png)

## 7. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core development |
| PyTorch | Deep learning models |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Scikit-learn | Feature scaling and preprocessing |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| Plotly | Interactive charts |
| Streamlit | Dashboard development |

## 8. Project Structure

```text
SmartStock-AI/
│
├── app/
│   └── smartstock_app.py
│
├── data/
│   ├── Practise dataset.csv
│   └── model_ready_weekly.csv
│
├── models/
│   ├── best_demand_mlp.pth
│   ├── demand_scaler.pkl
│   └── model_config.json
│
├── notebooks/
│   └── 01_data_analysis.ipynb
│
├── results/
│   ├── mlp_feature_importance.csv
│   ├── smartstock_dashboard_data.csv
│   └── smartstock_recommendations.csv
│
├── src/
│
├── screenshots/
│   ├── dashboard_overview.png
│   ├── priority_products.png
│   ├── demand_forecasting.png
│   ├── top_demand_products.png
│   ├── model_evaluation.png
│   ├── inventory_records.png
│   └── replenishment_queue.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 9. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project

```bash
cd SmartStock-AI
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Dashboard

```bash
streamlit run app/smartstock_app.py
```

Open the local URL displayed in your terminal to access the dashboard.

## 10. Future Enhancements

- Real-time inventory data integration.
- Automated model retraining.
- Integration of supplier lead times.
- Improved inventory optimisation techniques.
- Automated stock alerts.
- Cloud-based dashboard deployment.

## 11. Project Outcome

SmartStock-AI demonstrates how deep learning-based demand forecasting can be integrated with inventory information to support stockout risk monitoring and replenishment decisions.

The project brings together data preprocessing, feature engineering, model development, performance evaluation and interactive dashboard development in one practical application.

---

**SmartStock-AI | Smarter Demand Forecasting. Better Inventory Decisions.**