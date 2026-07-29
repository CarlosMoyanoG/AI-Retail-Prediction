# Retail Demand Forecasting and Inventory Planning

A machine-learning application for forecasting product demand and generating inventory replenishment recommendations in a retail environment.

The project combines an XGBoost regression model, time-series feature engineering, ONNX model serving, and an interactive Streamlit dashboard. It was developed collaboratively as an academic artificial-intelligence project.

## Overview

Retail inventory decisions depend on anticipating future demand while avoiding both stockouts and excessive inventory.

This application supports that process by:

- Forecasting future product demand
- Analysing historical sales behaviour
- Generating multi-day recursive predictions
- Comparing projected demand with current stock
- Calculating suggested purchase quantities
- Visualising forecasts and inventory indicators
- Loading a pre-trained model through ONNX Runtime
- Allowing demonstration data to be replaced through file uploads

The deployed dashboard focuses on decision support rather than automatic purchasing.

## Main Features

### Demand Forecasting

The application predicts future demand using historical sales and calendar-based features.

Forecasts may be generated for:

- A single future date
- A selected date range
- Individual products
- Multiple products in a general analysis

### Recursive Multi-Day Prediction

For multi-day forecasts, predicted demand is fed back into the historical series so that subsequent dates can use updated lag and rolling-window values.

This allows the application to generate forecasts beyond the immediately following day.

### Inventory Recommendations

The application compares projected demand with:

- Current stock
- User-defined safety stock
- Expected demand over the selected horizon

The suggested replenishment quantity follows this rule:

```text
Suggested Purchase =
max(0, Projected Demand + Safety Stock - Current Stock)
```

Negative purchase recommendations are prevented when available stock already covers projected requirements.

### Product Rotation Classification

Products are classified according to their historical sales rotation:

- High rotation
- Medium rotation
- Low rotation

Rotation categories are incorporated into the predictive feature set.

### Interactive Dashboard

The Streamlit interface includes:

- Product selection
- Single-date and date-range forecasting
- Current-stock input
- Safety-stock configuration
- General product analysis
- Forecast tables
- Interactive Plotly charts
- Inventory recommendations
- Model-performance information
- Optional CSV data uploads

### Portable Model Inference

The trained model is exported to ONNX and loaded with ONNX Runtime.

This separates model training from application inference and reduces the runtime dependency on the original training pipeline.

## Model Summary

| Attribute | Value |
|---|---|
| Model | XGBoost Regressor |
| Export format | ONNX |
| Validation metric | Mean Absolute Error |
| Validation MAE | Approximately `5.33` units |
| Average daily sales in metadata | Approximately `56.81` units |
| Number of model features | `15` |
| Products included in model metadata | `10` |

The reported MAE comes from the included model metadata and should be interpreted in relation to the dataset and validation procedure used in the notebook.

## Feature Engineering

The model uses 15 engineered features.

### Calendar Features

- Day of the week
- Day of the month
- Month
- Quarter
- Year
- Weekend indicator

### Product-Rotation Features

- High-rotation indicator
- Medium-rotation indicator
- Low-rotation indicator

### Lag Features

- Sales lag of 7 days
- Sales lag of 14 days
- Sales lag of 30 days

### Rolling Features

- 7-day moving average
- 30-day moving average
- 7-day rolling standard deviation

These variables help the model capture seasonality, recent trends, product behaviour, and demand variability.

## Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| Dashboard | Streamlit |
| Predictive model | XGBoost |
| Model serving | ONNX Runtime |
| Data processing | pandas and NumPy |
| Visualisation | Plotly |
| Model evaluation | scikit-learn |
| Dataset storage | CSV and Parquet |
| Experimentation | Jupyter Notebook |

## Project Structure

```text
Prediccion-Retail/
├── app.py
├── verificar_modelo.py
├── Prototipo_V2_Retail.ipynb
├── requirements.txt
├── ejecutar.txt
├── config.toml
├── modelo/
│   ├── metadata.json
│   ├── modelo.onnx
│   └── historico_features.parquet
├── data/
│   ├── train.csv
│   └── stock_actual.csv
├── .streamlit/
│   └── config.toml
└── README.md
```

### File Responsibilities

- `app.py`: Streamlit interface, inference logic, feature generation, visualisation, and inventory recommendations
- `Prototipo_V2_Retail.ipynb`: model experimentation, training, evaluation, and ONNX export
- `verificar_modelo.py`: verifies that the model metadata and historical feature data can be loaded
- `modelo/modelo.onnx`: exported predictive model
- `modelo/metadata.json`: model configuration, features, products, and validation metrics
- `modelo/historico_features.parquet`: historical features used by the application
- `data/train.csv`: demonstration sales dataset
- `data/stock_actual.csv`: current-stock demonstration data
- `requirements.txt`: Python dependencies

## Requirements

Before running the project, install:

- Python 3.10 or later
- `pip`
- A Python virtual environment

## Installation

1. Clone the repository:

```bash
git clone https://github.com/CarlosMoyanoG/Prediccion-Retail.git
cd Prediccion-Retail
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment.

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```cmd
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

4. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Model Verification

Verify that the model and metadata are available:

```bash
python verificar_modelo.py
```

A successful execution should report:

- Model loaded correctly
- ONNX model type
- Validation MAE
- Number of historical records
- Number of model features

## Running the Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

The dashboard will normally be available at:

```text
http://localhost:8501
```

On Windows, this command may also be used:

```powershell
py -m streamlit run app.py
```

## Data Inputs

The repository includes demonstration data under `data/`.

Expected data should provide the fields required by the feature-engineering and forecasting logic, including:

- Product identifier
- Date
- Historical sales
- Current stock, when applicable

Before replacing the sample files, review the transformations in `app.py` and the training notebook to preserve compatible column names and data types.

## Model Inference

The application uses the `ModeloONNX` wrapper to expose an interface similar to a scikit-learn estimator:

```python
class ModeloONNX:
    def __init__(self, ruta_onnx):
        self.sesion = rt.InferenceSession(ruta_onnx)

    def predict(self, X):
        X_np = np.asarray(X, dtype=np.float32)
        return self.sesion.run(
            None,
            {"float_input": X_np},
        )[0].flatten()
```

The input feature order must match the order stored in:

```text
modelo/metadata.json
```

## Demonstration Video

A demonstration video is available on YouTube:

[Watch the predictive retail dashboard demonstration](https://youtu.be/qil-y4AQN30)

## Model Evaluation

The included metadata reports:

```text
MAE: 5.3341 units
Average sales: 56.8083 units
```

MAE measures the average absolute difference between predicted and observed sales.

The value should not be interpreted as universal production performance because it depends on:

- The source dataset
- The selected products
- The validation split
- The forecast horizon
- The feature-engineering process
- Changes in future retail behaviour

## Current Limitations

- The project uses a fixed pre-trained model.
- The model supports only the products included in its metadata.
- Retraining is performed separately in the Jupyter notebook.
- The repository does not provide an automated training pipeline.
- The validation methodology should be documented in greater detail.
- The included data appears intended for demonstration and academic evaluation.
- External retail events, promotions, prices, holidays, and supply constraints are not explicitly documented as features.
- Recursive forecasting may accumulate prediction error over longer horizons.
- No probabilistic prediction intervals are provided.
- The dashboard does not persist operational decisions in a production database.
- Authentication and user management are not included.
- Automated application tests are not currently included.

## Future Improvements

Recommended improvements include:

- Add a reproducible training pipeline
- Document the temporal validation strategy
- Add baseline-model comparisons
- Add additional metrics such as RMSE, MAPE, and WAPE
- Add prediction intervals
- Add holiday and promotion features
- Add price and discount variables
- Add external economic and seasonal variables
- Add automated data validation
- Add model-version tracking
- Add experiment tracking with MLflow
- Add scheduled retraining
- Add drift detection
- Add explainability with SHAP
- Add a production database
- Add authentication and user roles
- Add automated tests
- Add Docker support
- Add CI/CD validation
- Add deployment documentation
- Add model-monitoring dashboards

## Academic Purpose

This project demonstrates:

- Retail demand forecasting
- Supervised machine learning
- XGBoost regression
- Time-series feature engineering
- Lag and rolling-window features
- Recursive forecasting
- Model evaluation with MAE
- ONNX model export and inference
- Streamlit dashboard development
- Plotly visualisation
- Inventory-replenishment logic
- Team-based artificial-intelligence project development

## Contributors

- Daniela Auquilla — Project Manager
- José Salamea
- Pedro González
- [Carlos Moyano Guevara](https://github.com/CarlosMoyanoG)

## Project Status

Academic artificial-intelligence and retail-forecasting project maintained as part of a software engineering portfolio.

## License

No open-source licence has been added yet. Unless a licence is explicitly included, the source code and model artefacts may be viewed for educational and portfolio purposes but should not be copied, modified, distributed, or used commercially without permission.
