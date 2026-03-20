## Turbofan Streamlit App — Setup & Run

### 1. Create and activate a virtual environment

```bash
cd turbofan-pedictive-maintenance
python -m venv .venv
.venv\Scripts\activate  # on Windows
```

### 2. Install dependencies

```bash
pip install -r requirements_streamlit.txt
```

This will install Streamlit, Plotly, scikit-learn, SHAP, and other libraries needed for the UI only (model training has its own environment in the notebooks).

### 3. Configure environment variables

Create a `.env` file in the project root (if it does not already exist) and add:

```bash
XAI_API_KEY=your_xai_grok_api_key_here
```

If `XAI_API_KEY` is missing or invalid, the **AI Advisor** section will show a clear error message, but the Fleet Dashboard and Engine Deep Dive will still work.

### 4. Check that required data & model files exist

The app expects:

- `data/processed/tableau_all_datasets.csv`
- `data/processed/train_FD001_processed.csv`
- `data/processed/train_FD002_processed.csv`
- `data/processed/train_FD003_processed.csv`
- `data/processed/train_FD004_processed.csv`
- `outputs/models/best_random_forest.pkl`
- `outputs/models/feature_cols.pkl`

If any of these are missing, the app will display an error at the top instead of crashing silently.

### 5. Run the app

From the project root:

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`) in your browser.

