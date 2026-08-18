# Late Delivery Risk Predictor — ML Assignment 2

**Course:** M.Tech (AIML/DSE) — Machine Learning  
**Submission Deadline:** 18-Aug-2026  
**Student Name:** Shivam Kumar Singh  
**Stramlit.app:** https://mlassignment2shivam.streamlit.app/

---

## a. Problem Statement

Late deliveries are one of the most expensive and reputation-damaging problems for
any e-commerce / supply-chain business — they drive customer complaints, refund
requests, and churn. This project answers a concrete **business question**:

> **Can we predict, using information available at the moment an order is placed
> (customer segment, shipping mode, product category, order value, discount,
> scheduled shipping days, etc.), whether that order is at risk of a late
> delivery — *before* it actually ships?**

If the answer is "yes", a supply chain / operations team can use the model as a
**real-time risk-scoring step** at order confirmation: high-risk orders can be
routed to faster shipping, prioritized in the warehouse, or the customer can be
proactively notified of a possible delay — turning a reactive complaint into a
proactive service action.

This is framed as a **binary classification problem**: predict `Late_delivery_risk`
(`1` = at risk of late delivery, `0` = on-time/early).

---

## b. Dataset Description

- **Source:** DataCo Smart Supply Chain Dataset (`DataCoSupplyChainDataset.csv`)
- **Size:** 180,519 orders × 53 columns (original); a stratified, balanced sample
  of **20,000 orders** was used for modelling (10,000 per class) to keep training
  and the Streamlit free-tier deployment fast, while comfortably exceeding the
  assignment's minimum of 500 instances.
- **Target variable:** `Late_delivery_risk` (binary: `1` = late, `0` = on-time)
- **Features used (14 total, ≥ 12 required):**

  | Type | Features |
  |---|---|
  | Numeric (7) | `Days for shipment (scheduled)`, `Benefit per order`, `Sales per customer`, `Order Item Discount Rate`, `Order Item Product Price`, `Order Item Quantity`, `Order Item Profit Ratio` |
  | Categorical (7) | `Type` (payment type), `Category Name`, `Customer Segment`, `Department Name`, `Market`, `Order Region`, `Shipping Mode` |

- **Important design decision — avoiding data leakage:** the dataset also
  contains `Days for shipping (real)` and `Delivery Status`, both of which are
  only known *after* an order has already shipped/arrived and directly leak the
  target. These columns were **deliberately excluded** from the feature set so
  that the model reflects what is genuinely knowable at order time. PII columns
  (name, email, password, address) were also excluded.

---

## c. GitHub Repository Link

**Repository:** https://github.com/shivamk423/ML_Assignment_2

Repository structure:
```
project-folder/
│-- app.py                       # Streamlit web application
│-- requirements.txt             # Python dependencies
│-- README.md                    # This file
│-- test_data.csv                # Sample test data (features + true label)
│-- DataCoSupplyChainDataset.csv # Full raw dataset (see note below)
│-- model/
│   │-- train_models.ipynb       # Full training notebook (EDA -> training -> evaluation)
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest.pkl
│   │-- feature_columns.json     # Feature schema used by app.py
│   └-- model_comparison_table.csv
```

> **Note on `DataCoSupplyChainDataset.csv` (~95 MB):** it is under GitHub's
> 100 MB hard file-size limit but close to it. If `git push` is slow/rejected
> on your connection, either (a) use [Git LFS](https://git-lfs.com/) for this
> one file, or (b) `.gitignore` it and add a short note in your repo pointing
> to the original Kaggle source ("DataCo Smart Supply Chain") — the notebook
> only needs it for the one-time training step; `app.py` and grading only
> depend on `test_data.csv` and the saved `.pkl` models, neither of which
> needs the raw file.

---

## d. Models Used

All 5 models below were trained on an identical 80/20 train-test split of the
same 20,000-row balanced sample, each wrapped in the same preprocessing
pipeline (StandardScaler for numeric features, OneHotEncoder for categorical
features) for a fair comparison. Full code, comments, and step-by-step
explanation are in [`model/train_models.ipynb`](model/train_models.ipynb).

> **Note:** The assignment brief mentions "6 ML models" in Section 1, but
> Section 2 and the comparison-table template both list only 5 models
> (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest). All 5
> explicitly listed models have been implemented here.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7180 | 0.7448 | 0.8239 | 0.5545 | 0.6629 | 0.4614 |
| Decision Tree | 0.7120 | 0.7384 | 0.7944 | 0.5720 | 0.6651 | 0.4417 |
| kNN | 0.7048 | 0.7370 | 0.7627 | 0.5945 | 0.6682 | 0.4198 |
| Naive Bayes | 0.7075 | 0.7449 | 0.7664 | 0.5970 | 0.6712 | 0.4255 |
| Random Forest (Ensemble) | 0.7128 | 0.7487 | 0.7831 | 0.5885 | 0.6720 | 0.4393 |

*Every model was tuned with `RandomizedSearchCV` (3-fold stratified CV; see Section 10 of the training notebook) before evaluation. Exact values are reproduced from `model/model_comparison_table.csv`, generated by running the training notebook. Re-running the notebook may shift results very slightly depending on library versions, but the ranking/conclusions should hold.*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, well-balanced baseline — highest **Precision** and highest **MCC** of all models, meaning when it flags an order as "late," it's usually right. After tuning regularization (`C = 0.1`, still optimal after a wider second-pass search), its Accuracy, AUC, Precision and MCC all improved. Its linear decision boundary works well because several features (scheduled shipping days, discount rate) have a fairly monotonic relationship with late risk. |
| Decision Tree | Captures non-linear interactions (e.g., "Standard Class AND short scheduled window"). Extended tuning (50 draws) lifted its Accuracy to 0.7120 and AUC to 0.7384 with the third-best MCC (0.4417) — it still trails the ensemble on AUC, consistent with a single tree generalizing slightly worse than the averaged forest. |
| kNN | Weakest performer on Accuracy/Precision/MCC, though wider tuning (up to 41 neighbours, Manhattan distance) lifted its AUC to 0.7370. High-dimensional one-hot-encoded categorical space (Market, Region, Category, etc.) makes Euclidean distance a less meaningful similarity metric — a known weakness of kNN on mixed numeric/categorical data. |
| Naive Bayes | Performs respectably despite its strong (unrealistic) feature-independence assumption. A wider `var_smoothing` grid pushed its AUC to 0.7449 — nearly tying the ensemble — with competitive Precision (0.7664) and MCC (0.4255). |
| Random Forest (Ensemble) | **Best overall AUC (0.7487) and best F1 (0.6720)**. As an ensemble of trees it captures non-linear interactions like a single Decision Tree while reducing overfitting through averaging/bagging — the most robust choice for this mixed-type feature set. The tuned shallow-tree, 150-tree configuration traded a little precision for a healthier recall/F1 balance. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — best AUC (0.7487) and best F1 (0.6720), and the most balanced all-round performer. Extended hyperparameter tuning confirmed a shallow-tree, 150-tree configuration (`max_depth=8`, `min_samples_leaf=4`, `min_samples_split=5`, `max_features='log2'`, `criterion='entropy'`, `bootstrap=True`) as optimal, lifting F1 from 0.6645 to 0.6720 while keeping the top AUC. It generalizes better than the single Decision Tree and handles the categorical/numeric feature mix better than kNN, making it the recommended model for production risk-scoring. |

### Business Interpretation

- **Shipping Mode** and **scheduled shipping days** are the strongest operational
  levers: `Standard Class` shipments and orders promised a very short delivery
  window show disproportionately higher late rates (see EDA plots in the
  notebook), meaning tightening SLAs or reserving Standard Class for
  lower-priority orders is a concrete, actionable recommendation for the
  business — independent of which model is deployed.
- All 5 models beat the random-guess baseline (AUC 0.50), confirming that
  late-delivery risk **is predictable, to a meaningful degree, at order time**
  from order/customer/product attributes alone — directly answering the
  business question posed above.
- **Recommended action:** deploy Random Forest as a checkout-time / order-
  confirmation-time risk score; route high-risk orders to expedited shipping
  or proactive customer communication.

---

## Streamlit App Features

The deployed app (`app.py`) includes all required features:

- ✅ **(a)** CSV upload option for test data (sidebar)
- ✅ **(b)** Model selection dropdown (choose between all 5 trained models)
- ✅ **(c)** Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC) — shown automatically when the uploaded CSV includes the true `Late_delivery_risk` label column
- ✅ **(d)** Confusion matrix (heatmap) and full classification report

### Deployment

**Live Streamlit App Link:** 
Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) by:
1. Sign in with your GitHub account
2. Click "New App" 
3. Select this repository (`ML_assignment2`)
4. Choose branch `main`
5. Set main file path to `project_final/app.py`
6. Click Deploy

Once deployed, your app URL will be: `https://share.streamlit.io/[YOUR_GITHUB_USERNAME]/ML_assignment2/main/project_final/app.py`

### Running locally
```bash
# Clone the repository
git clone https://github.com/2025da04127-shivam/ML_assignment2.git
cd ML_assignment2/project_final

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

Then:
1. Upload your CSV file, or click "Use bundled test_data.csv" in the sidebar
2. Select a model from the dropdown (Logistic Regression, Decision Tree, kNN, Naive Bayes, or Random Forest)
3. Click "🚀 Run Model"
4. View predictions, evaluation metrics, confusion matrix, and classification report

**Sample test data included:** `test_data.csv` (500 orders with true Late_delivery_risk labels)

---

## How the Model Was Trained

See [`model/train_models.ipynb`](model/train_models.ipynb) for the complete,
commented, step-by-step notebook covering:
- **Data Loading & Preprocessing:** Loading 180,519 orders and extracting a stratified balanced sample of 20,000
- **Leakage-Safe Feature Selection:** Excluding post-shipping information (Days for shipping real, Delivery Status)
- **Exploratory Data Analysis:** Distribution analysis and correlation with target variable
- **Feature Engineering:** StandardScaler for 7 numeric features, OneHotEncoder for 7 categorical features
- **Model Training:** All 5 models trained on 80/20 split with stratified cross-validation
- **Hyperparameter Tuning:** RandomizedSearchCV (50 iterations, 3-fold stratified CV) for each model
- **Evaluation:** Computing Accuracy, AUC, Precision, Recall, F1, MCC for each model
- **Visualization:** Confusion matrices, ROC curves, and feature importance plots
- **Model Export:** Saving all trained pipelines as `.pkl` files for deployment

**Key Training Decisions:**
- **Data Split:** 80/20 train-test (stratified to maintain class balance)
- **Sample Size:** 20,000 orders (10,000 per class for balance) — faster training while maintaining statistical significance
- **Preprocessing:** Unified pipeline ensures no data leakage and fair model comparison
- **Hyperparameter Search:** Extensive RandomizedSearchCV to find optimal parameters for each model

---

## Assignment Requirements Checklist

✅ **Step 1: Dataset Choice**
- Source: DataCo Smart Supply Chain (Kaggle)
- Classification Type: Binary classification
- Instance Size: 20,000 (exceeds 500 minimum)
- Feature Size: 14 features (exceeds 12 minimum)

✅ **Step 2: ML Models Implemented** (All 5 required models)
1. Logistic Regression
2. Decision Tree Classifier  
3. K-Nearest Neighbor (kNN) Classifier
4. Naive Bayes Classifier (Gaussian)
5. Random Forest Classifier (Ensemble)

All metrics calculated: Accuracy, AUC, Precision, Recall, F1, MCC

✅ **Step 3: GitHub Repository** - Complete with all required files

✅ **Step 4: Requirements.txt** - All dependencies listed

✅ **Step 5: README.md** - Problem statement, dataset description, models, observations

✅ **Step 6: Streamlit App Features**
- (a) CSV upload option for test data ✓
- (b) Model selection dropdown ✓
- (c) Evaluation metrics display ✓
- (d) Confusion matrix & classification report ✓

✅ **Step 7: BITS Virtual Lab Screenshot** - 

---

## Screenshot (BITS Virtual Lab Execution)

<img width="958" height="502" alt="image" src="https://github.com/user-attachments/assets/3e09ff6e-62d3-4936-ae62-2de29253ec84" />

<img width="957" height="500" alt="image" src="https://github.com/user-attachments/assets/82789f50-13e7-434a-a84b-1be41935c4fe" />

---

## Tech Stack

**Language & Libraries:**
- **Python 3.8+** — Core programming language
- **pandas** — Data manipulation and analysis
- **scikit-learn** — Machine Learning algorithms and evaluation metrics
- **Streamlit** — Web application framework
- **matplotlib & seaborn** — Data visualization
- **joblib** — Model serialization and loading
- **numpy** — Numerical computing

**Tools & Platforms:**
- **Jupyter Notebook** — Training notebook (`train_models.ipynb`)
- **GitHub** — Version control and repository hosting
- **Streamlit Community Cloud** — Free deployment platform
- **BITS Virtual Lab** — Assignment execution environment

---

## License

This project is submitted as part of the M.Tech (AIML/DSE) Machine Learning assignment at BITS Pilani.

---

## Author

**Shivam Kumar Singh**  
M.Tech (AIML/DSE) — BITS Pilani  
Submission Date: 18-Aug-2026
#
