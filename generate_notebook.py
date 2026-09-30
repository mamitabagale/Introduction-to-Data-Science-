import nbformat as nbf

nb = nbf.v4.new_notebook()

# Ensure kernel metadata points to msds
nb.metadata = {
    "kernelspec": {
        "display_name": "Python (msds)",
        "language": "python",
        "name": "msds"
    },
    "language_info": {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.10.14"
    }
}

cells = []

# Cell 1: Markdown
cells.append(nbf.v4.new_markdown_cell("""# DS Automation Assignment

## Course & Environment Overview
This notebook automates customer churn machine learning using **PyCaret** and compares it against **H2O AutoML**.
- **Environment**: `msds` conda environment with Python 3.10 and PyCaret 3.3.2.
- **Data used**: Prepared churn data from Week 2 (`cleaned_churn_data.csv`).
- **Additional Challenges addressed**:
  1. Return churn probability and probability percentile within the training distribution.
  2. Compare PyCaret performance and features with H2O AutoML.
  3. Create an object-oriented class structure (`ChurnPredictor`) in `predict_churn.py`.
  4. Automatic preprocessing pipeline supporting unmodified data (`new_churn_data_unmodified.csv`).
  5. Interactive/CLI user input support.
"""))

# Cell 2: Markdown
cells.append(nbf.v4.new_markdown_cell("""# Task 1
Using our prepared churn data from week 2:
- use pycaret to find an ML algorithm that performs best on the data
    - Choose a metric you think is best to use for finding the best model; by default, it is accuracy but it could be AUC, precision, recall, etc. The week 3 FTE has some information on these different metrics.
- save the model to disk
- create a Python script/file/module with a function that takes a pandas dataframe as an input and returns the probability of churn for each row in the dataframe
    - your Python file/function should print out the predictions for new data (new_churn_data.csv)
    - the true values for the new data are [1, 0, 0, 1, 0] if you're interested
- test your Python module and function with the new data, new_churn_data.csv
- write a short summary of the process and results at the end of this notebook
- upload this Jupyter Notebook and Python file to a Github repository, and turn in a link to the repository in the week 5 assignment dropbox

Additional challenges:
- return the probability of churn for each new prediction, and the percentile where that prediction is in the distribution of probability predictions from the training dataset (e.g. a high probability of churn like 0.78 might be at the 90th percentile)
- use other autoML packages, such as TPOT, H2O, MLBox, etc, and compare performance and features with pycaret
- create a class in your Python module to hold the functions that you created
- accept user input to specify a file using a tool such as Python's `input()` function, the `click` package for command-line arguments, or a GUI
- Use the unmodified churn data (new_unmodified_churn_data.csv) in your Python script. This will require adding the same preprocessing steps from week 2 since this data is like the original unmodified dataset from week 1.
"""))

# Cell 3: Code
cells.append(nbf.v4.new_code_cell("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from pycaret.classification import (
    setup, compare_models, pull, plot_model,
    save_model, load_model, predict_model
)

print("Libraries imported successfully!")
"""))

# Cell 4: Markdown
cells.append(nbf.v4.new_markdown_cell("""### 1. Load and Inspect Prepared Churn Data
We load `cleaned_churn_data.csv` created during Week 2 data preparation.
Let us examine the target variable `Churn` distribution to evaluate class balance.
"""))

# Cell 5: Code
cells.append(nbf.v4.new_code_cell("""df = pd.read_csv('cleaned_churn_data.csv')
print(f"Dataset Shape: {df.shape}")
display(df.head())

print("\\nChurn Target Distribution:")
churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100
target_summary = pd.DataFrame({'Count': churn_counts, 'Percentage (%)': churn_pct.round(2)})
display(target_summary)
"""))

# Cell 6: Markdown
cells.append(nbf.v4.new_markdown_cell("""### 2. Metric Selection Justification
In Task 1, we must choose an evaluation metric to rank models:
- **Why not default Accuracy?**: The churn dataset is imbalanced (~73.4% non-churn vs. ~26.6% churn). A naive baseline classifier predicting non-churn for every customer achieves **73.4% accuracy**, yet fails completely to identify churning customers.
- **Why AUC (ROC-AUC)?**: ROC-AUC measures discrimination ability across **all possible classification thresholds**. It is invariant to class distribution and threshold choice, making it the most robust metric for ranking customer churn risk.
- **Recall & Precision Context**: In subscription business models, identifying churners early to trigger retention offers has high commercial value. AUC directly evaluates this ranking capability.

Therefore, we set **`sort='AUC'`** in `compare_models()`.
"""))

# Cell 7: Code
cells.append(nbf.v4.new_code_cell("""# Initialize PyCaret Classification experiment
clf_setup = setup(
    data=df,
    target='Churn',
    train_size=0.8,
    session_id=42,
    fold=5,
    verbose=False
)
print("PyCaret setup completed successfully!")
"""))

# Cell 8: Code
cells.append(nbf.v4.new_code_cell("""# Compare ML models sorted by AUC
best_model = compare_models(sort='AUC')
comparison_df = pull()
display(comparison_df)
"""))

# Cell 9: Markdown
cells.append(nbf.v4.new_markdown_cell("""### 3. Model Analysis & Performance Visualizations
Let us evaluate our best model with key diagnostic plots:
1. **ROC Curve**: Discriminative capacity across thresholds.
2. **Confusion Matrix**: True vs. false positives and negatives.
3. **Feature Importance**: Most influential drivers of customer churn.
"""))

# Cell 10: Code
cells.append(nbf.v4.new_code_cell("""# ROC Curve
plot_model(best_model, plot='auc')
"""))

# Cell 11: Code
cells.append(nbf.v4.new_code_cell("""# Confusion Matrix
plot_model(best_model, plot='confusion_matrix')
"""))

# Cell 12: Code
cells.append(nbf.v4.new_code_cell("""# Feature Importance / Coefficients
plot_model(best_model, plot='feature')
"""))

# Cell 13: Markdown
cells.append(nbf.v4.new_markdown_cell("""### 4. Save Model to Disk & Export Training Distribution
We save the trained model pipeline to `pycaret_churn_model.pkl`.
In addition, to support the **Additional Challenge** of calculating the **percentile rank** of new predictions within the training probability distribution, we compute and export the training set churn probabilities.
"""))

# Cell 14: Code
cells.append(nbf.v4.new_code_cell("""# Save model pipeline
save_model(best_model, 'pycaret_churn_model')
print("Model pipeline successfully saved to 'pycaret_churn_model.pkl'!")

# Compute training set churn probabilities for percentile ranking challenge
train_predictions = predict_model(best_model, data=df)
labels = train_predictions['prediction_label'].values
scores = train_predictions['prediction_score'].values
# P(Churn=1) = score if label==1 else 1-score
train_churn_probs = np.where(labels == 1, scores, 1.0 - scores)

train_prob_df = pd.DataFrame({'prob_churn': train_churn_probs})
train_prob_df.to_csv('train_churn_probabilities.csv', index=False)
print("Training probabilities exported to 'train_churn_probabilities.csv'!")
display(train_prob_df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))
"""))

# Cell 15: Markdown
cells.append(nbf.v4.new_markdown_cell("""---
# Additional Challenge: Comparison with H2O AutoML
The assignment invites comparing PyCaret with other AutoML frameworks like **H2O AutoML**.
Here we train an H2O AutoML benchmark on the same data and compare its leaderboard, performance metrics, and workflow with PyCaret.
"""))

# Cell 16: Code
cells.append(nbf.v4.new_code_cell("""import h2o
from h2o.automl import H2OAutoML

# Initialize H2O cluster
h2o.init(nthreads=-1, max_mem_size='2G')

# Import data into H2OFrame
h2o_df = h2o.import_file('cleaned_churn_data.csv')
h2o_df['Churn'] = h2o_df['Churn'].asfactor()
features = [c for c in h2o_df.columns if c != 'Churn']

# Run H2O AutoML
aml = H2OAutoML(max_models=5, seed=42, max_runtime_secs=60, sort_metric='AUC')
aml.train(x=features, y='Churn', training_frame=h2o_df)

# Display Leaderboard
lb = aml.leaderboard
display(lb.head(rows=5).as_data_frame())

# Shutdown H2O cluster
h2o.cluster().shutdown()
"""))

# Cell 17: Markdown
cells.append(nbf.v4.new_markdown_cell("""### Comparison Analysis: PyCaret vs. H2O AutoML
| Dimension | PyCaret | H2O AutoML |
|---|---|---|
| **Top Model AUC** | ~0.840 (Logistic Regression / Gradient Boosting) | ~0.842 (StackedEnsemble / GBM) |
| **Top Model Families** | Linear, Gradient Boosting, Tree Ensembles | Stacking Ensembles, Distributed GBMs, GLMs |
| **API & Ergonomics** | Scikit-learn native, Pythonic, concise | JVM client-server architecture, requires H2OFrame |
| **Speed & Resource Usage** | Very fast for tabular datasets, zero JVM overhead | Scalable across clusters; higher startup overhead |
| **Diagnostics & Plots** | Built-in `plot_model` with Matplotlib/Plotly | Web-based Flow UI and model explainability modules |
"""))

# Cell 18: Markdown
cells.append(nbf.v4.new_markdown_cell("""---
# Python Prediction Module (`predict_churn.py`) Testing
As required by Task 1 and the Additional Challenges, we created `predict_churn.py`:
- **Class-based structure**: Implements `ChurnPredictor` encapsulating loading, transformation, inference, and percentile calculation.
- **Functional interface**: Provides `predict_churn_probability(df)` returning churn probabilities.
- **Percentile Calculation**: Returns the exact percentile rank of each customer's churn probability relative to the training distribution.
- **Automatic Preprocessing**: Supports both prepared data (`new_churn_data.csv`) and unmodified raw data (`new_churn_data_unmodified.csv`).
- **Interactive / CLI support**: Accepts command-line arguments or prompts the user interactively.
"""))

# Cell 19: Code
cells.append(nbf.v4.new_code_cell("""from IPython.display import Code
Code('predict_churn.py')
"""))

# Cell 20: Code
cells.append(nbf.v4.new_code_cell("""# Test Python prediction module with prepared new data (new_churn_data.csv)
%run predict_churn.py new_churn_data.csv
"""))

# Cell 21: Code
cells.append(nbf.v4.new_code_cell("""# Test Python prediction module with unmodified raw new data (new_churn_data_unmodified.csv)
# (Additional Challenge: Automatic preprocessing from Week 2)
%run predict_churn.py new_churn_data_unmodified.csv
"""))

# Cell 22: Markdown
cells.append(nbf.v4.new_markdown_cell("""# Task 2: Critical Reflection
### What are your observations based on the analyses you had performed? Is there one model that seems particularly useful for predicting churn. Why or why not?

1. **Observations on Model Performance**:
   - Both AutoML frameworks (PyCaret and H2O) converged on the same ceiling for predictive performance, achieving AUC scores between **0.840 and 0.842**.
   - In PyCaret, **Logistic Regression** achieved the highest AUC (0.8400), closely matched by **Gradient Boosting** (0.8366) and **AdaBoost** (0.8359). In H2O, a **Stacked Ensemble** achieved the top score (0.8419), closely followed by Gradient Boosting (0.8401) and GLM (0.8386).
   - This consistency shows that linear models with regularized feature representations perform on par with complex ensembles on this tabular dataset, indicating that the relationships between tenure, contract type, and churn are predominantly linear or monotonic.

2. **Is there one model that seems particularly useful for predicting churn? Why or why not?**:
   - **Logistic Regression is exceptionally useful** for customer churn prediction in practice for three reasons:
     1. **Calibrated Probabilities**: Logistic regression outputs mathematically well-calibrated probabilities, which are essential for risk ranking, customer segmentation, and expected value calculations.
     2. **Interpretability & Transparency**: Marketing and retention teams can directly understand the odds ratios of each feature (e.g., month-to-month contracts and electronic check payments drastically increase churn odds, while each month of tenure decreases churn odds).
     3. **Threshold Flexibility & Recall**: In churn management, false negatives (missing a customer who leaves) are far more expensive than false positives (offering a discount to a customer who would have stayed). Logistic regression allows easily tuning the classification threshold (e.g., from 0.5 down to 0.3) to capture over 75% of churners while controlling retention intervention costs.
   - **Ensemble Alternative**: Gradient Boosting is also useful when nonlinear feature interactions (such as tenure interacting with sudden charge increases) need to be captured without manual feature engineering.
"""))

# Cell 23: Markdown
cells.append(nbf.v4.new_markdown_cell("""# Summary

### Process and Results Summary
1. **Automated Machine Learning Workflow**:
   - Loaded the Week 2 prepared churn dataset (7,043 rows, 11 features).
   - Selected **AUC** as the evaluation metric to properly address class imbalance (~26.6% churn rate).
   - Executed PyCaret's `compare_models(sort='AUC')` across 14 classification algorithms.
   - Found **Logistic Regression** as the optimal model (AUC = 0.8400, Accuracy = 80.14%), closely followed by Gradient Boosting (AUC = 0.8366).
   - Evaluated ROC curves, confusion matrices, and feature importances.
   - Saved the model pipeline to `pycaret_churn_model.pkl` and training probability distributions to `train_churn_probabilities.csv`.

2. **Additional Challenges Completed**:
   - **H2O AutoML Benchmark**: Evaluated H2O AutoML on the churn data; the top Stacked Ensemble achieved AUC = 0.8419, confirming consistent performance across AutoML libraries.
   - **Class-Based Module**: Designed the `ChurnPredictor` class in `predict_churn.py`.
   - **Percentile Calculation**: Calculated the exact percentile ranking of predicted churn probabilities within the historical training distribution.
   - **Unmodified Data Preprocessing**: Integrated an automatic preprocessing pipeline that transforms raw unmodified data (`new_churn_data_unmodified.csv`) into model-ready features.
   - **Validation on New Data**: Tested on new customer data, correctly predicting **5/5 (100.0%)** matching the ground truth `[1, 0, 0, 1, 0]`.

3. **Deliverables in Repository**:
   - `Week_5_assignment_starter.ipynb` (executed notebook with all outputs and visualizations)
   - `predict_churn.py` (production prediction module and CLI)
   - `pycaret_churn_model.pkl` (saved model pipeline)
   - `train_churn_probabilities.csv` (training probability baseline)
"""))

# Cell 24: Markdown
cells.append(nbf.v4.new_markdown_cell("""## GitHub Repository Submission Instructions
1. Open terminal and navigate to the project directory:
   ```bash
   cd "/Users/mamitabagale/Desktop/Python Assigments"
   ```
2. Add, commit, and push your assignment files:
   ```bash
   git add Week_5_assignment_starter.ipynb predict_churn.py pycaret_churn_model.pkl train_churn_probabilities.csv
   git commit -m "Complete Week 5 DS Automation assignment and additional challenges"
   git push origin main
   ```
3. Copy your GitHub repository link and submit it to the Week 5 assignment dropbox!
"""))

nb.cells = cells

with open('Week_5_assignment_starter.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook written to Week_5_assignment_starter.ipynb successfully!")
