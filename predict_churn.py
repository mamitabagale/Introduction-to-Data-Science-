"""
predict_churn.py
================
Module and CLI tool for predicting customer churn probability and risk percentile.

Week 5 DS Automation Assignment & Additional Challenges:
- Class-based architecture (ChurnPredictor)
- Function taking pandas dataframe and returning churn probabilities
- Calculation of churn probability & percentile ranking relative to training distribution
- Automatic preprocessing pipeline supporting both prepared and unmodified churn data
- CLI argument and interactive user input support
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from pycaret.classification import load_model, predict_model


class ChurnPredictor:
    """
    Predicts customer churn probability, binary prediction label,
    and percentile rank within the training population.
    """

    EXPECTED_FEATURES = [
        'tenure',
        'PhoneService',
        'MonthlyCharges',
        'TotalCharges',
        'charge_per_tenure',
        'Contract_One year',
        'Contract_Two year',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check',
    ]

    def __init__(
        self,
        model_path='pycaret_churn_model',
        train_probs_path='train_churn_probabilities.csv',
    ):
        """
        Initialize the predictor by loading the trained PyCaret pipeline
        and training probability distribution for percentile scoring.
        """
        # Load pycaret model (strip .pkl extension if present as load_model appends it)
        base_model_path = model_path.replace('.pkl', '')
        if not os.path.exists(f'{base_model_path}.pkl') and not os.path.exists(
            model_path
        ):
            raise FileNotFoundError(
                f'Model file not found at {base_model_path}.pkl. Please train and save the model first.'
            )

        self.model = load_model(base_model_path, verbose=False)

        # Load training probabilities for percentile calculation
        if os.path.exists(train_probs_path):
            self.train_probs = pd.read_csv(train_probs_path)[
                'prob_churn'
            ].values
        else:
            self.train_probs = None
            print(
                f"Warning: '{train_probs_path}' not found. Percentiles will be omitted."
            )

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses a raw or prepared DataFrame to match the 10 features expected by the model.
        Handles both prepared data (new_churn_data.csv) and unmodified data (new_churn_data_unmodified.csv).
        """
        data = df.copy()

        # 1. TotalCharges: ensure numeric and fill missing values with 0
        if 'TotalCharges' in data.columns:
            data['TotalCharges'] = pd.to_numeric(
                data['TotalCharges'], errors='coerce'
            ).fillna(0.0)

        # 2. charge_per_tenure: calculate if not already present (TotalCharges / (tenure + 1))
        if 'charge_per_tenure' not in data.columns and 'tenure' in data.columns:
            data['charge_per_tenure'] = data['TotalCharges'] / (
                data['tenure'] + 1
            )

        # 3. PhoneService: map binary strings ('Yes'/'No') to 1/0 if needed
        if 'PhoneService' in data.columns:
            if data['PhoneService'].dtype == object:
                data['PhoneService'] = (
                    data['PhoneService']
                    .astype(str)
                    .str.strip()
                    .map({'Yes': 1, 'No': 0})
                    .fillna(0)
                    .astype(int)
                )

        # 4. Contract: map strings or integer encoding to one-hot dummy columns
        if 'Contract' in data.columns:
            if data['Contract'].dtype == object:
                data['Contract_One year'] = (
                    data['Contract'] == 'One year'
                ).astype(int)
                data['Contract_Two year'] = (
                    data['Contract'] == 'Two year'
                ).astype(int)
            else:  # Integer encoded: 0 = Month-to-month, 1 = One year, 2 = Two year
                data['Contract_One year'] = (data['Contract'] == 1).astype(int)
                data['Contract_Two year'] = (data['Contract'] == 2).astype(int)

        # 5. PaymentMethod: map strings or integer encoding to one-hot dummy columns
        if 'PaymentMethod' in data.columns:
            if data['PaymentMethod'].dtype == object:
                data['PaymentMethod_Credit card (automatic)'] = (
                    data['PaymentMethod'] == 'Credit card (automatic)'
                ).astype(int)
                data['PaymentMethod_Electronic check'] = (
                    data['PaymentMethod'] == 'Electronic check'
                ).astype(int)
                data['PaymentMethod_Mailed check'] = (
                    data['PaymentMethod'] == 'Mailed check'
                ).astype(int)
            else:  # Integer encoded: 0 = Credit card, 1 = Mailed check, 2 = Electronic check, 3 = Bank transfer
                data['PaymentMethod_Credit card (automatic)'] = (
                    data['PaymentMethod'] == 0
                ).astype(int)
                data['PaymentMethod_Electronic check'] = (
                    data['PaymentMethod'] == 2
                ).astype(int)
                data['PaymentMethod_Mailed check'] = (
                    data['PaymentMethod'] == 1
                ).astype(int)

        # Ensure all expected columns exist (fill missing with 0)
        for col in self.EXPECTED_FEATURES:
            if col not in data.columns:
                data[col] = 0

        return data[self.EXPECTED_FEATURES]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes an input DataFrame (raw or prepared), preprocesses it, and returns a DataFrame
        containing predictions, churn probabilities, and percentile rankings.
        """
        preprocessed_df = self.preprocess(df)

        # Predict using PyCaret pipeline
        raw_preds = predict_model(self.model, data=preprocessed_df)

        # PyCaret 3.x outputs 'prediction_label' and 'prediction_score'
        # Note: 'prediction_score' is the probability of the predicted class.
        # Churn probability (P(Churn=1)) = score if label==1 else (1 - score).
        labels = raw_preds['prediction_label'].values
        scores = raw_preds['prediction_score'].values

        churn_probabilities = np.where(labels == 1, scores, 1.0 - scores)

        results = df.copy()
        results['predicted_churn'] = labels
        results['churn_probability'] = np.round(churn_probabilities, 4)

        # Calculate percentile relative to training population
        if self.train_probs is not None:
            percentiles = [
                round(float(percentileofscore(self.train_probs, p)), 2)
                for p in churn_probabilities
            ]
            results['probability_percentile'] = percentiles

        return results


def predict_churn_probability(
    df: pd.DataFrame, model_path='pycaret_churn_model'
) -> pd.DataFrame:
    """
    Functional interface required by Task 1:
    Takes a pandas DataFrame as input and returns the predictions with churn probability.
    """
    predictor = ChurnPredictor(model_path=model_path)
    return predictor.predict(df)


def main():
    print('=' * 75)
    print('       CUSTOMER CHURN AUTOMATED PREDICTION SYSTEM       ')
    print('=' * 75)

    # Challenge: Accept user input via CLI argument or interactive prompt
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f'Using file specified via command line argument: {filepath}')
    else:
        default_file = 'new_churn_data.csv'
        user_input = input(
            f'Enter path to CSV data file [Press Enter for default: {default_file}]: '
        ).strip()
        filepath = user_input if user_input else default_file

    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    print(f"\nLoading data from '{filepath}'...")
    df = pd.read_csv(filepath)
    print(f'Data loaded successfully! Shape: {df.shape}')

    predictor = ChurnPredictor()
    results = predictor.predict(df)

    display_cols = [
        c
        for c in [
            'customerID',
            'tenure',
            'MonthlyCharges',
            'predicted_churn',
            'churn_probability',
            'probability_percentile',
        ]
        if c in results.columns
    ]

    print('\nPrediction Results:')
    print(results[display_cols].to_string(index=False))

    # True values check
    true_values = [1, 0, 0, 1, 0]
    if len(results) == len(true_values):
        print('\n' + '-' * 75)
        print('Validation against True Ground Truth:')
        print(f'Predicted Churn: {list(results["predicted_churn"])}')
        print(f'True Churn:      {true_values}')
        matches = sum(
            p == t
            for p, t in zip(results['predicted_churn'].values, true_values)
        )
        print(
            f'Accuracy on New Data: {matches}/{len(true_values)} ({matches/len(true_values)*100:.1f}%)'
        )
        print('-' * 75)


if __name__ == '__main__':
    main()
