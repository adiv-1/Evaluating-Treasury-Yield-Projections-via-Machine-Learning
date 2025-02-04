import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from typing import List, Tuple
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "data", "processed")
output_dir = os.path.join(current_dir, "..", "data", "processed")
os.makedirs(output_dir, exist_ok=True)
LSTM_df = pd.read_csv(os.path.join(data_dir, "LSTM_Data_Processed.csv"))
merged_df = pd.read_csv(os.path.join(data_dir, "Lin_Reg_Data_Processed.csv"))


# LSTM Model
def creating_sequences(
    data: List[List[float]], sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Building a preprocessing function to collect data
    into batches of 2-weeks (10 days), as inputs
    to LSTM Model parameters

    Parameters:
        data list[list[float]]: inputs the dataset which is a list of rows
        sequence_length (int): specified length of a given sequence
    Returns:
        tuple: containing an array of sequences and an array of labels
    """
    sequences = []
    labels = []
    for row in range(len(data) - sequence_length):
        current_seq = []
        for length in range(sequence_length):
            current_seq.append(data[row + length])
        sequences.append(current_seq)
        label = data[row + sequence_length]
        labels.append(label)
    return np.array(sequences), np.array(labels)


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(LSTM_df[["10 Yr"]])
sequence_length = 10
sequences, labels = creating_sequences(scaled_data, sequence_length)
X_train, X_test, y_train, y_test = train_test_split(
    sequences, labels, test_size=0.2, random_state=42
)
timesplit = TimeSeriesSplit(n_splits=5)
for train_index, test_index in timesplit.split(sequences):
    X_train, X_val = sequences[train_index], sequences[test_index]
    y_train, y_val = labels[train_index], labels[test_index]


def sliding_window(
    sequences: np.ndarray, labels: np.ndarray, window_size: int, step_size: int
):
    """
    K-Fold cross validation needs special
    considerations when applied to time-series data.
    Unlike traditional datasets where rows are
    independent, time-series has temporal dependencies.
    The method utilized is sliding window validation
    for this dataset: essentially train on [1,2,3],
    test on [4]. Continue to slide to the next
    variable for the second fold: Train on [2,3,4],
    Test on [5]
    Parameters:
        Sequences: np.ndarray, input sequences
        labels: np.ndarray, labels corresponding
        to the sequences
        window_size: int, size of training
        step_size: int, step size
    Yields:
        X_train, X_val, y_train, y_val
    """
    for start in range(0, len(sequences) - window_size, step_size):
        train_end = start + window_size
        X_train, X_val = (
            sequences[start:train_end],
            sequences[train_end:train_end + step_size],
        )
        y_train, y_val = (
            labels[start:train_end],
            labels[train_end:train_end + step_size],
        )
        yield X_train, X_val, y_train, y_val


# Building LSTM Model
LSTM_model = Sequential(
    [LSTM(50, activation="relu", input_shape=(sequence_length, 1)), Dense(1)]
)
"""
Deciding between optimizers: RMSProp, AdaDelta and Adam algorithms.
Studies have shown that Adam was found to slightly outperform RMSProp.
Adam is generally chosen as the best overall choice for LSTM Models
"""
LSTM_model.compile(optimizer="adam", loss="mse")
# Training the LSTM Model
history = LSTM_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=1,
)
# Evaluating the test loss
test_loss = LSTM_model.evaluate(X_test, y_test, verbose=1)
y_pred = LSTM_model.predict(X_test)

history_df = pd.DataFrame(history.history)
history_output_file = os.path.join(output_dir, "LSTM_History.csv")
history_df.to_csv(history_output_file, index=False)
print(f"Data saved to {history_output_file}")

y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_actual = scaler.inverse_transform(y_pred)

y_test_actual_df = pd.DataFrame(y_test_actual, columns=["Actual"])
y_test_actual_file = os.path.join(output_dir, "LSTM_y_test.csv")
y_test_actual_df.to_csv(y_test_actual_file, index=False)
print(f"Data saved to {y_test_actual_file}")
y_pred_actual_df = pd.DataFrame(y_pred_actual, columns=["Predicted"])
y_pred_actual_file = os.path.join(output_dir, "LSTM_y_pred.csv")
y_pred_actual_df.to_csv(y_pred_actual_file, index=False)
print(f"Data saved to {y_pred_actual_file}")

# Linear Regression Model
cleaned_df = merged_df.drop(
    columns=["Personal consumption expenditures", "2 Yr", "30 Yr"]
)
features = cleaned_df.drop(columns=["Date", "10 Yr"])
original_features = features.columns.tolist()
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
# Performing PCA
pca = PCA()
pca_result = pca.fit_transform(features_scaled)
pca_columns = []
for feature_name in original_features:
    pca_columns.append(f"PC {feature_name}")
pca_df = pd.DataFrame(data=pca_result, columns=pca_columns)
pca_df["10 Yr"] = cleaned_df["10 Yr"].values
pca_df["Date"] = cleaned_df["Date"].values
pca_df = pca_df[["Date", "10 Yr"] + pca_columns]
pca_df_file = os.path.join(output_dir, "PCA_df.csv")
pca_df.to_csv(pca_df_file, index=False)
print(f"Data saved to {pca_df_file}")
# Testing for Heteroskedasticity
X = pca_df.drop(columns=["Date", "10 Yr"])
X = sm.add_constant(X)
y = pca_df["10 Yr"]
model = sm.OLS(y, X).fit()
residuals = model.resid
fitted_values = model.fittedvalues
Lin_Reg_residuals_file = os.path.join(
    output_dir, "Lin_Reg_residuals.csv")
residuals.to_csv(Lin_Reg_residuals_file, index=False)
print(f"Data saved to {Lin_Reg_residuals_file}")
Lin_Reg_fitted_values_file = os.path.join(
    output_dir, "Lin_Reg_fitted_values.csv")
fitted_values.to_csv(Lin_Reg_fitted_values_file, index=False)
print(f"Data saved to {Lin_Reg_fitted_values_file}")
dw = durbin_watson(model.resid)
print(
    f"Since Durbin-Watson statistic is: {round(dw,3)}"
    f"and the value is <1.5, suggests that autocorrelation is present"
)
# adjusting for both autocorrelation and heteroskedasticity, by using HAC...
ols_model = sm.OLS(y, X).fit()
ols_model_hac = ols_model.get_robustcov_results(cov_type="HAC", maxlags=1)
print(ols_model_hac.summary())
# Steps to lower biases even further. 1) transform dependent variable
pca_df["Log(10_Yr)"] = np.log(pca_df["10 Yr"])
y_log = pca_df["Log(10_Yr)"]
X = sm.add_constant(pca_df.drop(columns=["Date", "10 Yr", "Log(10_Yr)"]))
# 2) address autocorrelation via using lagged variables
pca_df_lagged = pca_df.copy()
pca_df_lagged["Lagged_Log(10_Yr)"] = pca_df_lagged["Log(10_Yr)"].shift(1)
pca_df_lagged = pca_df_lagged.dropna()
# Drop unneeded columns
X_lagged = pca_df_lagged.drop(columns=["Date", "10 Yr", "Log(10_Yr)"])
y_lagged = pca_df_lagged["Log(10_Yr)"]
X_lagged = sm.add_constant(X_lagged)
ols_model_lagged = sm.OLS(y_lagged, X_lagged).fit(
    cov_type="HAC", cov_kwds={"maxlags": 1}
)
print(ols_model_lagged.summary())
updated_residuals = ols_model_lagged.resid

residuals_df = pd.DataFrame(updated_residuals, columns=["Residuals"])
lin_reg_residuals_file = os.path.join(
    output_dir, "Updated_Lin_Reg_Residuals.csv")
residuals_df.to_csv(lin_reg_residuals_file, index=False)
print(f"Residuals saved to {lin_reg_residuals_file}")
# Linear Regression ML Model Build
lin_reg = LinearRegression()
X_train_lin_reg = X_train.reshape(X_train.shape[0], -1)
X_test_lin_reg = X_test.reshape(X_test.shape[0], -1)
lin_reg.fit(X_train_lin_reg, y_train)
y_pred_lin_reg = lin_reg.predict(X_test_lin_reg)
lin_mae = mean_absolute_error(y_test, y_pred_lin_reg)
lin_mse = mean_squared_error(y_test, y_pred_lin_reg)
lin_r2 = r2_score(y_test, y_pred_lin_reg)
metrics_lin_reg = {
    "Model": "Linear Regression",
    "MAE": lin_mae,
    "MSE": lin_mse,
    "R2": lin_r2
}
lstm_mae = mean_absolute_error(y_test_actual, y_pred_actual)
lstm_mse = mean_squared_error(y_test_actual, y_pred_actual)
lstm_r2 = r2_score(y_test_actual, y_pred_actual)
metrics_lstm = {
    "Model": "LSTM",
    "MAE": lstm_mae,
    "MSE": lstm_mse,
    "R2": lstm_r2
}
results_df = pd.DataFrame([metrics_lin_reg, metrics_lstm])
results_file = os.path.join(output_dir, "Model_Performance_Comparison.csv")
results_df.to_csv(results_file, index=False)
