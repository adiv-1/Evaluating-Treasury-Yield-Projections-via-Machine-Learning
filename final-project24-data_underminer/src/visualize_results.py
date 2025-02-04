import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = os.path.join(current_dir, "..", "data", "raw")
processed_data_dir = os.path.join(current_dir, "..", "data", "processed")

# 10 Yr Treasury Yield Over Time
filtered_treasury_data = pd.read_csv(
    os.path.join(raw_data_dir, "Treasury_Data_Raw.csv")
)
filtered_treasury_data["Date"] = pd.to_datetime(filtered_treasury_data["Date"])
filtered_treasury_data = filtered_treasury_data.set_index("Date")
temp = filtered_treasury_data["10 Yr"]
temp.plot(title="10 Yr Treasury Yield Over Time")
plt.xlabel("Date")
plt.ylabel("10 Yr Yield")
plt.show()

# Federal Funds Rate, GDP Growth Rate, and CPI Over Time
Fred_data = pd.read_csv(os.path.join(raw_data_dir, "FRED_Data_Raw.csv"))
fig, ax1 = plt.subplots(figsize=(15, 6))
ax1.plot(Fred_data.index,
         Fred_data["Federal Funds Rate"],
         label="Federal Funds Rate")
ax1.plot(Fred_data.index,
         Fred_data["GDP Growth Rate"],
         label="GDP Growth Rate")
ax1.set_xlabel("Date")
ax1.set_ylabel("Federal Funds Rate & GDP Growth Rate")
ax1.legend(loc="upper left")
ax2 = ax1.twinx()
ax2.plot(Fred_data.index,
         Fred_data["CPI"],
         label="CPI",
         color="grey")
ax2.set_ylabel("CPI")
ax2.legend(loc="upper right")
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
plt.title("Federal Funds Rate, GDP Growth Rate, and CPI Over Time")
plt.show()

# Labor Force Participation Rate Over Time
daily_labor_force_df = pd.read_csv(
    os.path.join(raw_data_dir, "BLS_Data_Raw.csv"))
daily_labor_force_df["date"] = pd.to_datetime(daily_labor_force_df["date"])
daily_labor_force_df = daily_labor_force_df.set_index("date")
daily_labor_force_df.plot(title="Labor Force Participation Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Labor Force Participation Rate")
plt.show()

# LSTM Training and Validation Loss Over Each Epoch
history = pd.read_csv(os.path.join(processed_data_dir, "LSTM_History.csv"))
plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("LSTM Training and Validation Loss Over Each Epoch")
plt.show()

# Linear Regression Variable Correlation Heatmap
merged_df = pd.read_csv(
    os.path.join(processed_data_dir, "Lin_Reg_Data_Processed.csv"))
numeric_df = merged_df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="BuPu", fmt=".2f")
plt.title("Linear Regression Variable Correlation Heatmap")
plt.show()

# Linear Regression Variable Correlation Heatmap after PCA
pca_df = pd.read_csv(os.path.join(processed_data_dir, "PCA_df.csv"))
numeric_pca_df = pca_df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_pca_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="BuPu", fmt=".2f")
plt.title("Linear Regression Variable Correlation Heatmap after PCA")
plt.show()

# Residuals vs. Fitted Values Plot to Test for Heteroskedasticity
residuals = pd.read_csv(
    os.path.join(processed_data_dir, "Lin_Reg_residuals.csv"))
fitted_values = pd.read_csv(
    os.path.join(processed_data_dir, "Lin_Reg_fitted_values.csv")
)
# if len(fitted_values) > len(residuals):
#     fitted_values = fitted_values.iloc[:len(residuals)]
# else:
#     residuals = residuals.iloc[:len(fitted_values)]
plt.figure(figsize=(10, 6))
plt.scatter(fitted_values, residuals, alpha=0.7)
plt.axhline(y=0, linestyle="--", color="r")
plt.title("Residuals vs. Fitted Values Plot to Test for Heteroskedasticity")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()

# Linear Regression Residual Histogram Plot
adj_residuals = pd.read_csv(
    os.path.join(processed_data_dir, "Updated_Lin_Reg_Residuals.csv")
)
plt.figure(figsize=(10, 6))
sns.histplot(adj_residuals, kde=True, bins=30, color="blue")
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Model Performance Comparison Table
model_performance_df = pd.read_csv(
    os.path.join(processed_data_dir, "Model_Performance_Comparison.csv")
)
plt.figure(figsize=(6, 3))
plt.table(
    cellText=model_performance_df.round(3).values,
    colLabels=model_performance_df.columns,
    loc="center",
    cellLoc="center",
)
plt.title("Model Performance Comparison Table")
plt.axis("off")
plt.show()
