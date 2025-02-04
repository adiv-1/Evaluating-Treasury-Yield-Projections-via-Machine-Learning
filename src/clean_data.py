import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "data", "raw")
output_dir = os.path.join(current_dir, "..", "data", "processed")
os.makedirs(output_dir, exist_ok=True)
TreasuryYield_df = pd.read_csv(os.path.join(data_dir, "Treasury_Data_Raw.csv"))
YF_df = pd.read_csv(os.path.join(data_dir, "YF_Data_Raw.csv"))
FRED_df = pd.read_csv(os.path.join(data_dir, "FRED_Data_Raw.csv"))
BLS_df = pd.read_csv(os.path.join(data_dir, "BLS_Data_Raw.csv"))
BEA_df = pd.read_csv(os.path.join(data_dir, "BEA_Data_Raw.csv"))

LSTM_df = TreasuryYield_df[["Date", "10 Yr"]]
LSTM_df["Pct_Chg"] = LSTM_df[["10 Yr"]].pct_change()
LSTM_df.dropna(inplace=True)
LSTM_output_file = os.path.join(output_dir, "LSTM_Data_Processed.csv")
LSTM_df.to_csv(LSTM_output_file, index=False)
print(f"Data saved to {LSTM_output_file}")

BLS_df.rename(columns={"date": "Date"}, inplace=True)
TreasuryYield_df["Date"] = pd.to_datetime(
    TreasuryYield_df["Date"], errors="coerce")
YF_df["Date"] = pd.to_datetime(YF_df["Date"], errors="coerce")
FRED_df["Date"] = pd.to_datetime(FRED_df["Date"], errors="coerce")
BLS_df["Date"] = pd.to_datetime(BLS_df["Date"], errors="coerce")
BEA_df["Date"] = pd.to_datetime(BEA_df["Date"], errors="coerce")
YF_pivot = YF_df.pivot(index="Date",
                       columns="Ticker Name",
                       values="Adj Return")
YF_pivot.reset_index(inplace=True)
Selected_Treasury_df = TreasuryYield_df[["Date", "2 Yr", "10 Yr", "30 Yr"]]
BEA_df[
    [
        "Residential",
        "Change in private inventories",
        "Personal consumption expenditures",
        "Intellectual property products",
    ]
] = BEA_df[
    [
        "Residential",
        "Change in private inventories",
        "Personal consumption expenditures",
        "Intellectual property products",
    ]
].replace(
    {",": "", "$": "", " ": ""}, regex=True
)
BEA_df[
    [
        "Residential",
        "Change in private inventories",
        "Personal consumption expenditures",
        "Intellectual property products",
    ]
] = BEA_df[
    [
        "Residential",
        "Change in private inventories",
        "Personal consumption expenditures",
        "Intellectual property products",
    ]
].apply(
    pd.to_numeric, errors="coerce"
)
BEA_df.drop("Metric_Name", axis=1, inplace=True)
merged_df = (
    Selected_Treasury_df.merge(YF_pivot, on="Date", how="left")
    .merge(FRED_df, on="Date", how="left")
    .merge(BLS_df, on="Date", how="left")
    .merge(BEA_df, on="Date", how="left")
)
merged_df = merged_df.dropna()
Lin_Reg_output_file = os.path.join(output_dir, "Lin_Reg_Data_Processed.csv")
merged_df.to_csv(Lin_Reg_output_file, index=False)
print(f"Data saved to {Lin_Reg_output_file}")
