import pandas as pd
import requests
from bs4 import BeautifulSoup
from fredapi import Fred
import yfinance as yf
import warnings
from pybea.client import BureauEconomicAnalysisClient as BEAC
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "..", "data", "raw")
os.makedirs(output_dir, exist_ok=True)
start_date = "1990-01-01"
end_date = "2024-12-31"


# Treasury Yield Data Retrieval
def scrape_treasury_yield(year: int) -> pd.DataFrame:
    """
    This function helps scrape U.S. treasury par-value yield
    curve data from U.S. department of treasury website.
    arguments:
        year(int): Based on the URL structure, the inputted argument will
        assist in retrieving the data for the corresponding year.
    returns:
        pd.DataFrame: The returned argument will be a dataframe
        of the retrieved treasury data.
        If no data is found, empty dataframe will be returned.
    raises:
        HTTPError if request is faulty
    """
    url = (
        f"https://home.treasury.gov/"
        f"resource-center/data-chart-center/interest-rates/"
        f"TextView?type=daily_treasury_yield_curve&field_tdr_date_value={year}"
    )
    r = requests.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "html.parser")
    table = soup.find("table")
    if not table:
        print(f"No table found in {year}")
        return pd.DataFrame()
    headers = []
    for th in table.find_all("tr")[0].find_all("th"):
        headers.append(th.text.strip())
    data = []
    rows = table.find_all("tr")[1:]
    for row in rows:
        cols = []
        for td in row.find_all("td"):
            cols.append(td.text.strip())
        data.append(cols)
    df = pd.DataFrame(data, columns=headers)
    return df


treasury_data = pd.DataFrame()
for year in range(1990, 2025):
    yearly_data = scrape_treasury_yield(year)
    treasury_data = pd.concat([treasury_data, yearly_data], ignore_index=True)
selected_columns = [
    "Date",
    "1 Mo",
    "2 Mo",
    "3 Mo",
    "4 Mo",
    "6 Mo",
    "1 Yr",
    "2 Yr",
    "3 Yr",
    "5 Yr",
    "7 Yr",
    "10 Yr",
    "20 Yr",
    "30 Yr",
]
filtered_treasury_data = treasury_data[selected_columns]
filtered_treasury_data["Date"] = pd.to_datetime(filtered_treasury_data["Date"])
filtered_treasury_data["Date"] = filtered_treasury_data["Date"].dt.strftime(
    "%m/%d/%Y")
filtered_treasury_data["10 Yr"] = pd.to_numeric(
    filtered_treasury_data["10 Yr"], errors="coerce"
)
treasury_output_file = os.path.join(output_dir, "Treasury_Data_Raw.csv")
filtered_treasury_data.to_csv(treasury_output_file, index=False)
print(f"Data saved to {treasury_output_file}")
# FRED Data Retrieval
fred = Fred(api_key=INSERT_API_KEY)
cpi_data = fred.get_series("CPIAUCSL",
                           start_date=start_date,
                           end_date=end_date)
fed_funds_rate_data = fred.get_series("DFF",
                                      start_date=start_date,
                                      end_date=end_date)
gdp_growth_rate_data = fred.get_series("A191RL1Q225SBEA",
                                       start_date=start_date,
                                       end_date=end_date)
cpi_data.name = "CPI"
fed_funds_rate_data.name = "Federal Funds Rate"
gdp_growth_rate_data.name = "GDP Growth Rate"
# Forward Fill
cpi_data = cpi_data.resample("D").ffill().loc[start_date:end_date]
gdp_growth_rate_data = (
    gdp_growth_rate_data.resample("D").ffill().loc[start_date:end_date]
)
fed_funds_rate_data = fed_funds_rate_data.resample(
    "D").ffill().loc[start_date:end_date]
Fred_data = pd.concat([fed_funds_rate_data,
                       cpi_data,
                       gdp_growth_rate_data], axis=1)
Fred_data.index = Fred_data.index.strftime("%m/%d/%Y")
Fred_data.insert(0, "Date", Fred_data.index)
fred_output_file = os.path.join(output_dir, "FRED_Data_Raw.csv")
Fred_data.to_csv(fred_output_file, index=False)
print(f"Data saved to {fred_output_file}")
# YahooFinance Data Retrieval
warnings.filterwarnings("ignore")
energy = pd.read_html(
    "https://en.wikipedia.org/wiki/List_of_traded_commodities")[4]
energy["Symbol"] = energy["Symbol"].str.replace(".", "-")
agricultural = pd.read_html(
    "https://en.wikipedia.org/wiki/List_of_traded_commodities")[1]
agricultural["Symbol"] = agricultural["Symbol"].str.replace(".", "-")
metals = pd.read_html(
    "https://en.wikipedia.org/wiki/List_of_traded_commodities")[7]
metals["Symbol"] = metals["Symbol"].str.replace(".", "-")
symbols_list = ["^GSPC", "ZC=F", "CL=F", "GC=F"]
yf_df = yf.download(tickers=symbols_list, start=start_date, end=end_date)
adj_close_df = yf_df["Adj Close"].stack().reset_index()
adj_close_df.columns = ["Date", "Ticker", "Adj Close"]
ticker_names = {
    "^GSPC": "S&P 500",
    "ZC=F": "Corn Futures",
    "CL=F": "Crude Oil Futures",
    "GC=F": "Gold Futures",
}
adj_close_df["Ticker Name"] = adj_close_df["Ticker"].map(ticker_names)
adj_close_df["Adj Return"] = adj_close_df.groupby(
    "Ticker")["Adj Close"].pct_change()
yf_output_file = os.path.join(output_dir, "YF_Data_Raw.csv")
adj_close_df.to_csv(yf_output_file, index=False)
print(f"Data saved to {yf_output_file}")
# BEA Data Retrieval
BEA_API_KEY = INSERT_API_KEY
bea_client = BEAC(api_key=BEA_API_KEY)
dataset_list = bea_client.get_dataset_list()
url = "https://apps.bea.gov/api/data/"
dataset_name = "NIPA"
table_name = "T10106"
frequency = "Q"
start_year = 1990
end_year = 2025
combined_df = pd.DataFrame()
for year in range(start_year, end_year):
    params = {
        "UserID": BEA_API_KEY,
        "method": "GetData",
        "datasetname": dataset_name,
        "TableName": table_name,
        "Frequency": frequency,
        "Year": str(year),
        "ResultFormat": "JSON",
    }
    response = requests.get(url, params=params)
    raw_data = response.json()
    if "Data" in raw_data.get("BEAAPI", {}).get("Results", {}):
        data_section = raw_data["BEAAPI"]["Results"]["Data"]
        df = pd.DataFrame(data_section)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    else:
        print(f"No data found for year {year}")
BEA_filtered_df = combined_df[
    combined_df["LineDescription"].isin(
        [
            "Residential",
            "Change in private inventories",
            "Personal consumption expenditures",
            "Intellectual property products",
        ]
    )
]
BEA_pivoted_df = BEA_filtered_df.pivot_table(
    index="TimePeriod",
    columns="LineDescription",
    values="DataValue",
    aggfunc="first"
).reset_index()
BEA_pivoted_df["Date"] = pd.to_datetime(
    BEA_pivoted_df["TimePeriod"].str.replace("Q", "-"), format="%Y-%m"
)
BEA_pivoted_df = BEA_pivoted_df.set_index("Date")
BEA_daily_df = BEA_pivoted_df.resample("D").ffill()
BEA_daily_df = BEA_daily_df.reset_index()
BEA_daily_df["Date"] = BEA_daily_df["Date"].dt.strftime("%Y/%m/%d")
BEA_daily_df["Metric_Name"] = "Chained Dollars"
BEA_final_df = BEA_daily_df[
    [
        "Date",
        "Residential",
        "Change in private inventories",
        "Personal consumption expenditures",
        "Intellectual property products",
        "Metric_Name",
    ]
]
BEA_final_csv_path = os.path.join(output_dir, "BEA_Data_Raw.csv")
BEA_final_df.to_csv(BEA_final_csv_path, index=False)
print(f"Data saved to {BEA_final_csv_path}")
# BLS Data Retrieval
BLS_API_KEY = INSERT_API_KEY
url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
series_id = "LNS11300000"
labor_force_df = pd.DataFrame()
# Loop smaller year ranges (initial full range sweep didn't work)
for start_year, end_year in [(1990, 2000), (2001, 2010), (2011, 2025)]:
    params = {
        "seriesid": [series_id],
        "startyear": str(start_year),
        "endyear": str(end_year),
        "registrationkey": BLS_API_KEY,
    }
    response = requests.post(url, json=params)
    data = response.json()
    if "Results" in data and "series" in data["Results"]:
        series_data = data["Results"]["series"][0]["data"]
        temp_df = pd.DataFrame(series_data)
        labor_force_df = pd.concat([labor_force_df,
                                    temp_df], ignore_index=True)
    else:
        print(f"No data found for range {start_year}-{end_year}")
labor_force_df = labor_force_df[labor_force_df["period"] != "M13"]
labor_force_df["date"] = pd.to_datetime(
    labor_force_df["year"]
    + "-"
    + labor_force_df["period"].str.replace("M", ""),
    format="%Y-%m",
)
labor_force_df = labor_force_df.sort_values("date")
labor_force_df["value"] = pd.to_numeric(
    labor_force_df["value"], errors="coerce")
labor_force_df = labor_force_df[["date", "value"]].rename(
    columns={"value": "Labor Force Participation Rate"}
)
labor_force_df = labor_force_df.set_index("date")
daily_labor_force_df = labor_force_df.resample("D").ffill()
daily_labor_force_df = daily_labor_force_df.reset_index()
daily_labor_force_df["date"] = daily_labor_force_df["date"].dt.strftime(
    "%Y/%m/%d")
BLS_final_csv_path = os.path.join(output_dir, "BLS_Data_Raw.csv")
daily_labor_force_df.to_csv(BLS_final_csv_path, index=False)
print(f"Data saved to {BLS_final_csv_path}")
