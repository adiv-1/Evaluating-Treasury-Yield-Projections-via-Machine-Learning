## Instructions to create a conda enviornment
Building the coda environment to ensure dependencies are managed efficiently.
```
conda create -n treasury_env python=3.11 -y cond activate treasury_env
```

## Instructions on how to install the required libraries

Install the python libraries either by installing the requirements.txt file, or by installing the libraries individually:

• Installation via requirements.txt
```
!pip install -r requirements.txt
```
• Manual installation:
```
!pip install pandas numpy matplotlib seaborn scikit-learn statsmodels tensorflow yfinance fred
```
## Instructions on how to download the data
1. Change directory to 'src'
```
cd src
```
2. Run the 'get_data.py' script
```
python get_data.py
```
The data fetching process, first scrapes data from the U.S. Department of Treasury, and then calls on four other data sources (BEA, Yahoo Finance, BLS, and FRED) via calling an API method. This will:
- Scrape treasury yield data from the U.S. Department of Treasury.
- Fetch economic indicators like CPI, GDP, and Federal Funds Rate from FRED.
- Retrieve commodity and financial data from Yahoo Finance.
- Extract labor force participation rate data from BLS.
- Fetch data on private inventories, residential spending, and intellectual property products from BEA.

**PLEASE NOTE**, that the entire data retrieval process can take OVER **10 MINUTES**, but code will work! I strongly advise to replace the API Keys listed in the scripts with your own actual API keys for FRED, BEA, and BLS. However, for the purposes of this project, you may utilize mine listed in the python file for the sole purpose of running the
'get_data.py' script.


## Instructions on how to clean the data

To clean and preprocess the raw data please run the 'clean_data.py' script:
```
python clean_data.py
```
This script will:
- Generate processed datasets for LSTM and Linear Regression models and save cleaned data in the data/processed folder.

## Instructions on how to run analysis code

To train the models and evaluate their performance, please run the 'run_analysis.py' script:
```
python run_analysis.py
```
This code will:
- Train an LSTM model to predict 10-year Treasury yields based
- Train a Linear Regression model using PCA-transformed economic indicators
- Finally, the code will save processed data in the data/processed folder for visualization purposes.

## Instructions on how to create visualizations

**Please Note**: Run this visualization script only AFTER you have run all other scripts sequentially. Please run  'get_data.py' script first, then 'clean_data py' script, and 'run_analysis.py' script in order before performing this step. Thanks!

To generate visualizations, please run the 'visualize_results.py' script:
```
python visualize_ results.py
```
The script will create:
- Time series plots of 10-year Treasury yields and economic indicators.
- Heatmaps of variable correlations before and after PCA.
- Residuals vs. fitted values plot to test for heteroskedasticity.
- Model performance comparison tables.
