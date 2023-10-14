import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import normaltest, anderson, shapiro, probplot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import pymannkendall as mk

station_id = {  
    724: 'SEMBAWANG', 
    1183: 'VICTORIA DOCK', 
    1248: 'SULTAN SHOAL', 
    1275: 'JURONG', 
    1351: 'RAFFLES LIGHT HOUSE', 
    1534: 'KEPPEL HARBOUR', 
    1746: 'TANJONG PAGAR', 
    1894: 'TUAS (WEST JURONG)', 
    1895: 'WEST COAST', 
    1896: 'WEST TUAS', 
    2032: 'BUKOM', 
    2033: 'TANJONG CHANGI', 
    2034: 'TANAH MERAH', 
    2068: 'UBIN'
}

def station_id_input(id):
    while True:
        try:
            id = int(input("\nPlease enter a Station ID: "))
            if id in station_id:
                print(f"The Station ID entered is {id} with Station Name: {station_id[id]}")
                break
            else:
                print("The entered ID does not exist in the dictionary. Please try again.")
        except ValueError:
            print("Invalid input. Please enter ID with an integer.")
    return id

def sealevel_timeseries(id):
    url = f"https://www.psmsl.org/data/obtaining/rlr.monthly.data/{id}.rlrdata"
    sealevel = pd.read_csv(url, sep='\\s+', usecols=[0, 1], names=['year', 'height'])

    sealevel['year'] = pd.DataFrame(sealevel['year'].str.replace(";",""))
    sealevel[['year', 'year_frac']] = sealevel['year'].str.split('.', expand=True)
    sealevel['year_frac'] = sealevel['year_frac'].str.split('-', expand=True)[0]
    value_month_dict = {417.0:'Jan', 1250.0:'Feb', 2083.0:'Mar', 2917.0:'Apr', 3750.0:'May', 4583.0:'Jun', 
                        5417.0:'Jul', 6250.0:'Aug', 7083.0:'Sep', 7917.0:'Oct', 8750.0:'Nov', 9583.0:'Dec'}     
    sealevel['month'] = pd.to_numeric(sealevel['year_frac'], errors='coerce').map(value_month_dict)
    sealevel['height'] = sealevel['height'].str.split(';', expand=True)[0]
    sealevel['height'] = pd.DataFrame(sealevel['height']).fillna(0).astype(int) 

    sealevel['date'] = sealevel.apply(lambda x: f"{x['year']}-{x['month']}", axis=1)
    sealevel['date'] = pd.to_datetime(sealevel['date'], format='%Y-%b')
    sealevel_df = sealevel[['date', 'height']]

    idx = sealevel_df.set_index('date', inplace=True)
    sealevel_timeseries_df_demo = pd.DataFrame(sealevel_df['height'], idx)

    # For Station ID with missing significant data values
    # Exclude prior date with range of wide gap of NaN values 
    if id == 724:
        idx_keep = sealevel_timeseries_df_demo.iloc[217].name 
        sealevel_timeseries_df_demo = sealevel_timeseries_df_demo[idx_keep:].copy()
    if id == 1351:
        idx_keep = sealevel_timeseries_df_demo.iloc[68].name 
        sealevel_timeseries_df_demo = sealevel_timeseries_df_demo[idx_keep:].copy()
    if id == 1896:
        idx_keep = sealevel_timeseries_df_demo.iloc[144].name 
        sealevel_timeseries_df_demo = sealevel_timeseries_df_demo[idx_keep:].copy()
    if id == 1248:
        idx_keep = sealevel_timeseries_df_demo.iloc[484].name 
        sealevel_timeseries_df_demo = sealevel_timeseries_df_demo[idx_keep:].copy()

    return sealevel_timeseries_df_demo


def sealevel_data_raw(id):

    print(f"Station ID ({id}): {station_id[id]}")
    url = f"https://www.psmsl.org/data/obtaining/rlr.monthly.data/{id}.rlrdata"
    sealevel = pd.read_csv(url, sep='\\s+', usecols=[0, 1], names=['year', 'height'])

    sealevel['year'] = pd.DataFrame(sealevel['year'].str.replace(";",""))
    sealevel[['year', 'year_frac']] = sealevel['year'].str.split('.', expand=True)
    sealevel['year_frac'] = sealevel['year_frac'].str.split('-', expand=True)[0]
    value_month_dict = {417.0:'Jan', 1250.0:'Feb', 2083.0:'Mar', 2917.0:'Apr', 3750.0:'May', 4583.0:'Jun', 
                        5417.0:'Jul', 6250.0:'Aug', 7083.0:'Sep', 7917.0:'Oct', 8750.0:'Nov', 9583.0:'Dec'}     
    sealevel['month'] = pd.to_numeric(sealevel['year_frac'], errors='coerce').map(value_month_dict)
    sealevel['height'] = sealevel['height'].str.split(';', expand=True)[0]
    sealevel['height'] = pd.DataFrame(sealevel['height']).fillna(0).astype(int) #replace na with "0" so that could assign dtype int
    sealevel['date'] = sealevel.apply(lambda x: f"{x['year']}-{x['month']}", axis=1)
    sealevel['date'] = pd.to_datetime(sealevel['date'], format='%Y-%b')
    sealevel_df = sealevel[['date', 'height']]
    
    idx = sealevel_df.set_index('date', inplace=True)
    sealevel_timeseries_df = pd.DataFrame(sealevel_df['height'], idx)

    return sealevel_timeseries_df


def scatter_plot_with_regression_line(df,id):
    assert 'height' in df.columns, "The input DataFrame must have a column named 'height'"
    data = df[['height']].reset_index()
    
    data['date_ordinal'] = pd.to_datetime(data['date']).apply(lambda x: x.toordinal())
    X = data['date_ordinal'].values.reshape(-1, 1)
    y = data['height'].values.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)

    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)
    plt.scatter(data['date'], data['height'], color='b', alpha=0.5)
    plt.plot(data['date'], y_pred, color='r')
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    plt.text(x_min + 0.05 * (x_max - x_min), y_max - 0.1 * (y_max - y_min), f'R^2 = {r2:.2f}')
    slope = reg.coef_[0][0]
    intercept = reg.intercept_[0]
    plt.text(x_min + 0.05 * (x_max - x_min), y_max - 0.15 * (y_max - y_min), f'y = {slope:.2f}x + {intercept:.2f}')
    # Convert from mm/day to mm/year
    rate_of_change = slope * 365.25  
    plt.text(x_min + 0.05 * (x_max - x_min), y_max - 0.20 * (y_max - y_min), f'Sea-Level Rate = {rate_of_change:.2f} mm/year')

    plt.title(f'Relative Sea-Level (RSL) @ {station_id[id]} (ID={id})')
    plt.xlabel('Date')
    plt.ylabel('Relative Sea-Level (RSL), mm')
    plt.show()


def sealevel_data_impute(id):

    print(f"Station ID ({id}): {station_id[id]}")

    url = f"https://www.psmsl.org/data/obtaining/rlr.monthly.data/{id}.rlrdata"
    sealevel = pd.read_csv(url, sep='\\s+', usecols=[0, 1], names=['year', 'height'])

    sealevel['year'] = pd.DataFrame(sealevel['year'].str.replace(";",""))
    sealevel[['year', 'year_frac']] = sealevel['year'].str.split('.', expand=True)
    sealevel['year_frac'] = sealevel['year_frac'].str.split('-', expand=True)[0]
    value_month_dict = {417.0:'Jan', 1250.0:'Feb', 2083.0:'Mar', 2917.0:'Apr', 3750.0:'May', 4583.0:'Jun', 
                        5417.0:'Jul', 6250.0:'Aug', 7083.0:'Sep', 7917.0:'Oct', 8750.0:'Nov', 9583.0:'Dec'}     
    sealevel['month'] = pd.to_numeric(sealevel['year_frac'], errors='coerce').map(value_month_dict)
    sealevel['height'] = sealevel['height'].str.split(';', expand=True)[0]
    sealevel['height'] = pd.DataFrame(sealevel['height']).fillna(0).astype(int) #replace na with "0" so that could assign dtype int
    sealevel['date'] = sealevel.apply(lambda x: f"{x['year']}-{x['month']}", axis=1)
    sealevel['date'] = pd.to_datetime(sealevel['date'], format='%Y-%b')
    sealevel_df = sealevel[['date', 'height']]

    idx = sealevel_df.set_index('date', inplace=True)
    sealevel_timeseries_df = pd.DataFrame(sealevel_df['height'], idx)

    # For Station ID with missing significant data values
    # Exclude prior date with range of wide gap of NaN values 
    if id == 724:
        idx_keep = sealevel_timeseries_df.iloc[217].name 
        sealevel_timeseries_df = sealevel_timeseries_df[idx_keep:].copy()
    if id == 1351:
        idx_keep = sealevel_timeseries_df.iloc[68].name 
        sealevel_timeseries_df = sealevel_timeseries_df[idx_keep:].copy()
    if id == 1896:
        idx_keep = sealevel_timeseries_df.iloc[144].name 
        sealevel_timeseries_df = sealevel_timeseries_df[idx_keep:].copy()
    if id == 1248:
        idx_keep = sealevel_timeseries_df.iloc[484].name 
        sealevel_timeseries_df = sealevel_timeseries_df[idx_keep:].copy()
    
    sealevel_timeseries_df['height'].replace(0, np.nan, inplace=True)
    sealevel_timeseries_df_imputed = sealevel_timeseries_df.assign(RollingMean=sealevel_timeseries_df.height.fillna(sealevel_timeseries_df.height.rolling(12,min_periods=1,).mean()))
    sealevel_timeseries_df_imputed['height'] = sealevel_timeseries_df_imputed['RollingMean']
    # RLR datum at each station is defined to be approximately 7000mm below mean sea level
    sealevel_timeseries_df_final = pd.DataFrame(sealevel_timeseries_df_imputed['height']-7000) 

    return sealevel_timeseries_df_final


def adf_test(series,alpha,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC')    # .dropna() handles differenced data
    
    labels = ['ADF Test Statistic','p-value','# Lags used','# Observations']
    out = pd.Series(result[0:4],index=labels)

    for key,value in result[4].items():
        out[f'Critical Value ({key})'] = value
        
    print(out.to_string())                              # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print(f"At a significance level of {alpha}, has strong evidence against the null hypothesis, we reject the null hypothesis")
        print("Conclude that the time series data has no unit root and is stationary")
    else:
        print(f"At a significance level of {alpha}, has weak evidence against the null hypothesis, we cannot reject the null hypothesis")
        print("Conclude that the time series data has a unit root and is non-stationary")


def kpss_test(timeseries, alpha, title=''):
    print(f'KPSS Test: {title}')
    kpsstest = kpss(timeseries, regression='c', nlags='auto')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','#Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print(kpss_output)
    if kpsstest[1] < alpha:
        print(f"At a significance level of {alpha}, we can reject the null hypothesis and conclude that the sea level time series data is not stationary.")
    else:
        print(f"At a significance level of {alpha}, we cannot reject the null hypothesis and conclude that the sea level time series data is stationary.")


def pp_test(series, alpha, title=''):
    """
    Pass in a time series and an optional title, returns a PP report
    """
    print(f'Phillips-Perron Test: {title}')
    result = PhillipsPerron(series)
    print("\n", result)
 
    if result.pvalue <= alpha:
        print(f"\nAt a significance level of {alpha}, we have strong evidence against the null hypothesis, we reject the null hypothesis, and conclude that the time series data has no unit root and is stationary")
    else:
        print(f"At a significance level of {alpha}, we have weak evidence against the null hypothesis, we fail to reject the null hypothesis, and conclude that the time series data has a unit root and is non-stationary")



def mk_test(series, alpha, title=''):
    """
    Pass in a time series and an optional title, returns a Mann-Kendall Test result
    """
    # Perform the Mann-Kendall trend test
    result = mk.original_test(series)
    print(f'Mann-Kendall Test: {title}')
    # Create a dictionary to map the labels with the result
    result_dict = {
        'Trend\t': result.trend,                #indicates whether there is an increasing, decreasing or no trend in the data.
        'H\t': result.h,                        #result of the hypothesis test. If True, it means that there is a significant trend in the data.
        'P-value\t': round(result.p, 4),        #small p-value indicates strong evidence against the null hypothesis.
        'Z\t': round(result.z, 4),              #standardized test statistic.
        'Tau\t': round(result.Tau, 4),          #Kendall's Tau, a measure of correlation between two variables.
        'Var_s\t': round(result.var_s, 4),      #variance of S, used to calculate the test statistic.
        'Slope\t': round(result.slope, 4),      #slope of the linear regression line fitted to the data.
        'Intercept': round(result.intercept, 4) #intercept of the linear regression line fitted to the data.
    }

    for key, value in result_dict.items():
        print(f"{key}\t:{value}")

    if result.p < alpha:
        if result.slope > 0:
            print(f"\nConclusion:\nAt a significance level of {alpha}, we can reject the null hypothesis and conclude that there is a significant upward trend in the time series data.")
        else:
            print(f"\nConclusion:\nAt a significance level of {alpha}, we can reject the null hypothesis and conclude that there is a significant downward trend in the time series data.")
    else:
        print(f"\nConclusion:\nAt a significance level of {alpha}, we cannot reject the null hypothesis and conclude that there is no significant trend in time series data.")


def pearson_normality_test(series):

    # Perform the D'Agostino and Pearson's test on the series data
    stat, p = normaltest(series)

    # Q-Q plot
    plt.subplot(1, 2, 1)
    probplot(series, plot=plt)
    plt.title('Q-Q Plot')
    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(series, bins=20)
    plt.title('Histogram')
    plt.show()

    print(f'Statistic: {stat:.4f}, p-value: {p:.4f}')

    # Interpret the results
    alpha = 0.05
    if p > alpha:
        print(f'At a significance level of {alpha}, we cannot reject the null hypothesis that the time series data is normally distributed.')
    else:
        print(f'At a significance level of {alpha}, we reject the null hypothesis that the time series data is normally distributed.')


def anderson_darling_test(series):
    # Q-Q plot
    plt.subplot(1, 2, 1)
    probplot(series, plot=plt)
    plt.title('Q-Q Plot')
    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(series, bins=20)
    plt.title('Histogram')
    plt.show()

    # Perform the Anderson-Darling test on the series data
    result = anderson(series)

    print(f'Anderson-Darling statistic: {result.statistic:.4f}')

    # Interpret the results
    for i in range(len(result.critical_values)):
        significance_level, critical_Value = result.significance_level[i], result.critical_values[i]
        if result.statistic < critical_Value:
            print(f'At a significance level of {significance_level/100:.2f}, we cannot reject the null hypothesis that the time series data is normally distributed.')
        else:
            print(f'At a significance level of {significance_level/100:.2f}, we reject the null hypothesis that the time series data is normally distributed.')


def shapiro_wilk_test(series):
    # Q-Q plot
    plt.subplot(1, 2, 1)
    probplot(series, plot=plt)
    plt.title('Q-Q Plot')
    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(series, bins=20)
    plt.title('Histogram')
    plt.show()

    # Perform the Shapiro-Wilk test on the series data
    stat, p = shapiro(series)

    print(f'Statistic: {stat:.4f}, p-value: {p:.4f}')

    # Interpret the results
    alpha = 0.05
    if p > alpha:
        print(f'At a significance level of {alpha}, we cannot reject the null hypothesis that the time series data is normally distributed.')
    else:
        print(f'At a significance level of {alpha}, we reject the null hypothesis that the time series data is normally distributed.')

if __name__ == "__main__": 
    id = station_id_input(id)
    sealevel_timeseries_df = sealevel_data_raw(id)
    sealevel_timeseries_df.to_csv(f"{id}_StationID_Tidegauge.csv")
    print(f"Raw data for Station ID {id} in .csv exported successfully")

    

