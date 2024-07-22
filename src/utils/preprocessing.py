import pandas as pd

def sort_extract_price(data, commodity):
    df = data.copy()
    df = df[df['commodity'].str.contains(commodity, case=False, na=False)]
    df = df[df['pricetype'] == 'Retail']
    df['date'] = pd.to_datetime(df['date'])
    df['usdprice'] = pd.to_numeric(df['usdprice'])
    df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
    df_avg = df.groupby('date')['usdprice'].mean().round(3).reset_index()
    return df_avg