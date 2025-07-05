import pandas as pd

def load_and_merge(train_path, store_path):
    train = pd.read_csv(train_path, parse_dates=['Date'], low_memory=False)
    store = pd.read_csv(store_path)
    
    df = pd.merge(train, store, on='Store', how='left')
    df.sort_values(['Store', 'Date'], inplace=True)
    return df

def clean_data(df):
    df = df[df['Open'] == 1]
    df = df[df['Sales'] > 0]

    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)

    df['StateHoliday'] = df['StateHoliday'].astype(str)

    return df

def feature_engineering(df):
    # Convert date column to datetime if not already
    if df["Date"].dtype != "datetime64[ns]":
        df["Date"] = pd.to_datetime(df["Date"])

    # Extract calendar features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

    # PromoInterval active flag
    df["IsPromoInterval"] = df.apply(lambda row: _is_promo_month(row), axis=1)

    return df

def _is_promo_month(row):
    if pd.isna(row["PromoInterval"]):
        return 0
    month_str = row["Date"].strftime("%b")
    return int(month_str in row["PromoInterval"].split(","))
