from xbbg import blp
import pandas as pd

# PARAMETERS #

ticker = "SPX"
start = "2021-01-01"
end = "2021-02-01"
opt_type = ["C", "P"]
flds = ["PX_BID", "PX_ASK"]
moneyness_variation = 0.1  # for each spot price, generate strike prices between 0.9*spot and 1.1*spot
increment = 5

# PRICE DATA
# Get underlying price data for same time period

df = blp.bdh(ticker + " Index", "PX_Open", start, end, timeout=2000)
if df.empty: raise Exception("DataFrame is empty")
df.columns = df.columns.droplevel(0)
df.index = pd.to_datetime(df.index)
df.index.name = "Date"

# import numpy as np
# df_ind = pd.bdate_range(start, end)
# df = pd.DataFrame(np.random.randint(5000, 6000, len(df_ind)),
#                   index=df_ind, columns=["PX_Open"])
# df.index.name = "Date"
# print(df)

# Create a function to generate strike ranges
def generate_strike_ranges(price):
    bottom = round(price * (1 - moneyness_variation) / increment) * increment
    top = round(price * (1 + moneyness_variation) / increment) * increment
    strike_range = range(bottom, top + increment, increment)
    return list(strike_range)

# Apply the function to each row and create a new column 'Strike_Ranges'
df['Strike_Ranges'] = df["PX_Open"].apply(generate_strike_ranges)

# EXPIRATION DATES GENERATION #

import pandas_market_calendars as mcal
nyse = mcal.get_calendar('NYSE')
early = nyse.schedule(start_date=start,
                      end_date=pd.to_datetime(end) + pd.DateOffset(years=1))
early.index.name = "Date"

opt_mat = early.reset_index()["Date"].to_frame()
opt_mat["DOW"] = opt_mat["Date"].dt.day_name()
opt_mat["next_dow"] = opt_mat["DOW"].shift(-1)
# Get index of all dates where DOW is thursday and next_dow is Monday
thursdays = (opt_mat["DOW"] == "Thursday") & (opt_mat["next_dow"] == "Monday")
# And all fridays
fridays = opt_mat["DOW"] == "Friday"
opt_mat = opt_mat[thursdays | fridays]
opt_mat["Date"] = opt_mat["Date"].dt.strftime("%m/%d/%Y")
opt_mat = opt_mat["Date"]
# print(opt_mat)
# Given a date from market_dates, slice only dates between 20 and 252 days from that date
def get_expiry_dates(ind):
    fil = (opt_mat.index >= (ind + 20)) & (opt_mat.index <= (ind + 252))
    return opt_mat[fil].values

# Clean main dataframe

main_df = df["Strike_Ranges"].reset_index()
main_df["Expiry"] = main_df.apply(lambda x: get_expiry_dates(x.name), axis=1)
# Merge price and futures prices to main_df
df.reset_index(inplace=True)
df.rename(columns={'PX_Open': 'Underlying Price'}, inplace=True)
main_df = pd.merge(main_df, df[["Date", "Underlying Price"]], on="Date", how="left")
# print(main_df)

# Create a function that concats the necessary contract info for bdh parameter
def generate_contracts(index):
    index_df = pd.DataFrame(index=index)
    index_df.reset_index(inplace=True)
    contracts = (ticker + " " + index_df["expiry"] + \
                 " " + index_df["opt_type"] + \
                 index_df["strike"].astype(str) + \
                 " Index").astype(str)
    return contracts.tolist()

# PULL OPTION CHAIN FOR EACH DAY

for _, row in main_df.iterrows():
    print(row["Date"])
    index = pd.MultiIndex.from_product(
        [row["Expiry"], opt_type, row["Strike_Ranges"]],
        names=["expiry", "opt_type", "strike"],
        sortorder=0
    )
    contracts = generate_contracts(index)
    contracts = pd.Series(contracts, name=row["Date"])
    contracts.to_csv(f"{ticker}_{row['Date'].date()}_Option_Chain.csv")
    temp = blp.bdh(contracts, flds, row["Date"], row["Date"], timeout=2000)
    temp.reset_index(drop=True, inplace=True)
    temp = temp.swaplevel(axis=1)
    temp = temp.stack().droplevel(level=0)
    temp.to_pickle(f"{ticker}_{row['Date'].date()}_Option_Chain.pkl")