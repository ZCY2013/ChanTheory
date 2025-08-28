from my.data import quote
from my.data import quote
from my.data.meta_api import get_trading_date_range
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


day = 20250528
def get_trading_days(st, et):
    d_lst = get_trading_date_range(int(st), int(et), 'SSE')
    return [i.replace('-', '') for i in d_lst]


day_list = np.array([int(d) for d in get_trading_days(day - 10000, day + 10000)])
pre_day = day_list[day_list < day][-1]
tick_used_cols = ['ticker','local_time', 'exch_time', 'pre_close_px','open_px','high_px', 'low_px', 'last_px', 'ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'av1', 'av2', 'av3', 'av4','av5',
                  'bp1', 'bp2', 'bp3', 'bp4', 'bp5', 'bv1', 'bv2', 'bv3', 'bv4', 'bv5', 'num_of_trades', 'total_vol', 'total_notional',
                  'total_bid_vol', 'total_ask_vol', 'upper_limit_px', 'lower_limit_px']
agg_dict = {
    'ticker': 'first',
    'exch_time': 'first',
    'local_time': 'first',
    'pre_close_px': 'first',
    'open_px':'first',
    'high_px': 'max',
    'low_px': 'min',
    'last_px': 'last',
    'num_of_trades':'last',
    'total_vol': lambda x: x.iloc[-1] if len(x) >0 else np.nan,
    'total_notional':lambda x: x.iloc[-1] if len(x) >0 else np.nan,
    'total_bid_vol': lambda x: x.iloc[-1] if len(x) >0 else np.nan,
    'total_ask_vol': lambda x: x.iloc[-1] if len(x) >0 else np.nan,
    'upper_limit_px': 'first',
    'lower_limit_px': 'first'
}
agg_dict.update({f'{j}{i}':'last' for i in range(1,6) for j in['ap','av','bp', 'bv']})

day_tick = pd.DataFrame(quote.data(pre_day, 209, 'all',0,0, flatten=True))
grouped = day_tick.groupby('ticker')

group_list = []
for t, gdf in tqdm(grouped):
    gdf = gdf[tick_used_cols]
    gdf['DateTime'] = pd.to_datetime(gdf.local_time / 1e6, unit='s', utc=True).apply(
        lambda x: x.tz_convert('Asia/Shanghai'))
    gdf.set_index('DateTime', drop=True, inplace=True)
    open_call_ = gdf[gdf['exch_time'] < 93000000][-1:]
    close_call_ = gdf[gdf['exch_time'] > 145700000][-1:]
    continues_ = gdf[(gdf.exch_time >= 93000000) & (gdf.exch_time <= 145700000)]
    continues_ = continues_.resample('T').agg(agg_dict).dropna()
    day_df = pd.concat([open_call_, continues_, close_call_])

    day_df['price_direction'] = (day_df.last_px - day_df.last_px.shift(1)).apply(lambda x: np.sign(x))
    day_df['min_trade_amount'] = day_df.total_notional - day_df.total_notional.shift(1)
    group_list.append(day_df)
df = pd.concat(group_list)