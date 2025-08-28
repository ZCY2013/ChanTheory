from datetime import datetime

import numpy as np
import pandas as pd
from IPython import embed
from tqdm import tqdm
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from enum import Enum


def get_trading_days(st, et):
    try:
        from my.data.meta_api import get_trading_date_range
        d_lst = get_trading_date_range(int(st), int(et), 'SSE')
        return [i.replace('-','') for i in d_lst]
    except Exception as e:
        d1_lst = [f'2025050{i}' for i in range(6,10)]
        d2_lst = [f'2025051{i}' for i in range(2,6)]
        d_lst = d1_lst + d2_lst
        logging.info("fake trading date return: {}", d_lst)
        return d_lst


class PivotCluster(Enum):
    Top = 1
    Bottom = 0


class Trending(Enum):
    Up = 1
    Down = 0


class TimeDim(Enum):
    MIN_1 = 0
    MIN_5 = 1
    MIN_15 = 2
    MIN_30 = 3
    MIN_60 = 4
    DAY_1 = 5


def load_data(st, et, level=TimeDim.MIN_1):
    """
    :return:
    """
    return_cols = ['high', 'low', 'open', 'close', 'volume']
    days = get_trading_days(int(st), int(et))
    fid = 231668
    ret = []
    for day in tqdm(days[:2], desc='Loading data'):
        try:
            from my.data import quote
            fac = quote.factor(day, 280,fid,0,0)
        except Exception as e:
            logging.info("use local cache data ...")
            fac = pd.read_csv(f'{day}.csv',index_col=0)
            fac['ticker'] = fac.ticker.apply(lambda x: x[2:-1].encode())
        ret.append(fac[(fac.ticker == b'300153') & (fac.exch_time>=92800000) & (fac.exch_time <= 145700000)])

    df = pd.concat(ret)
    cols = ['local_time', 'date', 'int_time', 'high', 'low', 'open','close', 'volume']
    df = df[cols]
    df.reset_index(drop=True, inplace=True)
    df['date_str'] = df['date'].astype(str).apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}")
    # 转换int_time列为时间字符串（HH:MM:SS）
    df['time_str'] = (df['int_time'] / 1000).astype(int).astype(str).str.zfill(6)
    df['time_str'] = df['time_str'].apply(lambda x: f"{x[:2]}:{x[2:4]}:{x[4:6]}")
    # # 合并日期和时间创建datetime列
    df['datetime'] = pd.to_datetime(df['date_str'] + ' ' + df['time_str'])
    df.set_index('datetime', inplace=True)
    df = df[return_cols]
    agg_func = {
        'high': 'max',
        'low': 'min',
        'open': 'first',
        'close': 'last',
        'volume': 'sum'
    }
    if level == TimeDim.MIN_1:
        df.reset_index(inplace=True)
        return df
    elif level == TimeDim.MIN_5:
        df = df.resample('5T')
    elif level == TimeDim.MIN_15:
        df = df.resample('15T')
    elif level == TimeDim.MIN_30:
        df = df.resample('30T')
    elif level == TimeDim.MIN_60:
        df = df.resample('60T')
    elif level == TimeDim.DAY_1:
        df = df.resample('1D')
    else:
        raise RuntimeError("input resample type invalid!")

    df = df.agg(agg_func)
    df.dropna(subset=['open'], inplace=True)

    return df


def preprocess(df):
    # 忽略开盘价收盘价， 只看最高价最低价
    # 判断是否存在包含关系，如果有包含关系则重新定义价格
    trend = None
    df['chan_high'] = df['high']
    df['chan_low'] = df['low']

    for i in range(1, df.shape[0] - 1):
        pre, cur, next = df.iloc[i - 1], df.iloc[i], df.iloc[i + 1]
        if next.high <= pre.high and next.low <= pre.low: # 存在包含关系
            #判断是向上的还是向下的
            if trend is None:
                if cur.high > pre.high and cur.low > pre.low:
                    trend = Trending.Up
                elif cur.high < pre.high and cur.low < pre.low:
                    trend = Trending.Down
                else:
                    trend = None

            if trend is None:
                continue

            func = max if trend == Trending.Up else min

            df.at[i+1, 'chan_high'] = func(cur.high, next.high)
            df.at[i+1, 'chan_low'] = func(cur.low, next.low)
    return df


def fractal(df):
    df['is_Top'] = None
    df['is_Bottom'] = None
    for i in range(1, df.shape[0] - 1):
        pre, cur, next = df.iloc[i - 1], df.iloc[i], df.iloc[i + 1]
        if cur.chan_high >= pre.chan_high and cur.chan_high >= next.chan_high and cur.chan_low >= pre.chan_low and cur.chan_low >= next.chan_low:
            df.at[i,'is_Top'] = True
        if cur.chan_low <= pre.chan_low and cur.chan_low <= next.chan_low and cur.chan_high <= pre.chan_high and cur.chan_high <= next.chan_high:
            df.at[i,'is_Bottom'] = True
        if df.at[i,'is_Top'] and df.at[i,'is_Bottom']:
            print('exception fractal index: ', df.iloc[i].datetime)
            df.at[i, 'is_Top'] = None
            df.at[i, 'is_Bottom'] = None
    return df


def show(df, consolidation_rect=None):
    df = df.copy()
    cols = ['open', 'high', 'low', 'close', 'stroke_price','segments_price']
    df = df.set_index('datetime', drop=True)
    df = df[cols]
    for col in cols:
        df[col] = df[col]/10000

    category_positions = np.arange(len(df))
    time_to_category = {}
    for i, ts in enumerate(df.index):
        time_to_category[ts] = i

    df['stroke_price_line'] = df.stroke_price.astype('float32').interpolate(method='linear')
    df['segments_price_line'] = df.segments_price.astype('float32').interpolate(method='linear')

    # 创建子图（共享X轴）
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    # 1. 添加K线图（蜡烛图）
    fig.add_trace(
        go.Candlestick(
            x=category_positions,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线',
            increasing_line_color='red',  # 上涨蜡烛颜色
            decreasing_line_color='green'  # 下跌蜡烛颜色
        ),
        secondary_y=False
    )
    # 2. 添加折线图（ChanBiPrice）
    fig.add_trace(
        go.Scatter(
            x=category_positions,
            y=df['stroke_price_line'],
            name='笔',
            line=dict(color='blue', width=2),
            mode='lines'
        ),
        secondary_y=False  # 与K线图共用Y轴
    )
    # 2. 添加折线图（ChanBiPrice）
    fig.add_trace(
        go.Scatter(
            x=category_positions,
            y=df['segments_price_line'],
            name='线段',
            line=dict(color='orange', width=2),
            mode='lines'
        ),
        secondary_y=False  # 与K线图共用Y轴
    )
    # 3. 根据笔添加 中枢和趋势
    if consolidation_rect:
        print(f'consolidation size: {len(consolidation_rect)}')
        for rect in consolidation_rect:
            st, et, lower, upper = rect
            st_pos = time_to_category.get(st, min(time_to_category.keys(), key=lambda t:abs((t-st).total_seconds())))
            ed_pos = time_to_category.get(et, min(time_to_category.keys(), key=lambda t:abs((t-et).total_seconds())))

            y_low, y_high = lower, upper
            print(f'consolidation: {st_pos}, {ed_pos}, {y_low}, {y_high}')
            fig.add_shape(
                type='rect',
                x0=st_pos,
                y0=y_low,
                x1=ed_pos,
                y1=y_high,
                line=dict(color='RoyalBlue', width=2),
                fillcolor='rgba(70,130,180,0.3)',
                layer='above',
                xref='x',
                yref='y'
            )
    # 4. 根据线段添加 中枢和趋势
    # 设置图表布局
    fig.update_layout(
        title='ChanTheory',
        yaxis_title='Price ($)',
        xaxis_rangeslider_visible=False,  # 隐藏范围滑块
        width=max(1200, df.shape[0]),  # 对应figratio=(16,9)
        height=max(675,int(df.shape[0] * 9 / 16)),
        template='plotly_white',  # 白色背景风格
        legend=dict(x=0.02, y=0.98),  # 图例位置
        # dragmode='pan' # 默认平移模式
    )
    fig.update_xaxes(
        type='category',  # 禁用自动日期刻度
        # type='date',  # 禁用自动日期刻度
        tickmode='auto',  # 等间距显示标签
        tickvals=category_positions,  # 直接使用数据索引
        ticktext=df.index.strftime('%Y-%m-%d')  # 自定义标签格式
    )
    # fig.update_shapes(dict(xref='x', yref='y'), secondary_y=False)
    # 保存为图片
    # fig.write_image("kline_plotly.png", scale=1.2)  # scale对应figscale
    # 显示图表（可选）
    fig.show(renderer='browser')
    return df


def identify_stroke(df):
    trending = None
    cnt = 0
    idx_lst = []
    df['stroke_price'] = None
    for i in range(1, df.shape[0] - 1):
        if trending is None and not df.iloc[i]['is_Top'] and not df.iloc[i+1]['is_Bottom']:
            continue
        if trending is None:
            trending = Trending.Up if df.iloc[i]['is_Top'] else Trending.Down
            idx_lst.append(i)
            df.at[i, 'stroke_price'] = df.at[i, 'chan_high'] if df.iloc[i]['is_Top'] else df.at[i, 'chan_low']
            cnt = 0
        elif trending == Trending.Up:
            if df.iloc[i]['is_Top']:
                if df.at[i, 'chan_high'] >= df.at[idx_lst[-1], 'chan_high']:
                    df.at[i, 'stroke_price'] = df.at[i, 'chan_high']
                    df.at[idx_lst[-1], 'stroke_price'] = None
                    idx_lst[-1] = i
                    cnt = 0
                else:
                    cnt += 1
            elif df.iloc[i]['is_Bottom'] and cnt >= 4 and df.at[i, 'chan_low'] < df.at[idx_lst[-1], 'chan_high']:
                    df.at[i, 'stroke_price'] = df.at[i, 'chan_low']
                    idx_lst.append(i)
                    trending = Trending.Down
                    cnt = 0
            else:
                cnt += 1

        elif trending == Trending.Down:
            if df.iloc[i]['is_Bottom']:
                if df.at[i, 'chan_low'] < df.at[idx_lst[-1], 'chan_low']:
                    df.at[i, 'stroke_price'] = df.at[i, 'chan_low']
                    df.at[idx_lst[-1], 'stroke_price'] = None
                    idx_lst[-1] = i
                    cnt = 0
                else:
                    cnt += 1
            elif df.iloc[i]['is_Top'] and cnt >= 4 and df.at[i, 'chan_high'] > df.at[idx_lst[-1], 'chan_low']:
                df.at[i, 'stroke_price'] = df.at[i, 'chan_high']
                idx_lst.append(i)
                trending = Trending.Up
                cnt = 0
            else:
                cnt += 1
    return idx_lst


def identify_segments(df, stroke_idx_lst):
    df['segments_price'] = None
    trending = None
    cnt = 0
    idx_lst = []
    i = 0
    while True:
        idx = stroke_idx_lst[i]
        row = df.iloc[idx]
        if trending is None and i >= len(stroke_idx_lst) - 3:
            break
        elif trending is not None and i >= len(stroke_idx_lst) -2:
            idx_lst.append(idx)
            df.at[idx, 'segments_price'] = row.stroke_price
            break

        # print(f"identify date: {row.datetime} seg i:{i}, idx:{idx}, row stroke: {row.stroke_price}", end=' ')
        if trending is None:
            #趋势建立
            justic_idx = i + 3
            # if justic_idx >= len(stroke_idx_lst):
            #     break
            justice_row = df.iloc[stroke_idx_lst[justic_idx]]
            trending = Trending.Down if row['is_Top'] else Trending.Up

            # print(f' new justice i: {justic_idx}, idx:{stroke_idx_lst[justic_idx]}price:{justice_row.stroke_price} treding: {"up" if trending ==Trending.Up else "down"}')
            if trending == Trending.Down:
                # print(justice_row['stroke_price'], row['stroke_price'], i, idx, justic_idx, len(stroke_idx_lst))
                if justice_row['stroke_price'] < row['stroke_price']:
                    idx_lst.append(idx)
                    df.at[idx, 'segments_price'] = row['stroke_price']
                    i = justic_idx
                else:
                    trending = None
                    i += 1
            else:
                if justice_row['stroke_price'] > row['stroke_price']:
                    idx_lst.append(idx)
                    df.at[idx, 'segments_price'] = row['stroke_price']
                    i = justic_idx
                else:
                    trending = None
                    i += 1
        else:
           #趋势延续
            justic_idx = i + 2
            justice_row = df.iloc[stroke_idx_lst[justic_idx]]
            # print(
            # f'along justice i: {justic_idx} idx:{stroke_idx_lst[justic_idx]} price:{justice_row.stroke_price} treding: {"up" if trending == Trending.Up else "down"}')

            if trending == Trending.Down:
                # print(justice_row['stroke_price'], row['stroke_price'], i, idx, justic_idx, len(stroke_idx_lst))
                if justice_row['stroke_price'] < row['stroke_price']:
                    i = justic_idx
                else:
                    idx_lst.append(idx)
                    df.at[idx, 'segments_price'] = row.stroke_price
                    trending = None

            else:
                if justice_row['stroke_price'] > row['stroke_price']:
                    i = justic_idx
                else:
                    idx_lst.append(idx)
                    df.at[idx, 'segments_price'] = row.stroke_price
                    trending = None

    # print('segments idx:', idx_lst)
    return idx_lst


def Consolidation(df, key_price_idx):
    """ s
    实际使用的价格都是strok price
    如何判断构成了中枢： 前三笔
    """
    csd_df = df.iloc[key_price_idx]
    def get_h_l(seg1, seg2):
        if seg1['is_Bottom']:
            return seg2['stroke_price'], seg1['stroke_price']
        elif seg1['is_Top']:
            return seg1['stroke_price'], seg2['stroke_price']
        else:
            raise RuntimeError("Columns Name error in parse stroke")

    i = 0
    ret = []
    while True:
        if i > len(csd_df) - 4:
            break
        seg1, seg2, seg3, seg4 = csd_df.iloc[i], csd_df.iloc[i+1], csd_df.iloc[i+2], csd_df.iloc[i+3]
        h1, l1 = get_h_l(seg1, seg2)
        h2, l2 = get_h_l(seg2, seg3)
        h3, l3 = get_h_l(seg3, seg4)
        upper = min(h1, h2, h3)
        lower = max(l1, l2, l3)
        if upper > lower:
            # todo expanding consolidation
            ret.append((seg1['datetime'], seg4['datetime'], lower/10000, upper/10000))
            i += 3
        else:
            i +=1
    return ret

if __name__ == '__main__':
    """
    TODO:
     1. 增加 附加条件 单独的笔 破坏性长度>=0.618 可以算作一笔线段
     2. 增加 计算 中枢的方法， 先单独实现
     3. 尝试把先计算趋势 在按照高低点 进行趋势升级的方式做抽象
    
    """
    df = load_data(20250501,20250515, TimeDim.MIN_1)
    load_df = df.copy()
    # print('after load:',df )
    pre_df = preprocess(df)
    # print('after preprocess: ', df)
    fractal(df) # 顶底分型
    fractal_df = df.copy()
    stroke_idx = identify_stroke(df) # 笔
    stroke_df = df.copy()
    print("stroke_idx", stroke_idx)
    #根据笔生成线段
    identify_segments(df, stroke_idx)
    segments_df = df.copy()
    # 根据笔定义中枢
    consolidations = Consolidation(df, stroke_idx)
    print("consolidations: ", consolidations)
    # 根据线段定义中枢
    sdf = show(df, consolidations)
    # embed(banner1="chan Theory", user_ns=locals())

