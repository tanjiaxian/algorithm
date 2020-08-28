from datetime import datetime
import numpy as np
import pandas as pd
import talib
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def preprocess(kxData):
    assert isinstance(kxData, pd.DataFrame)
    df = pd.DataFrame()

    df['open'] = kxData['open']
    df['high'] = kxData['high']
    df['low'] = kxData['low']
    df['close'] = kxData['close']
    df['volume'] = kxData['volume']
    df['turnover'] = kxData['turnover']
    df['bs'] = kxData['bs']
    # 时间
    kxData['datetime'] = kxData['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    # df['year'] = [t.year for t in kxData['datetime']]
    df['month'] = [t.month for t in kxData['datetime']]
    df['day'] = [t.day for t in kxData['datetime']]
    df['hour'] = [t.hour for t in kxData['datetime']]
    # df['minute'] = [t.minute for t in kxData['datetime']]
    # df['second'] = [t.second for t in kxData['datetime']]

    # todo 动量
    """ 
        1.作差法求动量值 Momentum_t = P_t - P_t_m  即今天的价格减去一段时间间隔(m期)以前的价格
        2.做除法求动量值 ROC_t = (P_t - P_t_m) / P_t_m
    """
    print(f"MOM: {talib.MOM(df['close'].values, 5)[-5:]}")
    df['MOM'] = talib.MOM(df['close'].values, 5)
    #
    # todo RSI
    print(f"RSI: {talib.RSI(df['close'].values, 5)[-5:]}")
    df['RSI'] = talib.RSI(df['close'].values, 5)
    # todo SMA MACD
    print(f"SMA: {talib.SMA(df['close'].values, 5)[-5:]}")
    df['SMA'] = talib.SMA(df['close'].values, 5)
    print(f"MACD: {talib.MACD(df['close'].values, 12, 26, 9)[-1]}")
    DIFF, DEA, MACD = talib.MACD(df['close'].values, 12, 26, 9)
    df["DIFF"] = DIFF
    df['DEA'] = DEA
    df['MACD'] = MACD
    # todo 唐奇安通道
    donchian_up = talib.MAX(df['high'].values, 5)
    donchian_down = talib.MIN(df['low'].values, 5)
    df['donchian_up'] = donchian_up
    df['donchian_down'] = donchian_down
    # todo 布林带通道
    mid = talib.SMA(df['close'].values, 5)
    std = talib.STDDEV(df['close'].values, 5)
    dev = 1 / 0.618
    boll_up = mid + std * dev
    boll_down = mid - std * dev
    df['boll_up'] = boll_up
    df['boll_down'] = boll_down
    # todo KDJ

    # todo 量价关系 : 价涨量增 价涨量平 价涨量缩 价平量增 价平量缩 价跌量增 价跌量平 价跌量缩
    # todo OBV
    print(f"OBV: {talib.OBV(df['close'].values, df['volume'].values)[-5:]}")
    df['obv'] = talib.OBV(df['close'].values, df['volume'].values)

    features = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'month', 'day', 'hour', 'MOM', 'RSI', 'SMA',
                'DIFF', 'DEA', 'MACD', 'donchian_up', 'donchian_down', 'boll_up', 'boll_down', 'obv']
    label = 'bs'
    df = df.dropna(axis=0)
    X = df[features]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # print(scaler.data_max_)
    # print(scaler.data_min_)
    # print(scaler.data_range_)
    # print(scaler.scale_)
    # print(scaler.feature_range)
    y = df.loc[:, label]
    return X, y


if __name__ == '__main__':
    filepath = 'data/' + '000300.csv'
    kxData = pd.read_csv(filepath)
    X, y = preprocess(kxData)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"X_train = {X_train}")
    print(f"y_train = {y_train}")
    model = Perceptron()
    model.fit(X_train, y_train)
    # 0.9311878487811018
    print(model.score(X_test, y_test))


