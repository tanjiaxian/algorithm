import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def preprocess(kxData):
    assert isinstance(kxData, pd.DataFrame)
    df = pd.DataFrame()

    df['open'] = kxData['open']
    df['high'] = kxData['high']
    df['low'] = kxData['low']
    df['close'] = kxData['close']
    df['volume'] = kxData['volume']
    df['turnover'] = kxData['turnover']
    df['preClose'] = kxData['preClose']

    # 时间
    kxData['datetime'] = kxData['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    df['year'] = [t.year for t in kxData['datetime']]
    df['month'] = [t.month for t in kxData['datetime']]
    df['day'] = [t.day for t in kxData['datetime']]
    df['hour'] = [t.hour for t in kxData['datetime']]
    df['minute'] = [t.minute for t in kxData['datetime']]
    df['second'] = [t.second for t in kxData['datetime']]

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
    dev = 0.618
    boll_up = mid + std * dev
    boll_down = mid - std * dev
    df['boll_up'] = boll_up
    df['boll_down'] = boll_down
    # todo KDJ

    # todo 量价关系 : 价涨量增 价涨量平 价涨量缩 价平量增 价平量缩 价跌量增 价跌量平 价跌量缩
    # todo OBV
    print(f"OBV: {talib.OBV(df['close'].values, df['volume'].values)[-5:]}")
    df['obv'] = talib.OBV(df['close'].values, df['volume'].values)




    print(df.dtypes)




if __name__ == '__main__':
    filepath = '000300.csv'
    kxData = pd.read_csv(filepath)
    preprocess(kxData)
