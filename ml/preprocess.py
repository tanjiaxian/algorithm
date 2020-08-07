import pandas as pd
import talib


def preprocess(kxData):
    assert isinstance(kxData, pd.DataFrame)
    df = pd.DataFrame()

    df['open'] = kxData['open']
    df['high'] = kxData['high']
    df['low'] = kxData['low']
    df['close'] = kxData['close']
    df['datetime'] = kxData['datetime']
    df['volume'] = kxData['volume']
    df['turnover'] = kxData['turnover']
    df['preClose'] = kxData['preClose']

    # todo 动量
    """ 
        1.作差法求动量值 Momentum_t = P_t - P_t_m  即今天的价格减去一段时间间隔(m期)以前的价格
        2.做除法求动量值 ROC_t = (P_t - P_t_m) / P_t_m
    """
    print(f"MOM: {talib.MOM(df['close'].values, 5)[-5:]}")
    #
    # todo RSI
    print(f"RSI: {talib.RSI(df['close'].values, 5)[-5:]}")
    # todo SMA MACD
    print(f"SMA: {talib.SMA(df['close'].values, 5)[-5:]}")
    print(f"MACD: {talib.MACD(df['close'].values, 12, 26, 9)[-5:]}")
    # todo 唐奇安通道
    donchian_up = talib.MAX(df['high'].values, 5)
    donchian_down = talib.MIN(df['low'].values, 5)

    # todo 布林带通道
    mid = talib.SMA(df['close'].values, 5)
    std = talib.STDDEV(df['close'].values, 5)
    dev = 0.618
    boll_up = mid + std * dev
    boll_down = mid - std * dev

    # todo KDJ
    
    # todo 量价关系 : 价涨量增 价涨量平 价涨量缩 价平量增 价平量缩 价跌量增 价跌量平 价跌量缩
    # todo OBV
    print(f"OBV: {talib.OBV(df['close'].values, df['volume'].values)[-5:]}")


if __name__ == '__main__':
    filepath = '000300.csv'
    kxData = pd.read_csv(filepath)
    preprocess(kxData)
