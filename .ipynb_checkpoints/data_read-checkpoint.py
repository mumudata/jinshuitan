import pandas as pd
def data_read(file_path):
    """
    读取需要处理的原始数据
    :param file_path:文件的路径
    :return:读取获取得到数据
    """
    pd_data_temperature = pd.read_csv(file_path,\
                                      encoding= 'gb18030')
    return pd_data_temperature

#"/usr/linyuanyuan/jinshuitan_power_plant/temperature_data/jinshuitan_Temperature.csv"