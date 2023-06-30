#首先对所有的样本数据进行缺失值的填充
def padding_missing_values(data):
    """
    对缺失值进行填充，首先是线性插值的方式，线性插值的方式，其次采用向后填充的方式
    :param data:需要插值的数据
    :return:插值后的数据
    """
    columns_list = list(data.columns)
    for i in columns_list:
        if type(i)==np.float64:
            #首先进行线性插值，但是线性插值针对数组第一个值为空值是不起作用的，需要再利用fillna的方式把后面的值填充到前面
            data[i] = data[i].interpolate()
            data[i] = data[i].fillna(method='bfill')
        else:
            data[i] = data[i].fillna(method='bfill')
    return data