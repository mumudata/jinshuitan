import numpy as np
import pandas as pd
#查看第几次开机的时间，并且对数据规整化
def select_start_time(i,equipment,pd_data_temperature):
    """
    选取温度的相关数据，可以选取那种设备部位以及第几次开机的数据
    :param i:第几次开机
    :param equipment:设备的名称
    :param pd_data_temperature:所有的温度数据
    :return:处理好的温度传感器数据
    """
    #获取数据框的字段
    columns_list = list(pd_data_temperature.columns)
    #选择第几次开机的数据
    pd_data_temperature = pd_data_temperature[pd_data_temperature["第几次开机"].isin(i)]
    #选择机械设备的某个部位
    judge_list = [equipment in y for y in columns_list]
    judge_result = [i for i,x in enumerate(judge_list) if x == True]
    judge_result_list = [columns_list[x] for x in judge_result]
    judge_result_list.insert(0,'time')
    judge_result_list.insert(1,'第几次开机')
    pd_data_temperature_part = pd_data_temperature[judge_result_list]
    #填充缺失值
    pd_data_temperature_part = pd_data_temperature_part.fillna(method='bfill')
    pd_data_temperature_part = pd_data_temperature_part.reset_index()
    pd_data_temperature_part = pd_data_temperature_part.sort_values(by=['time'])
    pd_data_temperature_part =  pd_data_temperature_part[judge_result_list]
    #对其中的日期进行截取
    pd_data_temperature_part["time_day"] = pd_data_temperature_part["time"].apply(lambda x :x[0:10])
    return pd_data_temperature_part

#滑动窗口获取数据，而不是按照次数的方式获取数据
def slide_window_get_data(start_time,p,equipment,pd_data_temperature):
    """
    选取温度的相关数据，可以选取那种设备部位以及第几次开机的数据
    :param i:第几次开机
    :param equipment:设备的名称
    :param pd_data_temperature:所有的温度数据
    :return:处理好的温度传感器数据
    """
    #获取数据框的字段
    columns_list = list(pd_data_temperature.columns)
    #选择机械设备的某个部位
    judge_list = [equipment in y for y in columns_list]
    judge_result = [i for i,x in enumerate(judge_list) if x == True]
    judge_result_list = [columns_list[x] for x in judge_result]
    #在名字中添加
    judge_result_list.insert(0,'time')
    pd_data_temperature_part = pd_data_temperature[judge_result_list]
    #填充缺失值
    pd_data_temperature_part = pd_data_temperature_part.fillna(method='bfill')
    pd_data_temperature_part = pd_data_temperature_part.reset_index()
    pd_data_temperature_part = pd_data_temperature_part.sort_values(by=['time'])
    pd_data_temperature_part =  pd_data_temperature_part[judge_result_list]
    #对其中的日期进行截取
    pd_data_temperature_part["time_day"] = pd_data_temperature_part["time"].apply(lambda x :x[0:10])
    #按照一定的时间范围进行数据的截取
    pd_data_temperature_part = pd_data_temperature_part[start_time:start_time+p]
    return pd_data_temperature_part

def calc_euclidean(actual, predic):
    """
    分别计算任意两个曲线之间的欧氏距离
    :param actual:变量1
    :param predic:变量2
    :return:欧式距离
    """
    return np.sqrt(np.sum((actual - predic) ** 2))


def calc_mape(actual, predic):
    """
    计算任意曲线之间的MAPE
    :param actual:变量1
    :param predic:变量2
    :return:MAPE
    """
    return np.mean(np.abs((actual - predic) / (actual+0.1)))


def calc_correlation(actual, predic):
    """
    计算任意曲线之间的皮尔逊相关系数
    :param actual:变量1
    :param predic:变量2
    :return:皮尔逊相关系数
    """
    a_diff = actual - np.mean(actual)
    p_diff = predic - np.mean(predic)
    numerator = np.sum(a_diff * p_diff)
    denominator = np.sqrt(np.sum(a_diff ** 2)) * np.sqrt(np.sum(p_diff ** 2))
    return numerator / denominator


def tep_calc(pd_data_temperature_part):
    """
    计算任意两个温度传感器之间的欧式距离、MAPE、皮尔逊相关系数
    :param actual:所有的温度传感器数据
    :return:任意两个温度传感器的欧式距离、皮尔逊相关系数、MAPE
    """
    part_list = list(pd_data_temperature_part.columns)
    part_list_copy = part_list.copy()
    corr_all_euclidean_list = []
    corr_all_mape_list = []
    corr_all_correlation_list = []
    for i in part_list:
        corr_euclidean_list = []
        corr_mape_list = []
        corr_correlation_list = []
        part_list_copy.remove(i)
        for j in part_list_copy:
            a = calc_euclidean(pd_data_temperature_part[i],pd_data_temperature_part[j])
            b = calc_mape(pd_data_temperature_part[i],pd_data_temperature_part[j])
            c = calc_correlation(pd_data_temperature_part[i],pd_data_temperature_part[j])
            corr_euclidean_list.append(a)
            corr_mape_list.append(b)
            corr_correlation_list.append(c)
        part_list_copy = part_list.copy()
        corr_all_euclidean_list.append(corr_euclidean_list)
        corr_all_mape_list.append(corr_mape_list)
        corr_all_correlation_list.append(corr_correlation_list)
    return corr_all_euclidean_list,corr_all_mape_list,corr_all_correlation_list


def columns_get(pd_data_temperature_part,columns):
    """
    获取固定字段的温度传感器数据
    :param pd_data_temperature_part:所有的温度传感器数据
    :param columns:字段数据
    :return:温度传感器数据
    """
    judge_list = [columns in y for y in list(pd_data_temperature_part.columns)]
    judge_result = [i for i,x in enumerate(judge_list) if x == True]
    judge_result_list = [list(pd_data_temperature_part.columns)[x] for x in judge_result]
    judge_result_list.insert(0,'time')
    pd_data_temperature_part = pd_data_temperature_part[judge_result_list]
    return pd_data_temperature_part


#首先进行一个部位进行测试,获取某一个部位的数据
def location_data_acquisition(i,j,columns,pd_data_temperature):
    """
    获取某个机械设备不同开机次数的数据
    :param i:第i次开机
    :param j:第j次开机
    :param columns:获取其中的某一个部位
    :return:获取得到最后某个部位的温度数据
    """
    pd_data_temperature_part = select_start_time([x for x in range(i,j)],columns,pd_data_temperature)
    pd_data_temperature_part = columns_get(pd_data_temperature_part,columns)
    return pd_data_temperature_part


#基于滑窗的方式获取每一个部位的数据
def location_data_acquisition_windows(start_time,p,columns):
    """
    获取某个机械设备不同开机次数的数据
    :param i:第i次开机
    :param j:第j次开机
    :param columns:获取其中的某一个部位
    :return:获取得到最后某个部位的温度数据
    """
    pd_data_temperature_part = slide_window_get_data(start_time,p,columns,pd_data_temperature)
    pd_data_temperature_part = columns_get(pd_data_temperature_part,columns)
    return pd_data_temperature_part


def parameter_calculation(pd_data_temperature_part):
    """
    计算同一个部位不同传感器温度数据之间的pearson相关系数和欧式距离，以及相关系数和欧式距离的平均值。
    :param i:第i次开机
    :param j:第j次开机
    :param columns:获取其中的某一个部位
    :return:获取得到最后某个部位的温度数据
    """
    corr_all_list = tep_calc(pd_data_temperature_part)
    corr_euclidean_mean = [np.mean(x) for x in corr_all_list[0]]
    corr_mape_mean = [np.mean(x) for x in corr_all_list[1]]
    corr_correlation_mean = [np.mean(x) for x in corr_all_list[2]]

    #整合所有的数据到一个数据框
    data_pd_anomaly_detection = pd.DataFrame({"Mechanical_part":list(pd_data_temperature_part.columns),
                                             "corr_euclidean_mean":corr_euclidean_mean,
                                             "corr_correlation_mean":corr_correlation_mean
                                             })

    data_pd_anomaly_detection.sort_values(by=['corr_euclidean_mean'],ascending=True)
    return data_pd_anomaly_detection


#获取得到典型的温度曲线
def typical_temperature_curve(i,p,columns,pd_data_temperature_part):
    """
    计算所有传感器中top距离最近的传感器，并且在top3中选择pearson相关系数最大的一个传感器
    :i:第几次开机
    :p:间隔开机的次数
    :return:获取得到典型温度曲线的那一个触感器
    """
    pd_data_temperature_part = location_data_acquisition(i,i+p,columns,pd_data_temperature_part)
    #获取计算的
    columns = list(set(pd_data_temperature_part.columns)-set(['time']))
    pd_data_temperature_part = pd_data_temperature_part[columns]
    parameter_calculation_pd = parameter_calculation(pd_data_temperature_part)
    
    #首先按欧式距离对数据框进行升序排序
    data_pd_anomaly_detection = parameter_calculation_pd.sort_values(by=['corr_euclidean_mean'])
    #获取得到top3的数据，距离作为优先选项，在最相似的三个传感器中选取persion最相似的传感器
    data_pd_anomaly_detection = data_pd_anomaly_detection.iloc[:3,:]
    #获取得到相关系数最大的一个字段名称
    data_pd_anomaly_detection=data_pd_anomaly_detection[data_pd_anomaly_detection["corr_correlation_mean"]==max(data_pd_anomaly_detection["corr_correlation_mean"])]
    columns = data_pd_anomaly_detection['Mechanical_part'].values[0]                                                   
    return columns


def mad_based_outlier(parameter_calculation_pd,columns,label,thresh=3.5):
    """
    基于MAD的方式计算异常值的数据
    :param parameter_calculation_pd:
    :param columns:
    :return:获取得到典型温度曲线的那一个触感器
    """
    points = parameter_calculation_pd[columns].values
    if type(points) is list:
        points = np.asarray(points)
    if len(points.shape) == 1:
        points = points[:, None]
    med = np.median(points, axis=0)
    abs_dev = np.absolute(points - med)
    med_abs_dev = np.median(abs_dev)
    
    MAD_down = med-thresh*med_abs_dev/0.6745
    MAD_up = med+thresh*med_abs_dev/0.6745
    columns_label = ["Mechanical_part",columns]
    parameter_calculation_pd =  parameter_calculation_pd[columns_label]
    if label == "distince":
        parameter_calculation_pd["MAD_up"] = MAD_up[0]
        Mechanical_part_list = parameter_calculation_pd[parameter_calculation_pd[columns]>parameter_calculation_pd["MAD_up"]]           ["Mechanical_part"].values
    if label == "correlation":
        parameter_calculation_pd["MAD_down"] = MAD_down[0]
        Mechanical_part_list = parameter_calculation_pd[parameter_calculation_pd[columns]<parameter_calculation_pd["MAD_down"]]["Mechanical_part"].values
    return Mechanical_part_list,parameter_calculation_pd

#传感器异常识别异常结果
def sensor_abnormal_recognition_result(start_time,p,equipment):
    """
    传感器异常识别异常结果
    :param i:第几次开机
    :param p:下一次开机的的时间间隔
    :param equipment:需要检测的部位名称
    :return:检测的结果
    """
    # pd_data_temperature_part = location_data_acquisition(i,i+q,equipment)
    
    pd_data_temperature_part = location_data_acquisition_windows(start_time,p,equipment)
    columns = list(set(pd_data_temperature_part.columns)-set(['time']))
    pd_data_temperature_part_i = pd_data_temperature_part[columns]
    parameter_calculation_pd = parameter_calculation(pd_data_temperature_part_i)

    parameter_calculation_euclidean = mad_based_outlier(parameter_calculation_pd,"corr_euclidean_mean","distince",thresh=15)
    mad_euclidean = parameter_calculation_euclidean[0]
    index_euclidean = parameter_calculation_euclidean[1]
    
    parameter_calculation_correlation = mad_based_outlier(parameter_calculation_pd,"corr_correlation_mean","correlation",thresh=40)
    mad_correlation = parameter_calculation_correlation[0]
    index_correlation = parameter_calculation_correlation[1]

    print(judge_condition(mad_euclidean,mad_correlation,index_euclidean,index_correlation))
    plot(pd_data_temperature_part,equipment)
    return


def judge_condition(mad_euclidean,mad_correlation,index_euclidean,index_correlation):
    """
    判断异常的触感器，并且输出异常的类型
    :param mad_euclidean:欧式距离的平均值
    :param mad_correlation:相关系数的平均值
    :param index_euclidean:评估异常传感器的欧氏距离指标
    :param index_correlation:评估异常传感器的persion相关系数指标
    :return:是否有坏的传感器并且返回异常的类型
    """
    if len(mad_euclidean)==0 and len(mad_correlation)==0:
        return ("没有异常传感器,\
                \n距离指标为:\n{},\
                \n\n相关系数指标为:\n{}".format(index_euclidean,index_correlation))
    elif len(mad_euclidean)!=0 and len(mad_correlation)==0:
        return ("该传感器相比其它传感器距离相对比较远，但是趋势走势无异常，具体的触感器为\n{},\
                \n距离指标为:\n{},\
                \n\n相关系数指标为:{}".format(mad_euclidean,index_euclidean,index_correlation))
    elif len(mad_euclidean)==0 and len(mad_correlation)!=0:
        return ("该传感器距离其它传感器接近，但是趋势走势异常，具体的传感器为{}疑似为异常传感器,\
                \n距离指标为:\n{},\
                \n\n相关系数指标为:\n{}".format(mad_correlation,index_euclidean,index_correlation))
    else:
        a = [x for x in mad_euclidean if x not in mad_correlation]  #在list1列表中而不在list2列表中
        b = [y for y in mad_correlation if y not in mad_euclidean]  #在list2列表中而不在list1列表中
        c = [z for z in mad_euclidean if z in mad_correlation]  #在list1列表中ye1在list2列表中
        if len(c)!=0:
            return ("该传感器疑似异常传感器{},\
                    \n距离指标为:\n{},\
                    \n\n相关系数指标为:\n{}".format(c,index_euclidean,index_correlation))
        if len(b)!=0:
            return ("该传感器距离其它传感器接近，但是趋势走势异常，具体的传感器为{},\
                    \n距离指标为:\n{},\
                    \n\n相关系数指标为:\n{}".format(b,index_euclidean,index_correlation))
        if len(a)!=0:
            return ("该传感器相比其它传感器距离相对比较远，但是趋势走势无异常，具体的传感器为{},\
                    \n距离指标为:\n{},\
                    \n\n相关系数指标为:\n{}".format(a,index_euclidean,index_correlation))