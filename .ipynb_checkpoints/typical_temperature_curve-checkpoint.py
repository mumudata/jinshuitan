#获取当前目录的上一层目录
import pandas as pd
import matplotlib.pylab as plt
import matplotlib
import seaborn as sns
import warnings
import numpy as np
from scipy.stats import norm
import os
os.chdir('../')
from data_read import data_read
from utils import parameter_calculation,location_data_acquisition,typical_temperature_curve
#进行数据的读取
def typical_temperature_curve_result(i,p,columns,file_path):
    """
    计算获取典型的温度曲线
    :param file_path:文件夹的路径
    :return:计算获取那一个是典型的温度曲线
    """
    pd_data_temperature = data_read(file_path)
    typical_temperature_columns = typical_temperature_curve(i,p,columns,pd_data_temperature)
    return typical_temperature_columns