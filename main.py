import os
# os.chdir('../')
from typical_temperature_curve import typical_temperature_curve_result

def main():
    """
    计算获取典型的温度曲线
    :param file_path:文件夹的路径
    :return:计算获取那一个是典型的温度曲线
    """
    result = typical_temperature_curve_result(2,400,"上导轴瓦","/usr/linyuanyuan/jinshuitan_power_plant/temperature_data/jinshuitan_Temperature.csv")
    return result 

if __name__ == '__main__':
    print(main())


    
    
    
    