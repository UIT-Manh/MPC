import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from car import *
yaw_f=[]
def plot_line():
    # Dữ liệu của đường cong
    x_values = np.linspace(0, 80, 45)
    x_values_para = np.linspace(40, 80, 45)
    # y_values = (2 * np.sin(x_values) + np.random.randn(20) * 0.1)*10+6

    # Hàm số để tìm hàm số phù hợp với dữ liệu
    def func(x, a, b, c):
        return a * np.sin(b * x) + c
    def func_para(x):
        return 19*(x**2)/800-57*x/20+68
    # Sử dụng hàm curve_fit để tìm hàm số phù hợp với dữ liệu
    # popt, pcov = curve_fit(func, x_values, y_values)
    # Vẽ đường cong
    # plt.plot(x_values, y_values, 'o', label='Dữ liệu thực tế')
    # fig, ax = plt.subplots()
    y_values=func(x_values, 6,0.5/3.14/4,5.13)
    y_values_para=func_para(x_values_para)
    
    x_values_1=np.append(x_values,[81.25,82.5,82.25,82.25,82])
    y_values_1=np.append(y_values,[3.75,2,-0.75,-2.25,-3.5])
    
    
    x_values_1=np.append(x_values_1,x_values_para[::-1])
    y_values_1=np.append(y_values_1,y_values_para[::-1])
    
    x_values_1=np.append(x_values_1,[39,38.5,39.5,42,44.5])
    y_values_1=np.append(y_values_1,[-5.75,-3.25,0,2,4])
        
    x_values_2=np.append(x_values_1,(x_values+50))
    y_values_2=np.append(y_values_1,y_values)
    # plt.plot(x_values, y_values, 'r-', label='Reference')
    # # Đặt tên cho trục x và y
    # plt.xlabel('Trục x')
    # plt.ylabel('Trục y')
    for i in range(1,len(x_values_2)):
        x1, y1 = x_values_2[i], y_values_2[i]
        x2, y2 = x_values_2[i-1], y_values_2[i-1]
        dx, dy = x1 - x2, y1 - y2
        yaw_f.append(np.arctan2(dy, dx) * 180 / np.pi)
        # plt.arrow(x2, y2, dx, dy, head_width=0.03, head_length=0.03, fc='blue', ec='blue',linewidth=2.5)
    # car
    # plt.arrow(x_values[0], y_values[0], 0.0588348, 0.2941, head_width=0.05, head_length=0.05, fc='g', ec='g',linewidth=2.5)  
    # rect = plt.Rectangle((x_values[0],y_values[0]), 0.1, 0.4, linewidth=2, edgecolor='g', facecolor='none')
    
    # ax.add_patch(rect)
    # # Hiển thị biểu đồ
    # plt.legend()
    # plt.show()
    return x_values_2,y_values_2,yaw_f
def plot_avoid(avoid,flag_random,x_avs,y_avs):
    if flag_random==0:
        
        for point in avoid:
            # Số lượng điểm trong hình dạng khép kín
            num_points = 4

            # Tạo hình dạng khép kín ngẫu nhiên từ điểm đầu tiên
            x_av = []
            y_av = []

            for _ in range(num_points ):
                x_av.append(point[0] + np.random.uniform(-0.5, 0.5))
                y_av.append(point[1]+ np.random.uniform(-0.5, 0.5))

            # Đóng hình dạng khép kín
            x_av.append(x_av[0])
            y_av.append(y_av[0])

            x_avs.append(x_av)
            y_avs.append(y_av)
            # Vẽ hình dạng khép kín
            plt.plot(x_av, y_av, marker='o')
        flag_random=1
        return flag_random,x_avs,y_avs
    else:
        for i in range (0,len(x_avs)):
            plt.plot(x_avs[i], y_avs[i], marker='o')
        return flag_random,x_avs,y_avs
def plot(x_values,y_values,pos_car,avoid):
    # # plt.clf()
    # plt.plot(x_values, y_values, 'r-', label='Reference')
    # plt.xlabel('Trục x')
    # plt.ylabel('Trục y')
    # for i in range(1,len(x_values)):
    #     x1, y1 = x_values[i], y_values[i]
    #     x2, y2 = x_values[i-1], y_values[i-1]
    #     dx, dy = x1 - x2, y1 - y2
    #     plt.arrow(x2, y2, dx, dy, head_width=0.03, head_length=0.03, fc='blue', ec='blue',linewidth=2.5)
    # plt.arrow(0, 5.09412287, 0.0588348, 0.2941, head_width=0.05, head_length=0.05, fc='g', ec='g',linewidth=2.5) 
    flag_random=0
    x_avs=[]
    y_avs=[]
    for j in range(len(pos_car)):
        plt.clf()
        # plt.plot(x_values, y_values, 'r-', label='Reference')
        plt.xlabel('Trục x')
        plt.ylabel('Trục y')

        for i in range(1,len(x_values)):
            x1, y1 = x_values[i], y_values[i]
            x2, y2 = x_values[i-1], y_values[i-1]
            dx, dy = x1 - x2, y1 - y2
            plt.arrow(x2, y2, dx, dy, head_width=0.03, head_length=0.03, fc='blue', ec='blue',linewidth=2.5)
        
        # x_front_car= np.cos(pos_car[j][2]/180*np.pi)*0.3
        # y_front_car= np.sin(pos_car[j][2]/180*np.pi)*0.3
        # plt.arrow(pos_car[j][0], pos_car[j][1], x_front_car, y_front_car, head_width=0.1, head_length=0.1, fc='g', ec='g',linewidth=5)
        car=Car()
        car.plot_car_2d(pos_car[j][0], pos_car[j][1], pos_car[j][2])

        flag_random,x_avs,y_avs=plot_avoid(avoid,flag_random,x_avs,y_avs)

        plt.pause(0.1)
    plt.legend()
    plt.show()
def test_avoid(x_re,y_re,yaw_re):
    a=y_re[13]
    for i in range(14,26):
        if i<=16:    
            a+=0.06
        elif i>16 and i<=19:
            a+=0.02
        elif i>19 and i<25:
            a-=0.04
        y_re[i]=a
        x1, y1 = x_re[i], y_re[i]
        x2, y2 = x_re[i-1], y_re[i-1]
        dx, dy = x1 - x2, y1 - y2
        yaw_re[i]=np.arctan2(dy, dx) * 180 / np.pi
        plt.arrow(x2, y2, dx, dy, head_width=0.03, head_length=0.03, fc='red', ec='red',linewidth=2.5)
        
    return x_re,y_re,yaw_re
        
    