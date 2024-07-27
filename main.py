import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from cost_funtion import *
from avoid_obs import *
from plot_elip import *
from car import *
import time
x=0
y=5
yaw=20
pre_closest_index=0
x_re,y_re,yaw_re=plot_line()
# idx_avoid=np.random.choice(np.arange(3, int(len(x_re)/1.5)), size=int(len(x_re)/20), replace=False)
# idx_avoid=np.random.choice(np.arange(4, int(len(x_re)/6)), size=1, replace=False)
idx_avoid=[40]
# print(idx_avoid)
avoid=list(zip(x_re[idx_avoid],y_re[idx_avoid]))

car=Car()

x_re_new,y_re_new,yaw_re_new=x_re.copy(),y_re.copy(),yaw_re.copy() # copy
# x_re_new,y_re_new,yaw_re_new=np.delete(x_re_new, 0),np.delete(y_re_new, 0),np.delete(yaw_re_new, 0)

# x_re,y_re,yaw_re=test_avoid(x_re,y_re,yaw_re)
count_step=0
n=0
pos_car=[]
pre_avoid=[0,0]
while (1):
    a=time.time()
    # x_re_new,y_re_new,yaw_re_new=distance_point(x,y,yaw,x_re_new,y_re_new,yaw_re_new)
    t=0
    x_re,y_re,yaw_re,pre_avoid,t=near_point_avoid(x,y,yaw,x_re[idx_avoid],y_re[idx_avoid],x_re,y_re,yaw_re,pre_closest_index,pre_avoid,t)
    # if(t==1):
    #     breakpoint()
    x,y,yaw,pre_closest_index=cost_funtion(x,y,yaw,x_re,y_re,yaw_re,pre_closest_index)
    # x,y,yaw,n=gradient(x,y,yaw,x_re_new,y_re_new,yaw_re_new,n)
    # Car.state_current=state_current
    
    pos_car.append([x,y,yaw])
    count_step+=1
    # plot(x_re,y_re,pos_car,avoid)
    # if(t==1 and count_step>10 ):
    #     plot(x_re,y_re,pos_car,avoid)
    #     plot(x_re,y_re,pos_car,avoid,pre_closest_index-2)
    # print("\n\n",count_step,len(x_re_new))
    print('time:',time.time()-a)
    if (count_step>100):
        break
print('pre_closest_index: ',pre_closest_index,x_re[pre_closest_index])
plot(x_re_new,y_re_new,pos_car,avoid)
# print("pos_car",pos_car)
