import numpy as np
from itertools import product
import matplotlib.pyplot as plt
v_values=[2,5,7.5,10.5,12.5,15,17.5,20,22.5,25]
steering_values=[-50-40,-35,-32.5,-30,-27.5,-25,-22.5,-20,-17.5,-15,-12.5,-11.5,-10.5,-9.5,-8.5,-7.5,-6.5,-5.5,-4.5,-3.5,-2.5,-1.5,0,
          50,40,35,32.5,30,27.5,25,22.5,20,17.5,15,12.5,11.5,10.5,9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5
          
          ]
input_values=list(product(v_values,steering_values))
# print(input_values,input_values[0][1])
class Car:
    ID=0
    def __init__(self):
        Car.ID+=1
        
        # state
        self.x=0
        self.y=0
        self.yaw=0
        
    def state_current(self,x,y,yaw):
        return x,y,yaw
    def update_state(self,x,y,yaw):
        """_summary_

        Args:
            dT= 0.02
            L= 0.3
            list_pos_yaw= list[(x_f,y_f),yaw_f]
        """
        self.x=x
        self.y=y
        self.yaw=yaw
        # x_f=  x + 0.04 * v * np.cos(yaw/180*np.pi)
        # y_f=  y + 0.04 * v * np.sin(yaw/180*np.pi)
        # yaw_f=  yaw + v * 0.02 * np.tan(steering/180*np.pi) / 0.3
        
        # x_f= lambda v: x + 0.04 * v * np.cos(yaw/180*np.pi)
        # y_f= lambda v: y + 0.04 * v * np.sin(yaw/180*np.pi)
        # yaw_f= lambda v,steering: yaw + v * 0.02 * np.tan(steering/180*np.pi) / 0.3
        
        # result_x = map(x_f, v_values)
        # result_y = map(y_f, v_values)
        # result_yaw = map(lambda values:yaw_f(values[0],values[1]), input_values)
        
        # list_x_f = list(result_x)
        # list_y_f = list(result_y)
        # list_yaw = list(result_yaw)
        # pos = list(zip(list_x_f, list_y_f))
        # list_pos_yaw = [(pos[i % len(pos)], list_yaw[i]) for i in range(len(list_yaw))]
        # return x_f,y_f,yaw_f
    def plot_car_2d(self,x, y, theta):
        car_length = 0.6  # Length of the car (arbitrary value for demonstration)
        car_width = 0.4  # Width of the car (arbitrary value for demonstration)
        theta = np.deg2rad(theta)
        # Define the car's corners in the car's local coordinate system
        car_corners = np.array([[-car_length / 2, -car_width / 2],
                                [car_length / 2, -car_width / 2],
                                [car_length / 2, car_width / 2],
                                [-car_length / 2, car_width / 2]])

        # Rotate the car's corners based on the car's orientation (theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        rotated_car_corners = np.dot(car_corners, rotation_matrix.T)

        # Translate the car's corners based on its position (x, y)
        translated_car_corners = rotated_car_corners + np.array([x, y])

        # Plot the car
        print('translated_car_corners[:, 0]',translated_car_corners[:, 0],translated_car_corners[:, 1],theta)
        
        plt.plot(translated_car_corners[:, 0], translated_car_corners[:, 1], 'g-')
        plt.plot(x, y, 'ro')  # Mark the car's center with a red dot
        
        print('--------------------------------',x,y)
        plt.xlim(x-6,x+6)
        plt.ylim(y-6,y+6)
        # Additional plot settings
        plt.grid(True)
        # plt.axis('equal')
        