import numpy as np

from car import *


v_values=[-10,-20,-30,-40,2,5,10,15,17.5,20,22.5,25,30]
steering_values=[-50,-40,-35,-32.5,-30,-27.5,-25,-22.5,-20,-17.5,-15,-12.5,-11.5,-10.5,-9.5,-8.5,-7.5,-6.5,-5.5,-4.5,-3.5,-2.5,-1.5,0,
          50,40,35,32.5,30,27.5,25,22.5,20,17.5,15,12.5,11.5,10.5,9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5
          
          ]
input_values=list(product(v_values,steering_values))

car=Car()

def distance_point(x,y,yaw,x_re_new,y_re_new,yaw_re_new):
    x_dis= (x-x_re_new[0])**2
    y_dis= (y-y_re_new[0])**2
    dis= np.sqrt(x_dis+y_dis)
    while dis<0.3 and abs(yaw-yaw_re_new[0])<4:
        # print('dis, abs(yaw-yaw_re_new) : ',dis,abs(yaw-yaw_re_new[0]))
        
        x_re_new,y_re_new,yaw_re_new=np.delete(x_re_new, 0),np.delete(y_re_new, 0),np.delete(yaw_re_new, 0)
        
        
        x_dis= (x-x_re_new[0])**2
        y_dis= (y-y_re_new[0])**2
        dis= np.sqrt(x_dis+y_dis)
    return x_re_new,y_re_new,yaw_re_new
def state_prediction(x,y,yaw):
    """
    x: Coordinates x
    y: Coordinates y
    yaw: Yaw angle
    n: Number of prediction horizon
    """
    """_summary_

    Args:
        dT= 0.05
        L= 0.3
        list_pos_yaw= list[(x_f,y_f),yaw_f]
    """ 
    
    
    x_f= lambda v: x + 0.05 * v * np.cos(yaw/180*np.pi)
    y_f= lambda v: y + 0.05 * v * np.sin(yaw/180*np.pi)
    yaw_f= lambda v,steering: yaw + v * 0.05 * np.tan(steering/180*np.pi) / 0.3
    
    result_x = map(x_f, v_values)
    result_y = map(y_f, v_values)
    result_yaw = map(lambda values:yaw_f(values[0],values[1]), input_values)
    
    list_x_f = list(result_x)
    list_y_f = list(result_y)
    list_yaw = list(result_yaw)
    
    list_yaw = [x + 360 if x <= -180 else x for x in list_yaw]
    
    pos = list(zip(list_x_f, list_y_f))
    list_pos_yaw = [(pos[i % len(pos)], list_yaw[i]) for i in range(len(list_yaw))]
    return list_pos_yaw

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
def near_point(x,y,x_re_new,y_re_new,pre_closest_index):
    pos = [[x1, y1] for x1, y1 in zip(x_re_new, y_re_new)]
    filtered_points = list(filter(lambda point: euclidean_distance([x, y], point) >= 1 and pos.index(point)-pre_closest_index<5 and pos.index(point)-pre_closest_index > -4, pos))
    # print('filtered_points',filtered_points)
    closest_point = min(filtered_points, key=lambda point: euclidean_distance([x,y], point))
    closest_index = pos.index(closest_point)
    
    vector1=[x_re_new[closest_index]-x , y_re_new[closest_index]-y]
    vector2=[x_re_new[closest_index+1]-x , y_re_new[closest_index+1]-y]
    def dot_product(vector1, vector2):
        return sum(x * y for x, y in zip(vector1, vector2))

    def magnitude(vector):
        return np.sqrt(sum(x**2 for x in vector))

    cosine_angle = dot_product(vector1, vector2) / (magnitude(vector1) * magnitude(vector2))
    angle = np.arccos(cosine_angle)

    angle_degrees = np.degrees(angle)
    print('closest_index_1',x,y,x_re_new[closest_index],y_re_new[closest_index],angle_degrees)
    if angle_degrees >80:
        closest_index +=2
    elif angle_degrees >27 and angle_degrees <=80:
        closest_index +=3
    print('closest_index_2',x_re_new[closest_index],y_re_new[closest_index])
    # print('near_point',angle_degrees,closest_index)
    return closest_index

def cost_funtion(x,y,yaw,x_re_new,y_re_new,yaw_re_new,pre_closest_index):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
        yaw (_type_): _description_
        x_re_new (_type_): _description_
        y_re_new (_type_): _description_
        yaw_re_new (_type_): _description_

    Returns:
        list_pos_yaw= list[((x_f,y_f),yaw_f),...]
    """
    # input_costs = []
    # list_pos_yaw=state_prediction(x,y,yaw)
    # cost = np.zeros(len(list_pos_yaw))
    
    # for i in range (len(list_pos_yaw)):
    #     cost[i]= 20*((list_pos_yaw[i][0][0]-x_re_new[0])**2 + (list_pos_yaw[i][0][1]-y_re_new[0])**2) + 1*(list_pos_yaw[i][1]-yaw_re_new[0])**2
        
    #     input_costs.append([list_pos_yaw[i], cost[i]])
    #     # if(i==2):
    #     #     print('cost',input_costs)
    # input_costs.sort(key=lambda x: x[1])
    # best_input = input_costs[0][0]

                
    list_cost=[[0]]
    list_state=[[((x,y),yaw)]]
    print('pre state',x,y,yaw)
    pre_closest_index=near_point(x,y,x_re_new,y_re_new,pre_closest_index)
    for i_prediction in range (3): # number of predictions
        
        
        list_cost.append([])
        list_state.append([])
        weight_state= 1/(i_prediction+0.5)
        closest_index = pre_closest_index + i_prediction
        for case in range (3**i_prediction):    # number of cases
            (x,y),yaw=list_state[i_prediction][case]
            
            list_pos_yaw=state_prediction(x,y,yaw) # all of the cases
            input_costs = []
            cost = np.zeros(len(list_pos_yaw))   
            
            pos=list_state[i_prediction]
            x,y,yaw= pos[case][0][0],pos[case][0][0],pos[case][1] # update state predictions
            # x_re_new,y_re_new,yaw_re_new=distance_point(x,y,yaw,x_re_new,y_re_new,yaw_re_new)
            
            for i in range (len(list_pos_yaw)): # calculate cost function
                cost[i]= weight_state*0.8*((list_pos_yaw[i][0][0]-x_re_new[closest_index])**2 + (list_pos_yaw[i][0][1]-y_re_new[closest_index])**2)+ weight_state*1*(list_pos_yaw[i][1]-yaw_re_new[closest_index])**2                
                input_costs.append([list_pos_yaw[i], cost[i]])                
            input_costs.sort(key=lambda x: x[1])
            # get the variables
            case_cost = list(map(lambda x: x[:3][-1], input_costs[:3]))
            case_state = list(map(lambda x: x[:3][0], input_costs[:3]))
            # print("case_cost",case_cost)
            # print("case_state",case_state)
            
            case_cost=np.array(case_cost)
            case_cost+=list_cost[i_prediction][case]
            
            # append list_cost and list_state
            list_cost[i_prediction+1].extend(case_cost)
            list_state[i_prediction+1].extend(case_state)
    # print("list_cost",list_cost)
    # print("list_state",list_state)
    best_cost=min(list_cost[3])
    cost_index=list_cost[3].index(best_cost)
    index=cost_index//9
    
    x,y,yaw=list_state[1][index][0][0],list_state[1][index][0][1],list_state[1][index][1]
    print('index',index)
    car.update_state(x,y,yaw)
    # car.update_state(best_input[0][0],best_input[0][1],best_input[1])
    # return best_input[0][0],best_input[0][1],best_input[1]
    # print("best_cost",best_cost,cost_index,x,y,yaw)
    return x,y,yaw,closest_index

    # for i in range(0,n):
def gradient(x,y,yaw,x_re_new,y_re_new,yaw_re_new,n):
    n+=1
    a=x_re_new[0]-x
    b=y_re_new[0]-y
    c=yaw_re_new[0]-yaw
    
    v = 10
    steering = -c*7  
    learning_rate = 0.1  
    num_iterations = 50  # Số lần lặp tối đa
    
    beta=0.9
    v1,v2=0,0
    # Lặp lại gradient descent
    for i in range(num_iterations):
        # Tính đạo hàm riêng (gradient)
        gradient_v = 2*v*(20+np.tan(steering/180*np.pi)**2)- 2*(20*a*np.cos(yaw/180*np.pi)+ 
                    20*b*np.sin(yaw/180*np.pi) + c*np.tan(steering/180*np.pi))
        gradient_steering = (2*np.tan(steering/180*np.pi)*v**2-2*v*c)/(np.cos(steering/180*np.pi)**2)
        
        # Momentum
        v1 = beta*v1+ (1-beta)*gradient_v
        v2 = beta*v2+ (1-beta)*gradient_steering
        # Cập nhật giá trị của x1 và x2
        v = v - learning_rate * v1
        steering = steering - learning_rate * v2

    # Trả về giá trị của x1 và x2
    return x + 0.05 * v * np.cos(yaw/180*np.pi), y + 0.05 * v * np.sin(yaw/180*np.pi), yaw + v * 0.05 * np.tan(steering/180*np.pi) / 0.3,n