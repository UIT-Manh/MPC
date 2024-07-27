import numpy as np
import matplotlib.pyplot as plt
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
def tangent_line(x_v,y_v,x_avoid,y_avoid,rd=3):
    """_summary_

    Args:
        x_v (_type_): _description_
        y_v (_type_): _description_
        x_avoid (_type_): _description_
        y_avoid (_type_): _description_
        rd (_type_): radios
    VTPT: (a,b)
    PTTT: a*x+b*y-a*x_v-b*y_v=0
    PT khoang cach: (a^2)*(rd^2-T)-2*a*R+4-U=0
        a:hidden
        default:b=1
    Returns:
        _type_: _description_
    """
    x_dis=x_avoid-x_v
    y_dis=y_avoid-y_v
    T=x_dis**2
    R=x_dis*y_dis
    U=y_dis**2
    delta= R**2-(rd**2-T)*(rd**2-U)
    a1= (R-np.sqrt(delta))/(rd**2-T)
    a2= (R+np.sqrt(delta))/(rd**2-T)
    return a1,a2
def giai_he_phuong_trinh(a1, b1, c1, a2, b2, c2):
    # Tạo ma trận hệ số
    he_so = np.array([[a1, b1], [a2, b2]])

    # Tạo vector kết quả
    ket_qua = np.array([c1, c2])

    # Giải hệ phương trình
    giai_nghiem = np.linalg.solve(he_so, ket_qua)

    return giai_nghiem
def process_after(a1_after,a2_after,x_h,y_h, x_v , y_v,x_avoid,y_avoid):
    print('a1_after',a1_after,a2_after)
    if np.any(np.isnan(a2_after)):
        return [],[]
    a1, b1, c1 = a1_after,  1, a1_after * x_v + y_v # PT đường thẳng
    a2, b2, c2 = 1, -a1_after, x_avoid - a1_after*y_avoid # PT liên quan đến vecto cung phuong (vuông góc)

    a3, b3, c3 = a2_after,  1, a2_after * x_v + y_v # PT đường thẳng
    a4, b4, c4 = 1, -a2_after, x_avoid - a2_after*y_avoid # PT liên quan đến vecto cung phuong (vuông góc)
    
    x_h1,y_h1 = giai_he_phuong_trinh(a1, b1, c1, a2, b2, c2)
    x_h2,y_h2 = giai_he_phuong_trinh(a3, b3, c3, a4, b4, c4)
    
    distance_ab = euclidean_distance((x_h1,y_h1),(x_h,y_h))
    distance_ac = euclidean_distance((x_h2,y_h2),(x_h,y_h))
    
    if distance_ab < distance_ac:
        x_h,y_h=x_h1,y_h1
        distance=distance_ab
        a,c=a1,c1
    else:
        x_h,y_h=x_h2,y_h2
        distance=distance_ac
        a,c=a3,c3
    print('a',a1_after,a2_after)
    x_values_avoid = np.linspace(x_h, x_v, int(euclidean_distance((x_v,y_v),(x_h,y_h))/0.8))
    y_values_avoid = (lambda x: -a * x + c)(x_values_avoid)
    return x_values_avoid,y_values_avoid
def fstp(x_v,y_v,x_re,y_re,filtered_points_after1,closest_point,rd):
    """PTTT: a*x+b*y-a*x_v-b*y_v=0
        default: b=1
 
    """
    yaw_f=[]
    x_avoid=closest_point[0]
    y_avoid=closest_point[1]
    a1,a2=tangent_line(x_v,y_v,x_avoid,y_avoid,rd)
    # a= np.random.choice([a1,a2])
    a=a1
    if not a:
        return None
    # Ví dụ: giải hệ phương trình a3*x_h + y_h - a3*x_v - y_v = 0 và 3 x_h - 2y_h = 1
    a3, b3, c3 = a,  1, a * x_v + y_v # PT đường thẳng
    a4, b4, c4 = 1, -a, x_avoid - a*y_avoid # PT liên quan đến vecto cung phuong (vuông góc)

    x_h,y_h = giai_he_phuong_trinh(a3, b3, c3, a4, b4, c4)
    x_values_avoid = np.linspace(x_v, x_h, int(euclidean_distance((x_v,y_v),(x_h,y_h))/1))
    y_values_avoid = (lambda x: -a3 * x + c3)(x_values_avoid)
    
    # fstp after
    
    a1_after,a2_after=tangent_line(filtered_points_after1[0],filtered_points_after1[1],x_avoid,y_avoid,rd)
    x_values_avoid_after,y_values_avoid_after=process_after(a1_after,a2_after,x_h,y_h,filtered_points_after1[0],filtered_points_after1[1],x_avoid,y_avoid)
    
    x_values_avoid_after_1 = np.linspace(x_values_avoid[-1], x_values_avoid_after[0], int(euclidean_distance((x_values_avoid[-1],y_values_avoid[-1]),(x_values_avoid_after[0],y_values_avoid_after[0]))/0.4))
    y_values_avoid_after_1 = np.linspace(y_values_avoid[-1], y_values_avoid_after[0], int(euclidean_distance((x_values_avoid[-1],y_values_avoid[-1]),(x_values_avoid_after[0],y_values_avoid_after[0]))/0.4))
    
    # x_values_avoid=np.append(x_v,x_values_avoid)
    # y_values_avoid=np.append(x_v,y_values_avoid)
    
    x_values_avoid=np.append(x_values_avoid,x_values_avoid_after_1)
    y_values_avoid=np.append(y_values_avoid,y_values_avoid_after_1)
    
    x_values_avoid=np.append(x_values_avoid,x_values_avoid_after)
    y_values_avoid=np.append(y_values_avoid,y_values_avoid_after)
    # plt.xlabel('Trục x')
    # plt.ylabel('Trục y')
    # plt.clf
    # for i in range(1,len(x_values_avoid)):
    #     x1, y1 = x_values_avoid[i], y_values_avoid[i]
    #     x2, y2 = x_values_avoid[i-1], y_values_avoid[i-1]
    #     dx, dy = x1 - x2, y1 - y2
    #     plt.arrow(x2, y2, dx, dy, head_width=0.03, head_length=0.03, fc='g', ec='g',linewidth=2.5)
    # plt.scatter(x_avoid,y_avoid, color='red', label='Điểm tiếp xúc')
    # plt.scatter(x_v,y_v, color='blue', label='Điểm tiếp11 xúc')
    # plt.legend()
    # plt.show()
    for i in range(1,len(x_values_avoid)):
        x1, y1 = x_values_avoid[i], y_values_avoid[i]
        x2, y2 = x_values_avoid[i-1], y_values_avoid[i-1]
        dx, dy = x1 - x2, y1 - y2
        yaw_f.append(np.arctan2(dy, dx) * 180 / np.pi)
    return x_values_avoid,y_values_avoid,yaw_f
    # PTTT a*x+y-a*x_v-y_v=0
def near_point_avoid(x_v,y_v,yaw,x_avoid,y_avoid,x_re,y_re,yaw_re,pre_closest_index,pre_avoid,t):
    def hieu_goc(x_avoid,y_avoid,x_v,y_v):
        vector1=[x_avoid-x_v , y_avoid-y_v]
        vector2=[x_re[pre_closest_index]-x_v , y_re[pre_closest_index]-y_v]
        def dot_product(vector1, vector2):
            return sum(x * y for x, y in zip(vector1, vector2))

        def magnitude(vector):
            return np.sqrt(sum(x**2 for x in vector))

        cosine_angle = dot_product(vector1, vector2) / (magnitude(vector1) * magnitude(vector2))
        angle = np.arccos(cosine_angle)

        angle_degrees = np.degrees(angle)
        if abs(angle_degrees) <60:
            return 1
        return 0
    pos = [[x1, y1] for x1, y1 in zip(x_avoid, y_avoid)]
    filtered_points = list(filter(lambda point: euclidean_distance([x_v, y_v], point) < 8 and euclidean_distance([x_v, y_v], point) > 3 and hieu_goc(point[0],point[1],x_v,y_v)==1, pos))
    
    
    if not filtered_points:
        print('filtered_points 0 ',filtered_points)
        return x_re,y_re,yaw_re,pre_avoid,t
    
    closest_point = min(filtered_points, key=lambda point: euclidean_distance([x_v,y_v], point))
    
    if closest_point[0] == pre_avoid[0] and closest_point[1] == pre_avoid[1] :
        return x_re,y_re,yaw_re,pre_avoid,t
    #after_point
    
    pos_after = [[x1, y1] for x1, y1 in zip(x_re[pre_closest_index:],y_re[pre_closest_index:])]
    filtered_points_after = list(filter(lambda point: euclidean_distance(closest_point, point) > 3 and euclidean_distance(closest_point, point) < 8 and euclidean_distance([x_v, y_v], point) > 13 , pos_after))
    print('filtered_points_after',filtered_points_after)
    
    if len(filtered_points_after)==0:
        return x_re,y_re,yaw_re,pre_avoid,t
    filtered_points_after1 = filtered_points_after[0]
    filtered_points_after_index = pos_after.index(filtered_points_after1)
    # delete points avoid_obs
    x_re= np.delete(x_re, slice(pre_closest_index-2, pre_closest_index + filtered_points_after_index))
    y_re= np.delete(y_re, slice(pre_closest_index-2, pre_closest_index + filtered_points_after_index))
    yaw_re= np.delete(yaw_re, slice(pre_closest_index-2, pre_closest_index + filtered_points_after_index))
    
    x_values_avoid,y_values_avoid,yaw_f=fstp(x_v,y_v,x_re,y_re,filtered_points_after1,closest_point,rd=3)
    
    x_re = np.insert(x_re, pre_closest_index-2, x_values_avoid)
    y_re = np.insert(y_re, pre_closest_index-2, y_values_avoid)
    yaw_re = np.insert(yaw_re, pre_closest_index-2, yaw_f)
    print('filtered_points',closest_point,pre_closest_index)
    
    # breakpoint()
    pre_avoid=closest_point
    # closest_index = pos.index(closest_point)
    
    t=1
    return x_re,y_re,yaw_re,pre_avoid,t

# def avoid(x_obs,y_obs,x,y):
    