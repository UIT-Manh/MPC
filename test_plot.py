import numpy as np
import matplotlib.pyplot as plt

def tangent_line_to_circle(a, b, r, x0, y0):
    """
    Tính phương trình đường tiếp tuyến từ điểm (x0, y0) đến đường tròn có tâm (a, b) và bán kính r.
    """
    m = -(x0 - a) / (y0 - b)  # Đạo hàm của đường tròn tại điểm tiếp xúc (x0, y0)
    c = y0 - m * x0  # Hệ số góc và hệ số góc của đường tiếp tuyến
    breakpoint()
    return lambda x: m * x + c

# Tọa độ của tâm và bán kính của đường tròn
a, b, r = 0, 0, 1

# Chọn một điểm trên đường tròn
x0, y0 = 0.5, 1

# Tính phương trình đường tiếp tuyến
line_eq = tangent_line_to_circle(a, b, r, x0, y0)

# Tạo dữ liệu để vẽ đường tròn
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = a + r * np.cos(theta)
circle_y = b + r * np.sin(theta)

# Tạo dữ liệu để vẽ đường tiếp tuyến
breakpoint()
tangent_x = np.linspace(-1, 1, 100)
breakpoint()
tangent_y = line_eq(tangent_x)
breakpoint()
# Vẽ đường tròn và đường tiếp tuyến
plt.plot(circle_x, circle_y, label='Đường tròn')
plt.plot(tangent_x, tangent_y, label='Đường tiếp tuyến')

# Đánh dấu điểm tiếp xúc
plt.scatter([x0], [y0], color='red', label='Điểm tiếp xúc')

plt.title('Đường tròn và Đường tiếp tuyến')
plt.xlabel('X')
plt.ylabel('Y')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.axis('equal')
plt.show()
