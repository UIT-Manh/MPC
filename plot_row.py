# Định nghĩa hàm số
def f(x):
    return x**2 + 5*x + 6

# Các giá trị của biến độc lập v_values cho trước
v_values = [2,5,10,15,20,25,30,40,50]

# Áp dụng hàm số f lên từng phần tử trong danh sách v_values
result = map(f, v_values)
print(result)
# Chuyển đổi đối tượng map thành danh sách để lấy các giá trị của f
result_list = list(result)

# In kết quả
for i, x in enumerate(v_values):
    print("f({}) = {}".format(x, result_list[i]))