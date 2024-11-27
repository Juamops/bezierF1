import matplotlib.pyplot as plt
from bCurve import Bezier, Point, Spline
import numpy as np
from PIL import Image


lower_track = [
    Point(571,508),
    Point(564,528),
    Point(554,552),
    Point(537,576),
    Point(516,596),
    Point(490,611),
    Point(461,627),
    Point(439,636),
    Point(416,646),
    Point(386,650),
    Point(361,656),
    Point(329,659),
    Point(308,659),
    Point(287,657),
    Point(267,641),
    Point(251,626)
]

# upper_track = [
#     Point(635,193),
#     Point(647,162),
#     Point(674,138),
#     Point(699,115),
#     Point(749,81),
#     Point(771,64),
#     Point(793,47),
#     Point(817,31),
# ]

upper_track = [
    Point(817, 31),
    Point(793, 47),
    Point(771, 64),
    Point(749, 81),
    Point(699, 115),
    Point(674, 138),
    Point(647, 162),
    # Point(635, 193),
    Point(680, 220),
]

# start = Point(0, 0)
# cp1 = Point(1, 2)
# cp2 = Point(4, 2)
# end = Point(5, 0)
# curve1 = Bezier(start, cp1, cp2, end)
# start = Point(5, 0)
# cp1 = Point(6, 1)
# cp2 = Point(9, 1)
# end = Point(10, 0)
# curve2 = Bezier(start, cp1, cp2, end)
#
# test_spline = Spline([curve1, curve2])
#
# n_points = 400
# x_coords, y_coords = test_spline.get_points(0, 1, 500)
#
# control_x, control_y = test_spline.get_control_points()
#
# test_point = curve1.get_poly_point(0.2)
# x_test = test_point.x
# y_test = test_point.y
# x_dev1, y_dev1 = curve1.get_first_derivative(0.2)
# x_dev2, y_dev2 = curve1.get_second_derivative(0.2)
#
# x_dev1 /= 5
# y_dev1 /= 5
# x_dev2 /= 5
# y_dev2 /= 5
#
# x_dev1 += x_test
# y_dev1 += y_test
# x_dev2 += x_test
# y_dev2 += y_test
#
# plt.plot(x_coords, y_coords)
# plt.scatter(control_x, control_y, color='r')
# plt.plot([x_test, x_dev1], [y_test, y_dev1], color='g')
# plt.plot([x_test, x_dev2], [y_test, y_dev2], color='b')
# plt.show()
#
# in_to_p = 97/16583
# m_to_in = 39.3701
# kmh_to_ms = 1/3.6
# kmh_to_p = m_to_in * in_to_p * kmh_to_ms
#
#
# t_vals_const, total_time_const = test_spline.traverse_const_vel(0, 1, 200 * kmh_to_p)
# t_vals_var, total_time_var = test_spline.traverse_fv_constraint(0, 1, 0.8, 200 * kmh_to_p)
# print(max(total_time_const))
# print(max(total_time_var))
# print(f'Length: {test_spline.length()}')
# plt.plot(total_time_const, t_vals_const)
# plt.show()

track = np.asarray(Image.open('track.png'))
# plt.imshow(track)

lower_curves = []
upper_curves = []

for i in range(0, len(lower_track)-1, 3):
    curve_i = Bezier(lower_track[i], lower_track[i+1], lower_track[i+2], lower_track[i+3])
    lower_curves.append(curve_i)

for i in range(1, len(upper_track)-1, 3):
    curve_i = Bezier(upper_track[i], upper_track[i+1], upper_track[i+2], upper_track[i+3])
    upper_curves.append(curve_i)

lower_spline = Spline(lower_curves)
upper_spline = Spline(upper_curves)

print(len(lower_spline.curves))
print(len(upper_spline.curves))

lower_spline.make_continuous()
upper_spline.make_continuous()
x_control, y_control = upper_spline.get_control_lines()

# for x, y in zip(x_control, y_control):
#     plt.plot(x, y, color='g')
#
# lower_pred_x, lower_pred_y = lower_spline.get_points(0, 1, 500)
# upper_pred_x, upper_pred_y = upper_spline.get_points(0, 1, 500)
# plt.plot(lower_pred_x, lower_pred_y)
# plt.plot(upper_pred_x, upper_pred_y)
#
# plt.show()

in_to_p = 404/16583
m_to_in = 39.3701
kmh_to_ms = 1/3.6
kmh_to_p = m_to_in * in_to_p * kmh_to_ms


# t_vals_const, total_time_const = lower_spline.traverse_const_vel(0, 1, 200 * kmh_to_p)
# t_vals_var, total_time_var = lower_spline.traverse_fv_constraint(0, 1, 1.0, 200 * kmh_to_p)
# plt.plot(total_time_const, t_vals_const)
# plt.plot(total_time_var, t_vals_var)
# print(len(t_vals_const))
# print(len(t_vals_var))

t_vals_lower, total_time_lower = lower_spline.traverse_const_vel(0, 1, 50 * kmh_to_p)
t_vals_upper, total_time_upper = upper_spline.traverse_const_vel(0, 1, 50 * kmh_to_p)
print(f'(Lower spline) time: {max(total_time_lower): .2f}s, n_steps: {len(t_vals_lower)}, velocity: {50 * kmh_to_p}')
print(f'(Upper spline) time: {max(total_time_upper): .2f}s, n_steps: {len(t_vals_upper)}, velocity: {50 * kmh_to_p}')

lengths_upper = []
lengths_lower = []

for vel in range(1, 100):
    lengths_upper.append(upper_spline.length(vel))
    lengths_lower.append(lower_spline.length(vel))

plt.plot(lengths_upper)
plt.plot(lengths_lower)

plt.show()