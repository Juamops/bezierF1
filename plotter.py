import matplotlib.pyplot as plt
from bCurve import Bezier, Point, Spline
import numpy as np
from PIL import Image


lower_track = [
    Point(571,508),
    Point(564,528),
    Point(554,552),
    Point(308,659),
    Point(537,576),
    Point(516,596),
    Point(490,611),
    Point(461,627),
    Point(439,636),
    Point(416,646),
    Point(386,650),
    Point(361,656),
    Point(329,659),
    Point(287,657),
    Point(267,641),
    Point(251,626)
]

upper_track = [
    Point(635,193),
    Point(647,162),
    Point(674,138),
    Point(699,115),
    Point(749,81),
    Point(771,64),
    Point(793,47),
    Point(817,31),
]

start = Point(0, 0)
cp1 = Point(1, 2)
cp2 = Point(4, 2)
end = Point(5, 0)
curve1 = Bezier(start, cp1, cp2, end)
start = Point(0, 0)
cp1 = Point(1, 2)
cp2 = Point(4, 2)
end = Point(5, 0)
curve2

n_points = 100
x_coords = []
y_coords = []

for t in np.linspace(0, 1, n_points):
    point_n = curve.get_point(t)
    x_coords.append(point_n.x)
    y_coords.append(point_n.y)

# plt.plot(x_coords, y_coords)
# plt.scatter(start.x, start.y, color='r')
# plt.scatter(cp1.x, cp1.y, color='r')
# plt.scatter(cp2.x, cp2.y, color='r')
# plt.scatter(end.x, end.y, color='r')
# plt.show()

track = np.asarray(Image.open('track.png'))
plt.imshow(track)

lower_curves = []
upper_curves = []

for i in range(0, len(lower_track)-4, 4):
    curve_i = Bezier(lower_track[i], lower_track[i+1], lower_track[i+2], lower_track[i+3])
    lower_curves.append(curve_i)

for i in range(0, len(upper_track)-4, 4):
    curve_i = Bezier(upper_track[i], upper_track[i+1], upper_track[i+2], upper_track[i+3])
    upper_curves.append(curve_i)

lower_spline = Spline(lower_curves)
upper_spline = Spline(upper_curves)

lower_pred_x = []
lower_pred_y = []

for t in np.linspace(0, 1, 500):
    point_t = lower_spline.get_point_norm(t)
    lower_pred_x.append(point_t.x)
    lower_pred_y.append(point_t.y)

plt.plot(lower_pred_x, lower_pred_y)
plt.show()