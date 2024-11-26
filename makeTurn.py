import matplotlib.pyplot as plt
from bCurve import Bezier, Point, Spline
import numpy as np
from PIL import Image


lower_track = [
    Point(571, 508),
    Point(562, 543),
    Point(554, 552),
    Point(537, 576),
    Point(516, 596),
    Point(490, 611),
    Point(461, 627),
    Point(439, 636),
    Point(416, 646),
    Point(386, 650),
    Point(361, 656),
    Point(329, 659),
    Point(308, 660),
    Point(287, 655),
    Point(263, 646),
    Point(251, 626)
]

upper_track = [
    Point(817, 31),
    Point(793, 47),
    Point(771, 64),
    Point(749, 81),
    Point(699, 115),
    Point(674, 138),
    Point(647, 162),
    Point(635, 193),
]

new_curve = Bezier(
    Point(563, 380),
    Point(567, 350),
    Point(600, 305),
    Point(610, 280)
)

lower_track = list(reversed(lower_track))
upper_track = list(reversed(upper_track))

lower_curves = []
upper_curves = []

for i in range(0, len(lower_track)-1, 3):
    curve_i = Bezier(lower_track[i], lower_track[i+1], lower_track[i+2], lower_track[i+3])
    lower_curves.append(curve_i)

for i in range(0, len(upper_track)-2, 3):
    curve_i = Bezier(upper_track[i], upper_track[i+1], upper_track[i+2], upper_track[i+3])
    upper_curves.append(curve_i)

lower_spline = Spline(lower_curves)
upper_spline = Spline(upper_curves)

lower_spline.make_continuous()
upper_spline.make_continuous()

lower_spline.add_curve(new_curve)
lower_spline.join_with(upper_spline)
lower_spline.make_continuous()

track = np.asarray(Image.open('track.png'))
plt.imshow(track)

x_lower, y_lower = lower_spline.get_points(0, 1, 500)
x_control, y_control = lower_spline.get_control_points()
plt.plot(x_lower, y_lower, lw=3)
plt.scatter(x_control, y_control, color='r')
# x_d, y_d = lower_spline.get_first_derivatives(0, 0.5, 500)
# plt.plot(x_d, y_d)
# lower_spline.print_spline()
lower_spline.print_spline()
plt.show()