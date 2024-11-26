import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from bCurve import Bezier, Point, Spline
import numpy as np
from PIL import Image


def hypot(x, y):
    return (x**2 + y**2)**0.5


def distance(p1, p2):
    return (((p1.x - p2.x) ** 2) + ((p1.y - p2.y) ** 2)) ** 0.5


def dot(x1, y1, x2, y2):
    return (x1 * x2) + (y1 * y2)


track_spline = Spline([
    Bezier(Point(251, 626), Point(263, 646), Point(286.7868127133783, 658.0197535168365), Point(308, 660)),
    Bezier(Point(308, 660), Point(329.2131872866217, 661.9802464831635), Point(358.48498177436346, 655.1220565995621), Point(386, 650)),
    Bezier(Point(386, 650), Point(413.51501822563654, 644.8779434004379), Point(435.34014682305747, 639.276416871691), Point(461, 627)),
    Bezier(Point(461, 627), Point(486.65985317694253, 614.723583128309), Point(517.8956587742977, 598.0903132675311), Point(537, 576)),
    Bezier(Point(537, 576), Point(556.1043412257023, 553.9096867324689), Point(562.0, 543.0), Point(571, 508)),
    Bezier(Point(571, 508), Point(580.0, 473.0), Point(559.0, 410.0), Point(563, 380)),
    Bezier(Point(563, 380), Point(567.0, 350.0), Point(600.0, 305.0), Point(610, 280)),
    Bezier(Point(610, 280), Point(620.0, 255.0), Point(623.0, 224.0), Point(635, 193)),
    Bezier(Point(635, 193), Point(647.0, 162.0), Point(662.0033989403206, 144.3387909605964), Point(699, 115)),
    Bezier(Point(699, 115), Point(735.9966010596794, 85.66120903940359), Point(771, 64), Point(793, 47)),
])
track_spline.make_continuous()

in_to_p = 404/16583
m_to_in = 39.3701
kmh_to_ms = 1/3.6
kmh_to_p = kmh_to_ms * m_to_in * in_to_p

track = np.asarray(Image.open('track.png'))
# plt.imshow(track)

const_t, const_time, dists = track_spline.traverse_const_vel(0, 1, 220*kmh_to_p, timestep=0.5)
var_t, var_time = track_spline.traverse_fv_constraint(0, 1, 1.7, 220*kmh_to_p, timestep=0.5)
vel_x, vel_y, acc_x, acc_y = track_spline.get_movement_vectors(const_t, const_time)
var_vel_x, var_vel_y, var_acc_x, var_acc_y = track_spline.get_movement_vectors(var_t, var_time)


# distances = []
# for i in range(len(const_t)-1):
#     p1 = track_spline.get_point_norm_m(const_t[i])
#     p2 = track_spline.get_point_norm_m(const_t[i + 1])
#     distances.append(distance(p1, p2))
#     vel_mag = hypot(vel_x[i], vel_y[i])

#     plt.scatter(p1.x, p1.y, color='r')
#     plt.arrow(p1.x, p1.y, vel_x[i], vel_y[i], head_width=3, color='b')
#     plt.arrow(p1.x, p1.y, acc_x[i], acc_y[i], head_width=3, color='g')
#     plt.xlim([200, 900])
#     plt.ylim([0, 700])
#     plt.grid = True
# plt.show()

# plt.plot(dists)
# plt.show()
# plt.plot(const_time, const_t)
# plt.show()

plt.plot(var_time, [hypot(x, y) for x, y in zip(var_vel_x, var_vel_y)])
plt.show()

fig, axs = plt.subplots(2, 2)
axs[0, 1].plot(acc_x, acc_y)
axs[0, 1].plot(1.7 * 9.81 * np.cos(np.linspace(0, 2*np.pi, 100)), 1.7 * 9.81 * np.sin(np.linspace(0, 2*np.pi, 100)))
axs[0, 1].scatter(0, 0)
axs[0, 1].axis('equal')

axs[0, 0].plot(vel_x, vel_y)
axs[0, 0].scatter(0, 0)
axs[0, 0].axis('equal')


axs[1, 1].plot(var_acc_x, var_acc_y)
axs[1, 1].plot(1.7 * 9.81 * np.cos(np.linspace(0, 2*np.pi, 100)), 1.7 * 9.81 * np.sin(np.linspace(0, 2*np.pi, 100)))
axs[1, 1].scatter(0, 0)
axs[1, 1].axis('equal')

axs[1, 0].plot(var_vel_x, var_vel_y)
axs[1, 0].scatter(0, 0)
axs[1, 0].axis('equal')

plt.show()