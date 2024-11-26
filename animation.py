import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from bCurve import Bezier, Point, Spline
import numpy as np
from PIL import Image


def hypot(x, y):
    return (x**2 + y**2)**0.5


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
kmh_to_p = m_to_in * in_to_p * kmh_to_ms

track = np.asarray(Image.open('track.png'))
x, y = track_spline.get_points(0, 1, 100)
x_d, y_d = track_spline.get_first_derivatives(0, 1, 100)
x_d2, y_d2 = track_spline.get_second_derivatives(0, 1, 100)

# const_t, const_time = track_spline.traverse_fv_constraint(0, 1, 0.47, 200*kmh_to_p, timestep=0.0001)
const_t, const_time, d = track_spline.traverse_const_vel(0, 1, 300*kmh_to_p, timestep=0.1)
vel_x, vel_y, acc_x, acc_y = track_spline.get_movement_vectors(const_t, const_time)

# danger_x = [track_spline.get_point_norm(t).x + vel_x[i]/2 if track_spline.get_max_vel(t, 1.7) < 50 else 0 for i, t in enumerate(const_t)]
# danger_y = [track_spline.get_point_norm(t).y + vel_y[i]/2 if track_spline.get_max_vel(t, 1.7) < 50 else 0 for i, t in enumerate(const_t)]

danger_x = [track_spline.get_point_norm(t).x + 80*vel_x[i]/hypot(vel_x[i], vel_y[i]) if hypot(acc_x[i], acc_y[i]) > 1.7 * 9.81 * 5 else 0 for i, t in enumerate(const_t)]
danger_y = [track_spline.get_point_norm(t).y + 80*vel_y[i]/hypot(vel_x[i], vel_y[i]) if hypot(acc_x[i], acc_y[i]) > 1.7 * 9.81 * 5 else 0 for i, t in enumerate(const_t)]

print(max([track_spline.get_max_vel(t, 1.7) for t in const_t]))

fig, ax = plt.subplots()
ax.imshow(track)
ax.plot(x, y, lw=3, zorder=0)

# animated_track, = ax.plot([], [], lw=3)
v_scale = 0.3
animated_point, = ax.plot(x[0], y[0], color='r', marker='o', markersize=6, zorder=15)
animated_velocity = ax.arrow(x[0], y[0], vel_x[0]*v_scale, vel_y[0]*v_scale, color='b', width=2, head_width=10, zorder=10)
animated_acceleration = ax.arrow(x[0], y[0], acc_x[0]*v_scale, acc_y[0]*v_scale, color='g', width=2, head_width=10, zorder=10)
animated_danger, = ax.plot([], [], linestyle='', marker='o', color='r')


def update(frame):
    # animated_plot.set_linewidth(3)
    # animated_track.set_data(x[:frame], y[:frame])
    point = track_spline.get_point_norm(const_t[frame])
    animated_velocity.set_data(x=point.x, y=point.y, dx=vel_x[frame]*v_scale, dy=vel_y[frame]*v_scale)
    animated_acceleration.set_data(x=point.x, y=point.y, dx=acc_x[frame]*v_scale, dy=acc_y[frame]*v_scale)
    animated_point.set_data([point.x], [point.y])

    if danger_x[frame] != 0:
        animated_danger.set_data(danger_x[:frame], danger_y[:frame])

    return animated_point,


animation = FuncAnimation(
        fig=fig,
        func=update,
        frames=len(const_t),
        interval=1
)
plt.show()