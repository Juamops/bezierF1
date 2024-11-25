import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from bCurve import Bezier, Point, Spline
import numpy as np
from PIL import Image

track_spline = Spline([
    Bezier(Point(251, 626), Point(267, 641), Point(286.97620407928537, 658.0011267666531), Point(308, 659)),
    Bezier(Point(308, 659), Point(329.02379592071463, 659.9988732333469), Point(358.48498177436346, 655.1220565995621),
           Point(386, 650)),
    Bezier(Point(386, 650), Point(413.51501822563654, 644.8779434004379), Point(435.34014682305747, 639.276416871691),
           Point(461, 627)),
    Bezier(Point(461, 627), Point(486.65985317694253, 614.723583128309), Point(517.8956587742977, 598.0903132675311),
           Point(537, 576)),
    Bezier(Point(537, 576), Point(556.1043412257023, 553.9096867324689), Point(564.0, 528.0), Point(571, 508)),
    Bezier(Point(571, 508), Point(578.0, 488.0), Point(545.0, 410.0), Point(550, 380)),
    Bezier(Point(550, 380), Point(555.0, 350.0), Point(608.0, 317.0), Point(620, 280)),
    Bezier(Point(620, 280), Point(632.0, 243.0), Point(623.0, 224.0), Point(635, 193)),
    Bezier(Point(635, 193), Point(647.0, 162.0), Point(662.0033989403206, 144.3387909605964), Point(699, 115)),
    Bezier(Point(699, 115), Point(735.9966010596794, 85.66120903940359), Point(771, 64), Point(793, 47)),
])

in_to_p = 404/16583
m_to_in = 39.3701
kmh_to_ms = 1/3.6
kmh_to_p = m_to_in * in_to_p * kmh_to_ms

track = np.asarray(Image.open('track.png'))
x, y = track_spline.get_points(0, 1, 100)
x_d, y_d = track_spline.get_first_derivatives(0, 1, 100)
x_d2, y_d2 = track_spline.get_second_derivatives(0, 1, 100)

const_t, const_time = track_spline.traverse_const_vel(0, 1, 500*kmh_to_p, timestep=1/30)

fig, ax = plt.subplots()
ax.imshow(track)

# animated_track, = ax.plot([], [], lw=3)
animated_point, = ax.plot(x[0], y[0], color='r', marker='o', markersize=6)
# animated_velocity = ax.arrow(x[0], y[0], x_d[0], y_d[0], color='b', width=2, head_width=10)
# animated_acceleration = ax.arrow(x[0], y[0], x_d2[0], y_d2[0], color='g', width=2, head_width=10)


def update(frame):
    # animated_plot.set_linewidth(3)
    # animated_track.set_data(x[:frame], y[:frame])
    # animated_velocity.set_data(x=x[frame], y=y[frame], dx=x_d[frame], dy=y_d[frame])
    # animated_acceleration.set_data(x=x[frame], y=y[frame], dx=x_d2[frame], dy=y_d2[frame])
    point = track_spline.get_point_norm(const_t[frame])
    animated_point.set_data([point.x], [point.y])
    return animated_point,


print(len(const_t))
animation = FuncAnimation(
        fig=fig,
        func=update,
        frames=len(const_t),
        interval=1000/30
)
plt.show()
