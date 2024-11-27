import numpy as np
from math import floor
import matplotlib.pyplot as plt


in_to_p = 404/16583
m_to_in = 39.3701
kmh_to_ms = 1/3.6
kmh_to_p = kmh_to_ms * m_to_in * in_to_p


def distance(p1, p2):
    return (((p1.x - p2.x) ** 2) + ((p1.y - p2.y) ** 2)) ** 0.5


def hypot(x, y):
    return (x**2 + y**2)**0.5


class Point:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def disp_to(self, p2):
        return (p2.x - self.x), (p2.y - self.y)

    def make_string(self):
        return f'Point({self.x}, {self.y})'


class Bezier:
    start = Point(0, 0)
    cp1 = Point(0, 0)
    cp2 = Point(0, 0)
    end = Point(0, 0)

    def __init__(self, p1, p2, p3, p4):
        self.start = p1
        self.cp1 = p2
        self.cp2 = p3
        self.end = p4

    def lerp(self, p1, p2, t):
        x = p1.x + (p2.x - p1.x) * t
        y = p1.y + (p2.y - p1.y) * t

        return Point(x, y)

    def get_point(self, t):
        p12 = self.lerp(self.start, self.cp1, t)
        p34 = self.lerp(self.cp2, self.end, t)
        p_final = self.lerp(p12, p34, t)

        return p_final

    def get_poly_point(self, t):
        p0 = self.start
        p1 = self.cp1
        p2 = self.cp2
        p3 = self.end

        x_coord = \
            p0.x + \
            t**1 * (-3*p0.x + 3*p1.x) + \
            t**2 * (3*p0.x - 6*p1.x + 3*p2.x) + \
            t**3 * (-p0.x + 3*p1.x - 3*p2.x + p3.x)

        y_coord = \
            p0.y + \
            t**1 * (-3*p0.y + 3*p1.y) + \
            t**2 * (3*p0.y - 6*p1.y + 3*p2.y) + \
            t**3 * (-p0.y + 3*p1.y - 3*p2.y + p3.y)

        return Point(x_coord, y_coord)

    def get_first_derivative(self, t):
        p0 = self.start
        p1 = self.cp1
        p2 = self.cp2
        p3 = self.end

        x = \
            (-3*p0.x + 3*p1.x) + \
            2 * t * (3*p0.x - 6*p1.x + 3*p2.x) + \
            3 * (t**2) * (-p0.x + 3*p1.x - 3*p2.x + p3.x)

        y = \
            (-3*p0.y + 3*p1.y) + \
            2 * t * (3*p0.y - 6*p1.y + 3*p2.y) + \
            3 * (t**2) * (-p0.y + 3*p1.y - 3*p2.y + p3.y)

        return x, y

    def get_second_derivative(self, t):
        p0 = self.start
        p1 = self.cp1
        p2 = self.cp2
        p3 = self.end

        x = \
            2 * (3 * p0.x - 6 * p1.x + 3 * p2.x) + \
            6 * t * (-p0.x + 3 * p1.x - 3 * p2.x + p3.x)

        y = \
            2 * (3 * p0.y - 6 * p1.y + 3 * p2.y) + \
            6 * t * (-p0.y + 3 * p1.y - 3 * p2.y + p3.y)

        return x, y

    def get_instantaneous_radius(self, t):
        x1, y1 = self.get_first_derivative(t)
        x2, y2 = self.get_second_derivative(t)

        radius = abs(((x1**2 + y1**2)**(3/2)) / ((x1 * y2) - (y1 * x2)))

        return radius

    def get_control_points(self):
        x_coords = [self.start.x, self.cp1.x, self.cp2.x, self.end.x]
        y_coords = [self.start.y, self.cp1.y, self.cp2.y, self.end.y]

        return x_coords, y_coords

    def make_string(self):
        return f'Bezier({self.start.make_string()}, {self.cp1.make_string()}, ' \
               f'{self.cp2.make_string()}, {self.end.make_string()})'


class Spline:
    def __init__(self, curves):
        self.curves = curves

    def get_point(self, t):
        curve_index = floor(t) if t < len(self.curves) else len(self.curves) - 1
        curve_t = t - curve_index
        t_point = self.curves[curve_index].get_poly_point(curve_t)

        return t_point

    def get_point_norm(self, t):
        curve_index = floor(len(self.curves) * t) if floor(len(self.curves) * t) < len(self.curves) else len(self.curves) - 1
        curve_t = len(self.curves) * t - curve_index
        t_point = self.curves[curve_index].get_poly_point(curve_t)

        return t_point

    def get_point_norm_m(self, t):
        curve_index = floor(len(self.curves) * t) if floor(len(self.curves) * t) < len(self.curves) else len(self.curves) - 1
        curve_t = len(self.curves) * t - curve_index
        t_point = self.curves[curve_index].get_poly_point(curve_t)
        t_point.x *= (1/in_to_p) * (1/m_to_in)
        t_point.y *= (1 / in_to_p) * (1 / m_to_in)

        return t_point

    def get_first_derivative_norm(self, t):
        curve_index = floor(len(self.curves) * t) if floor(len(self.curves) * t) < len(self.curves) else len(self.curves) - 1
        curve_t = len(self.curves) * t - curve_index
        x, y = self.curves[curve_index].get_first_derivative(curve_t)
        t_point = Point(x, y)

        return t_point

    def get_second_derivative_norm(self, t):
        curve_index = floor(len(self.curves) * t) if floor(len(self.curves) * t) < len(self.curves) else len(self.curves) - 1
        curve_t = len(self.curves) * t - curve_index
        x, y = self.curves[curve_index].get_second_derivative(curve_t)
        t_point = Point(x, y)

        return t_point

    def get_instant_radius_norm(self, t):
        curve_index = floor(len(self.curves) * t) if floor(len(self.curves) * t) < len(self.curves) else len(self.curves) - 1
        curve_t = len(self.curves) * t - curve_index
        r = self.curves[curve_index].get_instantaneous_radius(curve_t)

        return r

    def get_max_vel(self, t, friction):
        r = self.get_instant_radius_norm(t)
        r *= (16583/404) * (1/39.3701)
        g = 9.81

        return (r * g * friction)**0.5

    def get_points(self, start, end, n):
        x_coords = []
        y_coords = []
        for t in np.linspace(start, end, n):
            point_t = self.get_point_norm(t)
            x_coords.append(point_t.x)
            y_coords.append(point_t.y)

        return x_coords, y_coords

    def get_first_derivatives(self, start, end, n):
        x = []
        y = []
        for t in np.linspace(start, end, n):
            point_t = self.get_first_derivative_norm(t)
            x.append(point_t.x)
            y.append(point_t.y)

        return x, y

    def get_second_derivatives(self, start, end, n):
        x = []
        y = []
        for t in np.linspace(start, end, n):
            point_t = self.get_second_derivative_norm(t)
            x.append(point_t.x)
            y.append(point_t.y)

        return x, y

    def get_control_points(self):
        x_coords = []
        y_coords = []
        for curve in self.curves:
            curve_x, curve_y = curve.get_control_points()
            x_coords += curve_x
            y_coords += curve_y

        return x_coords, y_coords

    def get_control_lines(self):
        x_coords = []
        y_coords = []
        for curve in self.curves:
            x_coords.append([curve.start.x, curve.cp1.x])
            y_coords.append([curve.start.y, curve.cp1.y])
            x_coords.append([curve.cp2.x, curve.end.x])
            y_coords.append([curve.cp2.y, curve.end.y])

        return x_coords, y_coords

    def get_both_ends(self):
        x_coords = []
        y_coords = []
        for curve in self.curves:
            x_coords.append(curve.start.x)
            y_coords.append(curve.start.y)
            x_coords.append(curve.end.x)
            y_coords.append(curve.end.y)

        return x_coords, y_coords

    def get_best_control_line(self, center, p1, p2):
        theta1 = np.arctan((center.y - p1.y) / (center.x - p1.x))
        theta2 = np.arctan((center.y - p2.y) / (center.x - p2.x))

        r1 = ((center.y - p1.y)**2 + (center.x - p1.x)**2)**0.5
        r2 = ((center.y - p2.y)**2 + (center.x - p2.x)**2)**0.5

        theta = (theta1 + theta2) / 2
        slope = np.tan(theta)
        length = (r1 + r2) / 2
        x_disp = length * np.cos(theta)

        x1 = center.x - x_disp
        y1 = center.y - (slope * x_disp)

        x2 = center.x + x_disp
        y2 = center.y + (slope * x_disp)

        return Point(x1, y1), Point(x2, y2)

    def make_continuous(self, inplace=True):
        if inplace:
            for i, curve in enumerate(self.curves):
                if i > 0:
                    p1, p2 = self.get_best_control_line(curve.start, self.curves[i-1].cp2, curve.cp1)
                    self.curves[i-1].cp2 = p1
                    curve.cp1 = p2
        else:
            new_spline = Spline(self.curves.copy())
            for i, curve in enumerate(new_spline.curves):
                if i > 0:
                    p1, p2 = new_spline.get_best_control_line(curve.start, new_spline.curves[i-1].cp2, curve.cp1)
                    new_spline.curves[i-1].cp2 = p1
                    curve.cp1 = p2
            return new_spline

    def step(self, origin, wanted_distance, iter):
        para_v = self.get_first_derivative_norm(origin)
        v_mag = hypot(para_v.x, para_v.y) * len(self.curves)
        est_dist = wanted_distance / v_mag
        step_size = est_dist / 10000
        dist = 0
        t = origin
        while dist < wanted_distance:
            t += step_size
            dist = distance(self.get_point_norm(origin), self.get_point_norm(t))

        # plt.scatter(iter, distance(self.get_point_norm(origin), self.get_point_norm(t - (step_size / 2))), color='g')
        return t - (step_size / 2), distance(self.get_point_norm(origin), self.get_point_norm(t - (step_size / 2)))

    def length(self, velocity):
        vals, tot_time = self.traverse_const_vel(0, 1, velocity)
        total_time = tot_time[-1]
        return total_time * velocity

    def traverse_const_vel(self, start, end, velocity, timestep=0.01):
        t = start
        t_vals = [t]
        dist = velocity * timestep
        total_time = [0]
        iter = 0
        dists = []
        while t < end:
            t, real_dist = self.step(t, dist, iter)
            dists.append(real_dist)
            t_vals.append(t)
            total_time.append(total_time[-1] + timestep)
            iter += 1

        # plt.plot([0, iter], [dist, dist])
        # plt.show()
        return t_vals, total_time, dists

    def traverse_fv_constraint(self, start, end, f_coefficient, max_vel, timestep=0.01):
        t = start
        g = 9.81
        tan_vmax = (f_coefficient * g * self.get_instant_radius_norm(t))
        velocity = min(max_vel, tan_vmax)
        dist = velocity * timestep

        t_vals = [t]
        total_time = [0]
        iter = 0

        while t < end:
            tan_vmax = (f_coefficient * g * self.get_instant_radius_norm(t))**0.5
            velocity = min(max_vel, tan_vmax)
            dist = velocity * timestep
            t, d = self.step(t, dist, iter)
            total_time.append(total_time[-1] + timestep)
            t_vals.append(t)

        return t_vals, total_time

    def join_with(self, spline):
        start = self.curves[-1].end
        end = spline.curves[0].start

        x_disp_cp1, y_disp_cp1 = self.curves[-1].cp2.disp_to(start)
        x_disp_cp2, y_disp_cp2 = spline.curves[0].cp1.disp_to(end)

        cp1 = Point(start.x + x_disp_cp1, start.y + y_disp_cp1)
        cp2 = Point(end.x + x_disp_cp2, end.y + y_disp_cp2)

        connection = Bezier(start, cp1, cp2, end)

        spline.curves.insert(0, connection)
        self.curves += spline.curves

    def add_curve(self, curve):
        tmp_spline = Spline([curve])
        self.join_with(tmp_spline)

    def print_spline(self):
        print('Spline([')
        for curve in self.curves:
            print(curve.make_string() + ',')
        print('])')

    def get_movement_vectors(self, t_vals, time_vals, timestep=0.01):
        look_forward = 1
        vel_x = []
        vel_y = []
        for i in range(len(t_vals) - look_forward):
            p1 = self.get_point_norm(t_vals[i])
            p2 = self.get_point_norm(t_vals[i+look_forward])
            vx = (p2.x - p1.x) / (time_vals[i+look_forward] - time_vals[i])
            vy = (p2.y - p1.y) / (time_vals[i+look_forward] - time_vals[i])
            vx *= (1/in_to_p) * (1/m_to_in)
            vy *= (1 / in_to_p) * (1 / m_to_in)
            vel_x.append(vx)
            vel_y.append(vy)

        for k in range(look_forward):
            vel_x.append(vel_x[-1])
            vel_y.append(vel_y[-1])

        look_forward = 1
        acc_x = []
        acc_y = []

        for i in range(len(t_vals) - look_forward):
            # point = self.get_second_derivative_norm(t_vals[i])
            # magnitude = (point.x**2 + point.y**2)**0.5
            # point.x = point.x / magnitude
            # point.y = point.y / magnitude
            #
            # r = self.get_instant_radius_norm(t_vals[i])
            # new_mag = (vel_x[i]**2 + vel_y[i]**2) / r
            # point.x = point.x * new_mag
            # point.y = point.y * new_mag
            #
            # acc_x.append(point.x)
            # acc_y.append(point.y)

            acc_y.append((vel_y[i+look_forward] - vel_y[i]) / (time_vals[i+look_forward] - time_vals[i]))
            acc_x.append((vel_x[i+look_forward] - vel_x[i]) / (time_vals[i+look_forward] - time_vals[i]))

        for k in range(look_forward):
            acc_x.append(acc_x[0])
            acc_y.append(acc_y[0])

        return vel_x, vel_y, acc_x, acc_y

