import numpy as np
from math import floor


class Point:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y


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


class Spline:
    def __init__(self, curves):
        self.curves = curves

    def get_point_norm(self, t):
        curve_index = floor(len(self.curves) * t) - 1
        curve_t = len(self.curves) * t - curve_index
        t_point = self.curves[curve_index].get_point(curve_t)

        return t_point

    def get_point(self, t):
        curve_index = floor(t) - 1
        curve_t = t - curve_index
        t_point = self.curves[curve_index].get_point(curve_t)

        return t_point
