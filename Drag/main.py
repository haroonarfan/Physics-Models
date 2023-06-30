import numpy as np


g = 9.81  # g/ms-2


class Drag:
    def __init__(self, mass, pos, velocity, k, g=g):
        self.g = g
        self.k = k
        self.mass = mass
        self.pos = pos
        self.velocity = velocity
        self.weight = np.array([0, -self.mass * self.g])
        self.drag = self.k * np.where(self.velocity <= 0, 1, -1) * (self.velocity ** 2)
        self.resultant_force = self.weight + self.drag
        # self.resultant_acc = self.resultant_force/self.mass

    def update_resultant_force(self):
        self.drag = self.k * np.where(self.velocity <= 0, 1, -1) * (self.velocity ** 2)
        self.resultant_force = self.weight + self.drag
        # self.resultant_acc = self.resultant_force / self.mass

    def set_velocity(self, velocity):
        self.velocity = velocity

    def set_position(self, pos):
        self.pos = pos


# mass = 1  # m/kg
# weight = np.array([0, -mass*g])  # downwards acting weight
# pos = np.array([1, 1])  # d/m
# v = np.array([2, 2])  # v/ms-1
#
# drag = -k*(v**2)
#
# resultant_force = weight + drag
# resultant_acc = resultant_force/mass

# time = np.linspace(0, 10)
# print(
#     f"drag = {drag}\n"
#     f"weight = {weight}\n"
#     f"resultant force = {resultant_force}\n"
#     f"resultant acceleration = {resultant_acc}\n"
#     f"time = {time}"
# )


