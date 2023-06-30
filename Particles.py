import numpy as np
import matplotlib.pyplot as plt


class Space:
    def __init__(self, gravity=9.81):
        self.gravity = gravity


class Particle:
    def __init__(self, space: Space, mass: float, position=np.array([0.0, 0.0])):
        """
        :param space: The physical reality e.g, gravity
        :param mass: in kg
        :param position: in (x, y) form.
        """
        self.space = Space
        self.gravity = space.gravity
        self.mass = mass
        self.position = position
        self.forces = np.array([[0.0, - self.mass * self.gravity]])  # 1st column = x-axis; 2nd column = y-axis
        self.acceleration = np.array([0.0, 0.0])  # 1st column = x-axis; 2nd column = y-axis
        self.velocity = np.array([0.0, 0.0])  # 1st column = x-axis; 2nd column = y-axis

    def set_force(self, force, angle) -> None:
        """
        :param force: It should be relative to horizontal in anti-clockwise direction.
                      They can also be negative for 180 rotation.
        :param angle: Clockwise from the horizontal in `radians`.
        """
        self.forces = np.vstack([
            self.forces,
            [
                force * np.cos(angle),
                force * np.sin(angle)
            ]])

    def set_force_by_components(self, fx, fy):
        self.forces[0][0] = fx
        self.forces[0][1] = fy

    def update_acceleration(self):
        self.acceleration = self.forces.sum(axis=0)/self.mass

    def update_mass(self, mass):
        self.mass = mass
        self.forces[0] = [0.0, mass * self.gravity]

    def predict_pos_in_time(self, time_increment: float, update_values=False):
        position = self.velocity*time_increment + 0.5*(time_increment**2)*self.acceleration
        if update_values:
            self.position = self.position + position
            self.velocity = self.predict_velocity_in_time(time_increment)
        return self.position + position

    def predict_velocity_in_time(self, time_increment: float, update_values=False):
        velocity = self.acceleration*time_increment + self.velocity
        if update_values:
            self.velocity = velocity
            self.position = self.predict_pos_in_time(time_increment)
        return velocity


def model_air_resistance_for_projectile():
    # Testing model specs
    SpaceX = Space(gravity=9.81)
    car = Particle(space=SpaceX, mass=1.5)
    car.set_force(15, np.pi)
    car.update_acceleration()
    car.velocity = np.array([13 * (3 ** 0.5), 13.0])

    # Running the simulation using the preset model.
    time_against_velocity = np.array([0, car.velocity[0]])  # initialized the array
    distance_against_time = np.array([0, car.position[0]])  # initialized the array
    vertical_against_horizontal = np.array([0, 0])
    for i in range(1, 30):
        car.predict_pos_in_time(time_increment=0.1, update_values=True)
        vertical_against_horizontal = np.vstack([vertical_against_horizontal, [car.position[0], car.position[1]]])
        time_against_velocity = np.vstack([time_against_velocity, [0.1 * i, car.velocity[0]]])
        distance_against_time = np.vstack([distance_against_time, [0.1 * i, car.position[0]]])
        car.set_force(3, np.pi + (-1.5) ** i)
        car.update_acceleration()

    # Graphing the outcomes
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 3, 1, title="Velocity against Time")
    ax1.plot(time_against_velocity[:, 0], time_against_velocity[:, 1])
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Velocity")
    plt.grid()

    ax2 = fig.add_subplot(1, 3, 2, title="Displacement against Time")
    ax2.plot(distance_against_time[:, 0], distance_against_time[:, 1])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Displacement")
    plt.grid()

    ax3 = fig.add_subplot(1, 3, 3, title="Vertical Against Horizontal")
    ax3.plot(vertical_against_horizontal[:, 0], vertical_against_horizontal[:, 1])
    ax3.set_xlabel("Horizontal")
    ax3.set_ylabel("Vertical")
    plt.grid()
    plt.show()


def model_simple_force_addition():
    SpaceX = Space(gravity=9.81)
    shuttle = Particle(space=SpaceX, mass=1.5)
    shuttle.set_force_by_components(5, 10)
    shuttle.update_acceleration()

    # Running the simulation using the preset model.
    time_against_velocity = np.array([0, shuttle.velocity[0]])  # initialized the array
    distance_against_time = np.array([0, shuttle.position[0]])  # initialized the array
    vertical_against_horizontal = np.array([0, 0])
    for i in range(1, 30):
        shuttle.predict_pos_in_time(time_increment=0.1, update_values=True)
        vertical_against_horizontal = np.vstack([vertical_against_horizontal, [shuttle.position[0], shuttle.position[1]]])
        time_against_velocity = np.vstack([time_against_velocity, [0.1 * i, shuttle.velocity[0]]])
        distance_against_time = np.vstack([distance_against_time, [0.1 * i, shuttle.position[0]]])
        shuttle.update_acceleration()

        # Graphing the outcomes
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 3, 1, title="Velocity against Time")
    ax1.plot(time_against_velocity[:, 0], time_against_velocity[:, 1])
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Velocity")
    plt.grid()

    ax2 = fig.add_subplot(1, 3, 2, title="Displacement against Time")
    ax2.plot(distance_against_time[:, 0], distance_against_time[:, 1])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Displacement")
    plt.grid()

    ax3 = fig.add_subplot(1, 3, 3, title="Vertical Against Horizontal")
    ax3.plot(vertical_against_horizontal[:, 0], vertical_against_horizontal[:, 1])
    ax3.set_xlabel("Horizontal")
    ax3.set_ylabel("Vertical")
    plt.grid()
    plt.show()


def photon():
    pass


def model_drag(velocity, position, mass, k, time, increments):
    loops = int(time/increments)
    from Drag.main import Drag
    mass = mass
    SpaceX = Space(gravity=9.81)
    shuttle = Particle(space=SpaceX, mass=mass, position=position)
    shuttle.velocity = velocity
    shuttle_drag = Drag(mass=mass, pos=shuttle.position, velocity=shuttle.velocity, k=k, g=SpaceX.gravity)
    shuttle.set_force_by_components(shuttle_drag.resultant_force[0], shuttle_drag.resultant_force[1])
    shuttle.update_acceleration()

    # Running the simulation using the preset model.
    time_against_velocity = np.array([0, shuttle.velocity[0]])  # initialized the array
    distance_against_time = np.array([0, shuttle.position[0]])  # initialized the array
    vertical_against_horizontal = shuttle.position

    # New vertical things
    time_against_velocity_y = np.array([0, shuttle.velocity[1]])
    distance_against_time_y = np.array([0, shuttle.position[1]])
    for i in range(1, loops):
        shuttle.predict_pos_in_time(time_increment=increments, update_values=True)
        vertical_against_horizontal = np.vstack(
            [vertical_against_horizontal, [shuttle.position[0], shuttle.position[1]]])
        time_against_velocity = np.vstack([time_against_velocity, [0.1 * i, shuttle.velocity[0]]])
        distance_against_time = np.vstack([distance_against_time, [0.1 * i, shuttle.position[0]]])
        time_against_velocity_y = np.vstack([time_against_velocity_y, [0.1 * i, shuttle.velocity[1]]])
        distance_against_time_y = np.vstack([distance_against_time_y, [0.1 * i, shuttle.position[1]]])
        shuttle_drag.set_velocity(velocity=shuttle.velocity)
        shuttle_drag.set_position(pos=shuttle.position)
        shuttle_drag.update_resultant_acc()
        shuttle.set_force_by_components(shuttle_drag.resultant_force[0], shuttle_drag.resultant_force[1])
        shuttle.update_acceleration()

        # Graphing the outcomes
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 3, 1, title="Velocity(x) against Time")
    ax1.plot(time_against_velocity[:, 0], time_against_velocity[:, 1])
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Velocity")
    plt.grid()

    ax2 = fig.add_subplot(2, 3, 2, title="Displacement(x) against Time")
    ax2.plot(distance_against_time[:, 0], distance_against_time[:, 1])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Displacement")
    plt.grid()

    ax4 = fig.add_subplot(2, 3, 3, title="Velocity(y) against Time")
    ax4.plot(time_against_velocity_y[:, 0], time_against_velocity_y[:, 1])
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Velocity")
    plt.grid()

    ax5 = fig.add_subplot(2, 3, 4, title="Displacement(y) against Time")
    ax5.plot(distance_against_time_y[:, 0], distance_against_time_y[:, 1])
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Displacement")
    plt.grid()

    ax3 = fig.add_subplot(2, 3, 5, title="Vertical Against Horizontal")
    ax3.plot(vertical_against_horizontal[:, 0], vertical_against_horizontal[:, 1])
    ax3.set_xlabel("Horizontal")
    ax3.set_ylabel("Vertical")
    plt.grid()

    # plt.subplots_adjust(wspace=0.3, hspace=0.3, bottom=1, left=0.11, right=0.96, top=0.95)
    plt.show()


# def non_constant_acceleration():
#     Earth = Space()
#     bird = Particle(Earth, mass=0.5)
#     for i in range(2):
#         bird.velocity = np.array([])


if __name__ == '__main__':
    # model_air_resistance_for_projectile()
    # model_simple_force_addition()
    model_drag(
        position=np.array([1, 15]),
        velocity=np.array([2, 12]),
        mass=0.5, k=0.2,
        time=10, increments=0.1)
