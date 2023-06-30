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


def model_air_resistance_for_projectile(wind_force, wind_angle, particle_velocity, time=30, increments=0.1):
    loops = int(time / increments)
    # Testing model specs
    SpaceX = Space(gravity=9.81)
    car = Particle(space=SpaceX, mass=1.5)
    car.set_force(wind_force, wind_angle)
    car.update_acceleration()
    car.velocity = particle_velocity

    # Running the simulation using the preset model.
    time_against_velocity = np.array([0, car.velocity[0]])  # initialized the array
    distance_against_time = np.array([0, car.position[0]])  # initialized the array
    vertical_against_horizontal = np.array([0, 0])
    for i in range(1, loops):
        car.predict_pos_in_time(time_increment=increments, update_values=True)
        vertical_against_horizontal = np.vstack([vertical_against_horizontal, [car.position[0], car.position[1]]])
        time_against_velocity = np.vstack([time_against_velocity, [0.1 * i, car.velocity[0]]])
        distance_against_time = np.vstack([distance_against_time, [0.1 * i, car.position[0]]])
        # car.set_force(3, np.pi + (-1.5) ** i)
        car.update_acceleration()

    # Graphing the outcomes
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 3, 1, title="Velocity(x) against Time")
    ax1.plot(time_against_velocity[:, 0], time_against_velocity[:, 1])
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Velocity(x)")
    plt.grid()

    ax2 = fig.add_subplot(1, 3, 2, title="Displacement(x) against Time")
    ax2.plot(distance_against_time[:, 0], distance_against_time[:, 1])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Displacement(x)")
    plt.grid()

    ax3 = fig.add_subplot(1, 3, 3, title="Trajectory")
    ax3.plot(vertical_against_horizontal[:, 0], vertical_against_horizontal[:, 1])
    ax3.set_xlabel("Horizontal")
    ax3.set_ylabel("Vertical")
    plt.grid()

    plt.subplots_adjust(wspace=0.3)
    plt.show()


def photon():
    pass


def model_drag(velocity, position, mass, k, time=5, increments=0.01):
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
    time_against_velocity_x = np.array([0, shuttle.velocity[0]])  # initialized the array
    distance_against_time_x = np.array([0, shuttle.position[0]])  # initialized the array
    vertical_against_horizontal = shuttle.position

    # New vertical graphs
    time_against_velocity_y = np.array([0, shuttle.velocity[1]])
    distance_against_time_y = np.array([0, shuttle.position[1]])
    for i in range(1, loops):
        shuttle.predict_pos_in_time(time_increment=increments, update_values=True)
        vertical_against_horizontal = np.vstack(
            [vertical_against_horizontal, [shuttle.position[0], shuttle.position[1]]])
        time_against_velocity_x = np.vstack([time_against_velocity_x, [0.1 * i, shuttle.velocity[0]]])
        distance_against_time_x = np.vstack([distance_against_time_x, [0.1 * i, shuttle.position[0]]])
        time_against_velocity_y = np.vstack([time_against_velocity_y, [0.1 * i, shuttle.velocity[1]]])
        distance_against_time_y = np.vstack([distance_against_time_y, [0.1 * i, shuttle.position[1]]])
        shuttle_drag.set_velocity(velocity=shuttle.velocity)
        shuttle_drag.set_position(pos=shuttle.position)
        shuttle_drag.update_resultant_force()
        shuttle.set_force_by_components(shuttle_drag.resultant_force[0], shuttle_drag.resultant_force[1])
        shuttle.update_acceleration()

    plot_all_graphs(
            time_against_velocity_x, distance_against_time_x,
            time_against_velocity_y, distance_against_time_y,
            vertical_against_horizontal)


def plot_all_graphs(
        time_against_velocity_x, distance_against_time_x,
        time_against_velocity_y, distance_against_time_y,
        vertical_against_horizontal):
    # Graphing the outcomes
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(3, 2, 1)  # Velocity(x) against Time"
    ax1.plot(time_against_velocity_x[:, 0], time_against_velocity_x[:, 1])
    ax1.set_xlabel("Time/s")
    ax1.set_ylabel(r"Velocity(x)/$ms^{-1}$")
    plt.axvline(x=0, c="black")
    plt.axhline(y=0, c="black")
    plt.grid()

    ax2 = fig.add_subplot(3, 2, 2)  # Displacement(x) against Time
    ax2.plot(distance_against_time_x[:, 0], distance_against_time_x[:, 1])
    ax2.set_xlabel("Time/s")
    ax2.set_ylabel("Displacement(x)/m")
    plt.axvline(x=0, c="black")
    plt.axhline(y=0, c="black")
    plt.grid()

    ax4 = fig.add_subplot(3, 2, 3)  # Velocity(y) against Time
    ax4.plot(time_against_velocity_y[:, 0], time_against_velocity_y[:, 1])
    ax4.set_xlabel("Time/s")
    ax4.set_ylabel(r"Velocity(y)/$ms^{-1}$")
    plt.axvline(x=0, c="black")
    plt.axhline(y=0, c="black")
    plt.grid()

    ax5 = fig.add_subplot(3, 2, 4)  # Displacement(y) against Time
    ax5.plot(distance_against_time_y[:, 0], distance_against_time_y[:, 1])
    ax5.set_xlabel("Time/s")
    ax5.set_ylabel("Displacement(y)/m")
    plt.axvline(x=0, c="black")
    plt.axhline(y=0, c="black")
    plt.grid()

    ax3 = fig.add_subplot(3, 1, 3, title="Trajectory")  # Vertical Against Horizontal
    ax3.plot(vertical_against_horizontal[:, 0], vertical_against_horizontal[:, 1])
    ax3.set_xlabel("x/m")
    ax3.set_ylabel("y/m")
    plt.axvline(x=0, c="black")
    plt.axhline(y=0, c="black")
    plt.grid()

    plt.subplots_adjust(hspace=0.3, top=0.93, bottom=0.08)
    plt.show()

# def non_constant_acceleration():
#     Earth = Space()
#     bird = Particle(Earth, mass=0.5)
#     for i in range(2):
#         bird.velocity = np.array([])


if __name__ == '__main__':
    # model_air_resistance_for_projectile(
    #     wind_force=15, wind_angle=np.pi,
    #     particle_velocity=np.array([15.0, 23.0]),
    #     time=5, increments=0.01)

    model_drag(
        position=np.array([0, 0]),
        velocity=np.array([12, 6]),
        mass=0.5, k=0.2,
        time=3, increments=0.01)
