import numpy as np
import matplotlib.pyplot as plt


class Space:
    def __init__(self, gravity=9.81):
        self.gravity = gravity


class Particle:
    def __init__(self, space: Space, mass: float, position=np.array([0.0, 0.0]), drag_coefficient=0):
        """
        :param space: The physical reality e.g, gravity
        :param mass: in kg
        :param position: in (x, y) form.
        """
        self.space = Space
        self.gravity = space.gravity
        self.mass = mass
        self.position = position
        self.forces = np.array([
            [0.0, - self.mass * self.gravity],
            [0.0, 0.0]])  # 1st column = x-axis; 2nd column = y-axis
        self.drag, self.drag_coefficient = np.array([0.0, 0.0]), drag_coefficient
        self.acceleration = np.array([0.0, 0.0])  # 1st column = x-axis; 2nd column = y-axis
        self.velocity = np.array([0.0, 0.0])  # 1st column = x-axis; 2nd column = y-axis
        self.forces_labels = np.array(['Weight', 'Drag'])  # These are in order of occurrences

    def update_drag(self):
        self.drag = self.drag_coefficient * np.where(self.velocity <= 0, 1, -1) * (self.velocity ** 2)
        self.forces[1] = self.drag
        self.update_acceleration()

    def update_velocity(self, vx, vy):
        self.velocity = np.array([vx, vy])
        self.update_drag()

    def add_force(self, force, angle, label='unknown') -> None:
        """
        :param force: It should be relative to horizontal in anti-clockwise direction.
                      They can also be negative for 180 rotation.
        :param angle: Clockwise from the horizontal in `radians`.
        :param label: Force name to add for the labels
        """
        self.forces = np.vstack([
            self.forces,
            [
                force * np.cos(angle),
                force * np.sin(angle)
            ]])
        np.append(self.forces_labels, label)
        self.update_acceleration()

    def add_force_by_components(self, fx, fy, label='unknown') -> None:
        self.forces = np.vstack([
            self.forces,
            [
                fx,
                fy
            ]])
        np.append(self.forces_labels, label)
        self.update_acceleration()

    def set_force_by_components(self, fx, fy, label='unknown'):  # This over-rides all other forces.
        self.forces[0][0] = fx
        self.forces[0][1] = fy
        self.forces_labels = np.array([label])
        self.update_acceleration()

    def update_acceleration(self):
        self.acceleration = self.forces.sum(axis=0)/self.mass

    def update_mass(self, mass):
        self.mass = mass
        self.forces[0] = [0.0, mass * self.gravity]
        self.update_acceleration()

    def predict_pos_in_time(self, time_increment: float, update_values=False):
        position = self.velocity*time_increment + 0.5*(time_increment**2)*self.acceleration
        if update_values:
            self.position = self.position + position
            self.velocity = self.predict_velocity_in_time(time_increment)
            self.update_drag()
        return self.position + position

    def predict_velocity_in_time(self, time_increment: float, update_values=False):
        velocity = self.acceleration*time_increment + self.velocity
        if update_values:
            self.velocity = velocity
            self.position = self.predict_pos_in_time(time_increment)
            self.update_drag()
        return velocity


class Tracker:
    def __init__(self, particle: Particle):
        self.particle = particle
        self.time_against_velocity_x = np.array([0, particle.velocity[0]])
        self.distance_against_time_x = np.array([0, particle.position[0]])
        self.time_against_velocity_y = np.array([0, particle.velocity[1]])
        self.distance_against_time_y = np.array([0, particle.position[1]])
        self.vertical_against_horizontal = particle.position

    def update(self, time):
        """
        :param time: It should be linear
        :return: None
        """
        self.vertical_against_horizontal = np.vstack(
            [self.vertical_against_horizontal, [self.particle.position[0], self.particle.position[1]]])
        self.time_against_velocity_x = np.vstack([self.time_against_velocity_x, [time, self.particle.velocity[0]]])
        self.distance_against_time_x = np.vstack([self.distance_against_time_x, [time, self.particle.position[0]]])
        self.time_against_velocity_y = np.vstack([self.time_against_velocity_y, [time, self.particle.velocity[1]]])
        self.distance_against_time_y = np.vstack([self.distance_against_time_y, [time, self.particle.position[1]]])

    def plot_all_graphs(self):
        # Graphing the outcomes
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(3, 2, 1)  # Velocity(x) against Time
        ax1.plot(self.time_against_velocity_x[:, 0], self.time_against_velocity_x[:, 1])
        ax1.set_xlabel(r"Time/s")
        ax1.set_ylabel(r"Velocity(x)/$ms^{-1}$")
        plt.axvline(x=0, c="black")
        plt.axhline(y=0, c="black")
        plt.grid()

        ax2 = fig.add_subplot(3, 2, 2)  # Displacement(x) against Time
        ax2.plot(self.distance_against_time_x[:, 0], self.distance_against_time_x[:, 1])
        ax2.set_xlabel(r"Time/s")
        ax2.set_ylabel("Displacement(x)/m")
        plt.axvline(x=0, c="black")
        plt.axhline(y=0, c="black")
        plt.grid()

        ax4 = fig.add_subplot(3, 2, 3)  # Velocity(y) against Time
        ax4.plot(self.time_against_velocity_y[:, 0], self.time_against_velocity_y[:, 1])
        ax4.set_xlabel(r"Time/s")
        ax4.set_ylabel(r"Velocity(y)/$ms^{-1}$")
        plt.axvline(x=0, c="black")
        plt.axhline(y=0, c="black")
        plt.grid()

        ax5 = fig.add_subplot(3, 2, 4)  # Displacement(y) against Time
        ax5.plot(self.distance_against_time_y[:, 0], self.distance_against_time_y[:, 1])
        ax5.set_xlabel(r"Time/s")
        ax5.set_ylabel("Displacement(y)/m")
        plt.axvline(x=0, c="black")
        plt.axhline(y=0, c="black")
        plt.grid()

        ax3 = fig.add_subplot(3, 1, 3, title="Trajectory")  # Vertical Against Horizontal
        ax3.plot(self.vertical_against_horizontal[:, 0], self.vertical_against_horizontal[:, 1])
        ax3.set_xlabel("x/m")
        ax3.set_ylabel("y/m")
        plt.axvline(x=0, c="black")
        plt.axhline(y=0, c="black")
        plt.grid()

        plt.subplots_adjust(hspace=0.3, top=0.93, bottom=0.08)
        plt.show()
