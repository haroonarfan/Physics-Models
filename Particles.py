import numpy as np
import matplotlib.pyplot as plt


class Space:
    def __init__(self, gravity):
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


if __name__ == '__main__':
    # Testing model specs
    SpaceX = Space(gravity=9.81)
    car = Particle(space=SpaceX, mass=10)
    car.set_force(60, 0)
    car.update_acceleration()
    car.velocity = np.array([-24.0, 13.0])

    # Running the simulation using the preset model.
    time_against_velocity = np.array([0, car.velocity[0]])  # initialized the array
    distance_against_time = np.array([0, car.position[0]])  # initialized the array
    for i in range(1, 100):
        car.predict_pos_in_time(time_increment=0.1, update_values=True)
        time_against_velocity = np.vstack([time_against_velocity, [0.1 * i, car.velocity[0]]])
        distance_against_time = np.vstack([distance_against_time, [0.1 * i, car.position[0]]])

    # Graphing the outcomes
    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1, title="Velocity against Time")
    ax1.plot(time_against_velocity[:, 0], time_against_velocity[:, 1])
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Velocity")

    ax2 = fig.add_subplot(1, 2, 2, title="Displacement against Time")
    ax2.plot(distance_against_time[:, 0], distance_against_time[:, 1])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Displacement")

    plt.show()
