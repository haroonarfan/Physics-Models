import numpy as np
from Particles import Space, Particle, Tracker
# Kinematics
# 1. What do I know?
# 	 WEP, ConEn, Equations of motions, Tensions, Newton’s 1st 2nd and 3rd law,
# 	 acceleration graphs, friction, Experimental Law
# 2. What is feasible or something that I should make in python?
# 	    i. Graphs of displacement against velocity for acceleration in directions,
# 	    ii. Make graphs for the trajectory of projectiles in wind
# 	    iii. Make collision simulations - if possible.
# 	    iv. Make velocity/acceleration/displacement/distance/time etc predictions.
# 	    v. Friction modelled
# 	    vi. If possible allow GUI models.
# Objective: Make a simulation


def model_air_resistance_for_projectile(wind_force, wind_angle, particle_velocity, time=30, increments=0.1):
    loops = int(time / increments)
    # Testing model specs
    SpaceX = Space(gravity=9.81)
    frisbee = Particle(space=SpaceX, mass=1.5)
    frisbee.add_force(wind_force, wind_angle, label='Wind')
    frisbee.update_velocity(particle_velocity[0], particle_velocity[1])
    frisbee_tracker = Tracker(frisbee)

    for i in range(1, loops):
        frisbee.predict_pos_in_time(time_increment=increments, update_values=True)
        frisbee_tracker.update(time=i*increments)
        # car.set_force(3, np.pi + (-1.5) ** i)
    frisbee_tracker.plot_all_graphs()


def model_drag(velocity, position, mass, k, time=5, increments=0.01):
    loops = int(time / increments)
    SpaceX = Space(gravity=9.81)
    shuttle = Particle(space=SpaceX, mass=mass, position=position, drag_coefficient=k)
    shuttle.update_velocity(velocity[0], velocity[1])
    shuttle_tracker = Tracker(particle=shuttle)

    for i in range(1, loops):
        shuttle.predict_pos_in_time(time_increment=increments, update_values=True)
        shuttle_tracker.update(time=i*increments)
    shuttle_tracker.plot_all_graphs()
    shuttle.print_forces()
    shuttle.print_velocity()


def photon():
    pass

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
        velocity=np.array([28, 124]),
        mass=2.5, k=0.3,
        time=5, increments=0.01)
