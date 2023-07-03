import numpy as np
from Particles import Space, Particle, Tracker


class CollisionsSpace:
    def __init__(self):
        self.particles = np.ndarray(shape=(1, 2))
        self.e = {}

    def add_particle(self, particle: Particle, radius: float):
        if particle in self.particles:
            raise Exception("Particle Already exists inside the CollisionSpace!")
        self.particles = np.vstack([self.particles, [particle, radius]])

    def update_e(self, particle1: Particle, particle2: Particle, e: float) -> None:
        if not(particle1 in self.particles[:, 0] and particle2 in self.particles[:, 0]):
            raise Exception("One of the particle does not exists in CollisionSpace!")
        if hash(particle1) > hash(particle2):
            self.e[f"{hash(particle1)}{hash(particle2)}"] = e
            return
        self.e[f"{hash(particle2)}{hash(particle1)}"] = e

    def get_e(self, particle1: Particle, particle2: Particle) -> float:
        if hash(particle1) > hash(particle2):
            return self.e.get(f"{hash(particle1)}{hash(particle2)}", 1)
        return self.e.get(f"{hash(particle2)}{hash(particle1)}", 1)

    def debug_temp(self):
        self.particles[1][0].space.gravity = 12

