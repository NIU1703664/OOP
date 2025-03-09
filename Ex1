import numpy as np
import pygame
import random

class Body:
    G = 6.67e-11
    
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
    
    @staticmethod
    def random_vector(a, b, dim=1):
        return a + (b - a) * 2 * (np.random.rand(dim) - 0.5)
    
    @classmethod
    def random1(cls):
        mass = Body.random_vector(1e10, 1e30)
        position = Body.random_vector(-1e20, +1e20, 2)
        velocity = Body.random_vector(1e15, +1e25, 2)
        return cls(mass, position, velocity)
    
    def _force(self, other):
        r = other.position - self.position
        d = np.linalg.norm(r)
        if d == 0:
            return np.zeros(2)
        magnitude = self.G * self.mass * other.mass / d**2
        direction = r / d
        return magnitude * direction
    
    def total_force(self, other_bodies):
        force = np.zeros(2)
        for body in other_bodies:
            force += self._force(body)
        return force
    
    def update(self, force, dt):
        acceleration = force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

class Star(Body):
    def __init__(self, mass, position, velocity, percent_mass):
        super().__init__(mass, position, velocity)
        self.percent_mass = percent_mass
    
    def update(self, force, dt):
        super().update(force, dt)
        u = 2 * (np.random.rand() - 0.5)
        self.mass *= (1 + (self.percent_mass / 100) * u)

class PygameSimulator:
    def __init__(self, bodies, window_size=600):
        self.bodies = bodies
        self.window_size = window_size
        self.space_radius = 1e12
        self.factor = self.window_size / 2 / self.space_radius
        pygame.init()
        self.screen = pygame.display.set_mode([self.window_size, self.window_size])
        pygame.display.set_caption('N-Body Simulation')
    
    def animate(self, time_step, trace=False):
        running = True
        color_background, color_body, color_trace = (128,128,128), (0,0,0), (192,192,192)
        self.screen.fill(color_background)
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            if trace:
                for body in self.bodies:
                    self._draw(body.position, color_trace)
                self._update(time_step)
                for body in self.bodies:
                    self._draw(body.position, color_body)
                pygame.display.flip()
            else:
                self.screen.fill(color_background)
                for body in self.bodies:
                    self._draw(body.position, color_body)
                self._update(time_step)
                pygame.display.flip()
        pygame.quit()
    
    def _draw(self, position_space, color, size=5):
        position_pixels = (self.factor * position_space + self.window_size / 2).astype(int)
        pygame.draw.circle(self.screen, color, position_pixels, size)
    
    def _update(self, time_step):
        forces = [b.total_force([other for other in self.bodies if other is not b]) for b in self.bodies]
        for b, f in zip(self.bodies, forces):
            b.update(f, time_step)

if __name__ == "__main__":
    bodies = [
        Body(5.97e24, [0, 0], [5e02, 0]),
        Star(1.989e30, [0, 4.5e10], [3.0e04, 0], 1.0),
        Star(1.989e30, [0, -4.5e10], [-3.0e04, 0], 1.0)
    ]
    dt = 5000
    sim = PygameSimulator(bodies)
    sim.animate(dt, trace=True)

