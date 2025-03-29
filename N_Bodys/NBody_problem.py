import numpy as np
import pygame
import random

class NBodySimulator:
    def __init__(self, universe, window_size=600): #The implemented constructor for the class
        self.universe = universe
        self.window_size = window_size

    def simulate(self, time_step, trace=False):
        pygame.init()
        # Set up the drawing window
        self.screen = pygame.display.set_mode([self.window_size, self.window_size])
        pygame.display.set_caption(self.universe.name + ' {} sec./step'.format(1000))
        # Run until the user asks to quit
        running = True
        color_background = (128, 128, 128)
        color_bodies = (0, 0, 0)
        color_trace = (192, 192, 192)
        self.screen.fill(color_background)
        
        while running:
            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            if trace:
                self._draw_bodies(color_trace)
                # Show display with drawn bodies (double buffering)
                self.universe.update(time_step)
                self._draw_bodies(color_bodies)
                pygame.display.flip() # update
            else:
                self.screen.fill(color_background)
                self._draw_bodies(color_bodies)
                self.universe.update(time_step)
                pygame.display.flip() # update
        pygame.quit()
    
    def _draw_bodies(self, color, size=5.):
        for i in range(self.universe.num_bodies):
            pos = (self.window_size / 2.) * self.universe.get_body_position(i) / self.universe.radius
            pygame.draw.circle(self.screen, color, (pos + self.window_size / 2.).astype(int), size)

class Universe:
    def __init__(self, bodies, radius, name):
        self._bodies = bodies
        self._num_bodies = len(self._bodies)
        self._radius = radius
        self._name = name
    
    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
        num_bodies = int(lines[0])
        bodies = []
        radius = float(lines[1])
        for i in range(num_bodies):
            x, y, vx, vy, mass = [float(string) for string in lines[2 + i].split()]
            position = np.array([x, y])
            velocity = np.array([vx, vy])
            bodies.append(Body(position, velocity, mass))
        return cls(bodies, radius, filename)
    
    @classmethod
    def nplus1(cls, n):
        MAX_VELOCITY = 1e05
        MIN_VELOCITY = 1e04
        MAX_MASS = 1e24
        MIN_MASS = 1e22
        BIGGEST_MASS = 1e39
        num_bodies = n + 1
        RADIUS = 1e12
        name = 'n={} + 1 bodies'.format(num_bodies)
        bodies = [Body(np.zeros(2), np.zeros(2), BIGGEST_MASS)]
        
        for _ in range(n): #The implemented principal bucle
            rho = random.uniform(RADIUS / 4., RADIUS / 2.)
            angle = random.uniform(-np.pi, np.pi)

            x = np.cos(angle) * rho
            y = np.sin(angle) * rho

            vx = -1e-3 * y + random.uniform(MIN_VELOCITY, MAX_VELOCITY)
            vy = 1e-3 * x + random.uniform(MIN_VELOCITY, MAX_VELOCITY)

            mass = random.uniform(MIN_MASS, MAX_MASS)
            bodies.append(Body(np.array([x, y]), np.array([vx, vy]), mass))
        return cls(bodies, RADIUS, name)
    
    @classmethod
    def central_configuration(cls, num_bodies, angle_vel_pos=np.pi / 4, gamma=1e-4):
        RADIUS = 1e11
        MASS = 1e34
        distance = 0.5 * RADIUS
        name = 'central configuration {} bodies'.format(num_bodies)
        velocity_magnitude = gamma * distance
        bodies = []
        for i in range(num_bodies):
            angle = 2 * np.pi * i / num_bodies
            position = distance * np.array([np.cos(angle), np.sin(angle)])
            velocity = velocity_magnitude * np.array([np.cos(angle + angle_vel_pos), np.sin(angle + angle_vel_pos)])
            bodies.append(Body(position, velocity, MASS))
        return cls(bodies, RADIUS, name)
    
    #A few simple functions
    def get_body_position(self, idx_body):
        return self._bodies[idx_body].position
    
    @property
    def num_bodies(self):
        return self._num_bodies
    
    @property
    def radius(self):
        return self._radius
    
    @property
    def name(self):
        return self._name
    
    def _compute_forces(self): #The implemented function wich calculates the force for every object
        forces = [np.zeros(2) for _ in self._bodies]
        for i, body in enumerate(self._bodies):
            for j, other in enumerate(self._bodies):
                if i != j:
                    forces[i] += body.force_from(other)
        return forces
    
    def update(self, dt):
        forces = self._compute_forces()
        for i, body in enumerate(self._bodies):
            body.move(forces[i], dt)
    
    def __str__(self):
        return '\n'.join([str(b) for b in self._bodies])

class Body:
    G = 6.67e-11
    
    def __init__(self, pos, vel, mass):
        self._position = pos
        self._velocity = vel
        self._mass = mass
    
    @property
    def mass(self):
        return self._mass
    
    @property
    def position(self):
        return self._position
    
    @property
    def velocity(self):
        return self._velocity
    
    def move(self, force, dt):
        acceleration = force / self.mass
        self._velocity += acceleration * dt
        self._position += self._velocity * dt
    
    def force_from(self, another_body):
        delta = another_body.position - self._position
        dist = np.linalg.norm(delta)
        if dist == 0:
            return np.zeros(2)
        magnitude = (Body.G * another_body.mass * self.mass) / (dist * dist)
        return (magnitude * delta) / dist
    
    def __str__(self):
        return 'mass {}, position {}, velocity {}'.format(self.mass, self.position, self.velocity)

if __name__ == '__main__':
    type_universe = 1
    if type_universe == 1:
        universe = Universe.from_file('data/3body.txt')
        time_step = 1000

    if type_universe == 2:
        universe = Universe.nplus1(n=5)
        time_step = 2

    if type_universe == 3:
        universe = Universe.central_configuration(num_bodies=8,angle_vel_pos=np.pi / 4,gamma=1e-4)
        time_step = 10

    simulator = NBodySimulator(universe)
    print(simulator.universe)
    simulator.simulate(time_step, trace=True)
