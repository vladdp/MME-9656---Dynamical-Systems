import numpy as np
import pygame


NUM_BIRDS = 100     # Number stated in assignment
WIDTH = 600
HEIGHT = 450
FPS = 60
dt = 1 / FPS
ETA = 0.5
r_nbr = 100

pygame.init()

screen = pygame.display.set_mode([WIDTH, HEIGHT])
clock = pygame.time.Clock()
pygame.display.set_caption("Sparrow Flocking Simulation")

texture = pygame.image.load('sparrow.png').convert_alpha()

class Sparrow:
    def __init__(self, x, y, theta):
        self.pos = pygame.Vector2(x, y)
        self.theta = theta
        self.speed = 200
        self.vel = pygame.Vector2(self.speed * np.cos(np.deg2rad(self.theta)),
                                  self.speed * -np.sin(np.deg2rad(self.theta)))
        
    def update_theta(self, theta_i):
        self.theta += theta_i
        self.vel.x = self.speed * np.cos(np.deg2rad(self.theta))
        self.vel.y = self.speed * -np.sin(np.deg2rad(self.theta))

    def tick(self):
        self.pos = self.pos + self.vel * dt

        # Check boundary
        if self.pos.x < 0:
            self.pos.x = WIDTH
        if self.pos.x > WIDTH:
            self.pos.x = 0
        if self.pos.y < 0:
            self.pos.y = HEIGHT
        if self.pos.y > HEIGHT:
            self.pos.y = 0

        # Rebound boundary. Funny?
        # if self.pos.x < 0:
        #     self.theta = 180 - self.theta
        # if self.pos.x > WIDTH:
        #     self.theta = 180 - self.theta
        # if self.pos.y < 0:
        #     self.theta *= -1
        # if self.pos.y > HEIGHT:
        #     self.theta *= -1

# Generate random starting positions for sparrows
sparrows = []
for i in range(NUM_BIRDS):
    x = np.random.randint(10, 790)
    y = np.random.randint(10, 590)
    theta = np.random.rand() * 360
    sparrows.append(Sparrow(x, y, theta))

sparrows_posx = np.zeros(NUM_BIRDS)
sparrows_posy = np.zeros(NUM_BIRDS)
sparrows_theta = np.zeros(NUM_BIRDS)

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("grey")

    for i, sparrow in enumerate(sparrows):
        sparrows_posx[i] = sparrow.pos.x
        sparrows_posy[i] = sparrow.pos.y
        sparrows_theta[i] = sparrow.theta

    for i, sparrow in enumerate(sparrows):
        rotated_texture = pygame.transform.rotate(texture, sparrow.theta)
        screen.blit(rotated_texture, sparrow.pos)
        sparrow.tick()

        theta_sin = 0
        theta_cos = 0
        theta_j = 0
        N_i = 0

        for j in range(len(sparrows_posx)):
            if sparrow.pos.distance_to((sparrows_posx[j], sparrows_posy[j])) < r_nbr:
                theta_sin += np.sin(np.deg2rad(sparrows_theta[j]))
                theta_cos += np.cos(np.deg2rad(sparrows_theta[j]))
        
        if theta_sin != 0:
            d_theta = np.random.rand() * ETA - ETA / 2
            theta_i = np.arctan(theta_sin / theta_cos) + d_theta
            sparrow.update_theta(np.rad2deg(theta_i) * dt)

    pygame.display.flip()
    clock.tick(FPS)
    
pygame.quit()