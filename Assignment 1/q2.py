import numpy as np
import pygame


NUM_BIRDS = 100     # Number stated in assignment
WIDTH = 800
HEIGHT = 600
FPS = 60
dt = 1 / FPS
r_sep = 20
r_align = 50
r_coh = 200
zeta = 360          # factor for random noise
fov = 270           # unused, unoptimized and slow


pygame.init()
screen = pygame.display.set_mode([WIDTH, HEIGHT])
clock = pygame.time.Clock()
pygame.display.set_caption("Sparrow Flocking Simulation")
texture = pygame.image.load('sparrow.png').convert_alpha()


def get_heading(vector):
    c = np.sqrt(vector.x ** 2 + vector.y ** 2)
    
    if vector.y < 0 and vector.x > 0: # quadrant 4
        return np.rad2deg(np.arcsin(-vector.y / c))
    if vector.y < 0 and vector.x < 0: # quadrant 3
        return 180 - np.rad2deg(np.arcsin(-vector.y / c))
    if vector.y > 0 and vector.x < 0: # quadrant 2
        return 180 + np.rad2deg(np.arcsin(vector.y / c))
    if vector.y > 0 and vector.x > 0: # quadrant 1
        return 360 - np.rad2deg(np.arcsin(vector.y / c))
    
    return 0


class Sparrow:
    def __init__(self, x, y, theta):
        self.pos = pygame.Vector2(x, y)
        self.theta = theta
        self.theta_0 = theta
        self.speed = 200
        self.vel = pygame.Vector2(self.speed * np.cos(np.deg2rad(self.theta)),
                                  self.speed * -np.sin(np.deg2rad(self.theta)))
        
    def separate(self, separation):
        phi = get_heading(separation)
        noise = (np.random.rand()) * zeta
        self.update_theta(phi, self.theta_0, noise)

    def align(self, alignment):
        noise = (np.random.rand()) * zeta
        self.update_theta(alignment, self.theta_0, noise)

    def cohesion(self, cohesion):
        phi = get_heading(cohesion)
        noise = (np.random.rand()) * zeta
        self.update_theta(phi, self.theta_0, noise)

    def update_theta(self, phi, theta_0, noise=0):
        phi_1 = phi + 180

        if phi_1 < 360:
            if theta_0 > phi_1:
                self.theta += (phi - (180-(360-theta_0)) + noise) * dt
            else:
                self.theta += (phi - theta_0 + noise) * dt

        if phi_1 > 360:
            phi_1 -= 360
            if theta_0 < phi_1 :
                self.theta += (phi - (360+theta_0) + noise) * dt
            else:
                self.theta += (phi - theta_0 + noise) * dt

    def tick(self):
        self.pos += self.vel * dt
        
        # Check boundary
        if self.pos.x < 0:
            self.pos.x = WIDTH
        if self.pos.x > WIDTH:
            self.pos.x = 0
        if self.pos.y < 0:
            self.pos.y = HEIGHT
        if self.pos.y > HEIGHT:
            self.pos.y = 0

        self.vel.x = self.speed * np.cos(np.deg2rad(self.theta))
        self.vel.y = self.speed * -np.sin(np.deg2rad(self.theta))

        if self.theta > 360:
            self.theta_0 = self.theta - 360
        elif self.theta < 0:
            self.theta_0 = 360 + self.theta
        else:
            self.theta_0 = self.theta


# Generate random starting positions and headings for sparrows
sparrows = []
for i in range(NUM_BIRDS):
    x = np.random.randint(10, WIDTH - 10)
    y = np.random.randint(10, HEIGHT - 10)
    theta = np.random.rand() * 360
    sparrows.append(Sparrow(x, y, theta))

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("grey")

    for sparrow in sparrows:
        rotated_texture = pygame.transform.rotate(texture, sparrow.theta)
        screen.blit(rotated_texture, sparrow.pos)

        count_sep = 0
        count_align = 0
        count_coh = 0
        separation = pygame.Vector2(0, 0)
        alignment = 0
        cohesion = pygame.Vector2(0, 0)

        for other in sparrows:
            r = sparrow.pos.distance_to(other.pos)
            # phi = get_heading(other.pos)

            # beta_1 = sparrow.theta + fov / 2
            # beta_2 = sparrow.theta - fov / 2
            # in_view = False

            # if beta_1 > 360:
            #     beta_1 -= 360
            # if beta_2 < 0:
            #     beta_2 += 360
            
            # if phi >= beta_1 and phi <= beta_2:
            #     in_view = True

            if r < r_sep and sparrow is not other:
                count_sep += 1
                separation += (sparrow.pos - other.pos) * ((r_sep - r) / r_sep)
            if r < r_align:
                count_align += 1
                alignment += other.theta * ((r_align - r) / r_align)
            if r < r_coh and sparrow is not other:
                count_coh += 1
                cohesion += other.pos

        if count_sep != 0:
            sparrow.separate(separation)
            # pass
        if count_align != 0:
            alignment /= count_align
            sparrow.align(alignment)
        if count_coh != 0:
            cohesion /= count_coh
            sparrow.cohesion(cohesion)

        sparrow.tick()

        # break

    pygame.display.flip()
    clock.tick(FPS)

    # break
    

pygame.quit()