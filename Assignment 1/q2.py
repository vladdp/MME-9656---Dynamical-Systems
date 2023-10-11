"""
MME 9656 - Dynamical Systems
Assignment #1
Due: October 11, 2023

Author: Vlad Pac

Behaviour ideas taken from Craig Reynolds' website,
https://www.red3d.com/cwr/boids/
"""

import numpy as np
import pygame


NUM_BIRDS = 100     # Number stated in assignment
WIDTH = 800         # Width of the window
HEIGHT = 600        # Height of the window
FPS = 60            # framerate of the simulation
dt = 1 / FPS        # time per frame
r_sep = 20          # radius of interest for separation
r_align = 50        # radius of interest for alignment
r_coh = 200         # radius of interest for cohesion
zeta = 360          # factor for random noise
fov = 270           # unused, unoptimized and slow (field of view)


# Pygame initialization and boilerplate code
pygame.init()
screen = pygame.display.set_mode([WIDTH, HEIGHT])
clock = pygame.time.Clock()
pygame.display.set_caption("Sparrow Flocking Simulation")
# the texture holds the sprite for each sparrow
texture = pygame.image.load('sparrow.png').convert_alpha()


def get_heading(vector):
    """ Returns a heading between 0 and 360 from the argument. 
        A heading of 0 corresponds to the positive x-axis. 
        
        Input: pygame.Vector2
        Output: heading """
    c = np.sqrt(vector.x ** 2 + vector.y ** 2)
    
    # Pygame y-axis is positive downwards, so the calculated 
    # angle must be mirrored on the x-axis to get the heading
    if vector.y < 0 and vector.x > 0: # quadrant 4
        return np.rad2deg(np.arcsin(-vector.y / c))
    if vector.y < 0 and vector.x < 0: # quadrant 3
        return 180 - np.rad2deg(np.arcsin(-vector.y / c))
    if vector.y > 0 and vector.x < 0: # quadrant 2
        return 180 + np.rad2deg(np.arcsin(vector.y / c))
    if vector.y > 0 and vector.x > 0: # quadrant 1
        return 360 - np.rad2deg(np.arcsin(vector.y / c))
    
    return 0 # if no condition is met


class Sparrow:
    """ A Sparrow object holds a position vector, a velocity vector,
        and a heading (theta). It has 3 behaviour functions: separate,
        align, and cohesion; an update_theta function that updates the
        heading each frame; and a tick function to update the position
        and velocity of the bird each frame as well as border detection
    """

    def __init__(self, x, y, theta):
        """ Initializes a Sparrow object given x, y coordinates
            and a heading. pos holds the x, y position and vel
            holds the x, y velocity. theta holds the heading,
            theta_0 holds the previous heading
        """
        self.pos = pygame.Vector2(x, y)
        self.theta = theta
        self.theta_0 = theta
        self.speed = 200
        self.vel = pygame.Vector2(self.speed * np.cos(np.deg2rad(self.theta)),
                                  self.speed * -np.sin(np.deg2rad(self.theta)))
        
    def separate(self, separation):
        """ Gets the heading towards the separation vector into phi,
            then calls update_theta with phi and noise
        """
        phi = get_heading(separation)
        noise = (np.random.rand()) * zeta
        self.update_theta(phi, self.theta_0, noise)

    def align(self, alignment):
        """ Calls update_theta with the average heading (alignment)
            and adds noise
        """
        noise = (np.random.rand()) * zeta
        self.update_theta(alignment, self.theta_0, noise)

    def cohesion(self, cohesion):
        """ Gets the heading towards the cohesion vector into phi,
            then calls update_theta with phi and noise
        """
        phi = get_heading(cohesion)
        noise = (np.random.rand()) * zeta
        self.update_theta(phi, self.theta_0, noise)

    def update_theta(self, phi, theta_0, noise=0):
        """ Increments the current heading of the sparrow given a new
            heading phi. Tough to explain the logic in words. It has to
            do with the abrupt heading values on the positive x-axis
            where the heading changes from 360 to 0 and vice-versa.
        """
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
        """ Updates the position given position and dt.
            Checks the boundary so the bird continues on the
            opposite side. Updates velocity given new heading.
        """
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

        # Update velocity vector given new heading
        self.vel.x = self.speed * np.cos(np.deg2rad(self.theta))
        self.vel.y = self.speed * -np.sin(np.deg2rad(self.theta))

        # Corrects heading if it exceeds 360 or falls negative
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

# The game loop runs until the window is closed
frame_count = 0
sim_time = 60               # simulation time in seconds
running = True
while running:

    # standard code to exit the loop when window is closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("grey")

    # Iterate through all the sparrows in the sparrows list
    # Each sparrow is stored in the sparrow variable
    for sparrow in sparrows:
        rotated_texture = pygame.transform.rotate(texture, sparrow.theta)
        screen.blit(rotated_texture, sparrow.pos)

        # reset all the counters for how many birds were used in
        # separation, alignment, or cohesion function calls
        count_sep = 0
        count_align = 0
        count_coh = 0
        # reset the separation, alignment, and cohesion values
        separation = pygame.Vector2(0, 0)
        alignment = 0
        cohesion = pygame.Vector2(0, 0)

        # Check every other sparrow besides the current one to 
        # calculate separation, alignment, and cohesion 
        for other in sparrows:
            # r holds the distance from the sparrow to another sparrow
            r = sparrow.pos.distance_to(other.pos)

            # The commented code below implemented field-of-view
            # but it was too slow
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

            # Check to see if the other sparrow is within
            # the neighborhood radius r_sep
            if r < r_sep and sparrow is not other:
                count_sep += 1
                # separation vector calculated by summing all vectors from
                # each other bird to the current sparrow and multiplied by
                # a weight dependent on the distance
                separation += (sparrow.pos - other.pos) * ((r_sep - r) / r_sep)
            if r < r_align:
                count_align += 1
                # alignment holds the sum of all headings times their weight
                alignment += other.theta * ((r_align - r) / r_align)
            if r < r_coh and sparrow is not other:
                count_coh += 1
                # cohesion sums all bird positions to be later averaged
                cohesion += other.pos

        # The following if statements check if the sparrow has to 
        # update its heading based on either separation, alignment
        if count_sep != 0:
            sparrow.separate(separation)
            # pass
        if count_align != 0:
            # average the heading
            alignment /= count_align
            sparrow.align(alignment)
        if count_coh != 0:
            # average the position of local flockmates
            cohesion /= count_coh
            sparrow.cohesion(cohesion)

        sparrow.tick()  # update the sparrow

        # break

    pygame.display.flip()
    clock.tick(FPS)

    # code below from stackoverflow to save each frame
    # to develop into a video
    # frame_count += 1
    # filename = 'frames/frame_%04d.png' % frame_count
    # pygame.image.save(screen, filename)

    # if frame_count >= sim_time * FPS:
    #     break
    

pygame.quit()