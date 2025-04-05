import pygame, sys

pygame.init()
screen = pygame.display.set_mode((128, 720))
clock = pygame.time.Clock()
running = True

def main():
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        screen.fill('purple')

        pygame.display.flip()
        clock.tick(60)
