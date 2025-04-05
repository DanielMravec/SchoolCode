from cmu_graphics import *

def renderPoint(x, y, z, screen_width=800, screen_height=600, fov=90, near_plane=0.1, far_plane=1000):
    # Convert the field of view from degrees to radians
    fov_rad = fov * (3.141592653589793 / 180.0)
    
    # Calculate the projection matrix parameters
    aspect_ratio = screen_width / screen_height
    f = 1.0 / (aspect_ratio * (2 * 3.141592653589793 / fov_rad))
    
    # Simple perspective projection
    projected_x = x / z
    projected_y = y / z

    # Apply the perspective division
    screenX = (projected_x * f + 1) * screen_width / 2
    screenY = (1 - projected_y * f) * screen_height / 2

    Circle(screenX, screenY, 2)

# Example usage
screen_width = 800
screen_height = 600

renderPoint(10, 10, 50)
renderPoint(100, 10, 50)
renderPoint(10, 100, 50)
renderPoint(100, 100, 50)
renderPoint(10, 10, 100)
renderPoint(100, 10, 100)
renderPoint(10, 100, 100)
renderPoint(100, 100, 100)

cmu_graphics.run()
