from cmu_graphics import *

import math

print("dada is tha best")

app.setMaxShapeCount(99999999999999999999999999999)

app.fov = 120
app.fov = math.radians(app.fov)
app.c = 2 * math.tan(app.fov / 2)

app.objects = []
app.triangles = []

app.width = 600
app.height = 600

def renderPoint(x, y, z):
    screenX = (app.width / 2.0) * (1+((1/app.c) * (x/z)))
    screenY = (app.height / 2.0) * (1-((app.width/(app.c*app.height)) * (y/z)))
    return (screenX, screenY)

def drawTriangle(point1, point2, point3, col):
    p1 = renderPoint(point1[0], point1[1], point1[2])
    p2 = renderPoint(point2[0], point2[1], point2[2])
    p3 = renderPoint(point3[0], point3[1], point3[2])
    if p1 != None and p2 != None and p3 != None:
        return Polygon(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], fill=col)

def averageZDepth(triangle):
    return (triangle[0][2] + triangle[1][2] + triangle[2][2]) / 3

def multiplyQuaternions(q: tuple, p: tuple):
    ### RULES OF QUATERNION MULTIPLICATION ###
    # for quaternion multiplication a*b, left column is a and top row is b
    # Note: a*b != b*a
    #     1  i  j  k
    #   +-----------
    # 1 | 1  i  j  k
    # i | i -1  k -j
    # j | j -k -1  i
    # k | k  j -i -1

    # Gets the 4 values of the 2 quaternions
    # Note: pi is the i part for quaternion p, not 3.14159
    qr, qi, qj, qk = q
    pr, pi, pj, pk = p

    # Multiplies the quaternions for each part
    r = (qr * pr) - (qi * pi) - (qj * pj) - (qk * pk)
    i = (qr * pi) + (qi * pr) + (qj * pk) - (qk * pj)
    j = (qr * pj) - (qi * pk) + (qj * pr) + (qk * pi)
    k = (qr * pk) + (qi * pj) - (qj * pi) + (qk * pr)

    return (r, i, j, k)

def rotatePoint(point, angleDegree, i, j, k):
    angle = math.radians(angleDegree)
    vector = (i, j, k)

    vectorMag = math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
    unitVector = (vector[0] / vectorMag, vector[1] / vectorMag, vector[2] / vectorMag)

    # the quaternion for the angle in terms of real, i, j, k
    rotationQuaternion = (math.cos(angle / 2), unitVector[0] * math.sin(angle / 2), unitVector[1] * math.sin(angle / 2), unitVector[2] * math.sin(angle / 2))

    # implements quaternion multiplication
    rotation = multiplyQuaternions(rotationQuaternion, (0, point[0], point[1], point[2]))

    return (rotation[1], rotation[2], rotation[3])
    

class Shape():
    def __init__(self, triangles, colors):
        self.image = Group()
        self.triangles = triangles
        self.colors = colors
        self.draw()

    def rotate(self, quaternion):
        centroid = [0, 0, 0]
        for triangle in self.triangles:
            for point in triangle:
                centroid[0] += point[0]
                centroid[1] += point[1]
                centroid[2] += point[2]
        centroid[0] /= len(self.triangles * 3)
        centroid[1] /= len(self.triangles * 3)
        centroid[2] /= len(self.triangles * 3)
        rotatedTriangles = self.triangles
        for triangle in rotatedTriangles:
            for point in triangle:
                point[0] -= centroid[0]
                point[1] -= centroid[1]
                point[2] -= centroid[2]
        for idx in range(len(self.triangles)):
            triangle = self.triangles[idx]
            for index in range(len(triangle)):
                point = triangle[index]
                centerRotated = rotatePoint(point, quaternion[0], quaternion[1], quaternion[2], quaternion[3])
                triangle[index] = [centerRotated[0] + centroid[0], centerRotated[1] + centroid[1], centerRotated[2] + centroid[1]]
        self.draw()


    def draw(self):
        self.image.clear()
        for idx in range(len(self.triangles)):
            triangle = self.triangles[idx]
            color = self.colors[idx // 2]
            app.triangles.append([triangle, color, self, averageZDepth(triangle)])

mrCube = Shape([
    [[-25, 25, 50], [25, 25, 50], [-25, -25, 50]], [[25, 25, 50], [-25, -25, 50], [25, -25, 50]],
    [[-25, 25, 100], [25, 25, 100], [-25, -25, 100]], [[25, 25, 100], [-25, -25, 100], [25, -25, 100]],
    [[25, 25, 50], [25, 25, 100], [25, -25, 100]], [[25, 25, 50], [25, -25, 50], [25, -25, 100]],
    [[25, 25, 50], [25, 25, 100], [25, -25, 100]], [[25, 25, 50], [25, -25, 50], [25, -25, 100]],
    [[-25, 25, 50], [-25, 25, 100], [25, 25, 50]], [[25, 25, 50], [25, 25, 100], [-25, 25, 100]],
    [[-25, -25, 50], [-25, -25, 100], [25, -25, 50]], [[25, -25, 50], [25, -25, 100], [-25, -25, 100]]
], ['blue', 'green', 'red', 'orange', 'yellow', 'cyan'])

app.objects.append(mrCube)

def handleDrawing():
    for shape in app.objects:
        shape.draw()
    sortedTriangles = sorted(app.triangles, key=lambda x: x[3])
    for triangle in sortedTriangles:
        triangle[2].image.add(drawTriangle(triangle[0][0], triangle[0][1], triangle[0][2], triangle[1]))

app.stepsPerSecond = 30
def onStep():
    handleDrawing()
    mrCube.rotate((1, 1, 0, 0))

cmu_graphics.run()