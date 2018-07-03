import classymatrices as cm
import matplotlib.pyplot as plt
import math

# single point (circle)
shape = cm.Shape2D([
  cm.Point2D(0, 0, 5)
])

# # triangle
# shape = cm.Shape2D([
#   cm.Point2D(0, 0),
#   cm.Point2D(1, 0),
#   cm.Point2D(0.5, math.sqrt(0.75))
# ])


def ontype(event):
  if event.key == "n":
    global shape
    shape = cm.Shape2D.unitsquare()
  elif event.key == "a":
    shape.rotate(1)
  elif event.key == "d":
    shape.rotate(-1)
  elif event.key == "left":
    shape.translate(-0.1, 0)
  elif event.key == "right":
    shape.translate(0.1, 0)
  elif event.key == "up":
    shape.translate(0, 0.1)
  elif event.key == "down":
    shape.translate(0, -0.1)
  elif event.key == "e":
    shape.enlarge(1.1, 1.1)
  elif event.key == "r":
    shape.enlarge(0.9, 0.9)
  elif event.key == "t":
    shape.reflect(45)

  # plt.gca().clear()
  shape.plot(ax)
  # could change ylim and xlim here
  plt.draw()


fig, ax = plt.subplots()
shape.plot(ax)
ax.set_aspect("equal")
ax.set_ylim(-3, 3)
ax.set_xlim(-3, 3)
fig.canvas.mpl_connect("key_press_event", ontype)
plt.show()
plt.draw()

# idea: animation?
