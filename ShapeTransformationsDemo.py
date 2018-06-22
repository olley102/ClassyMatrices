import linalgebra as la
import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots()

shape = la.Shape2D.unitsquare()
shape.plot(ax1)
ax1.axis("scaled")
fig1.show()

t_mat = la.Transformation2D([
    [1, 2],
    [0, 1]
])

shape.transform(t_mat)
shape.plot(ax1)
ax1.axis("scaled")

fig2, ax2 = plt.subplots()

new_sq = la.Shape2D.unitsquare()
new_sq.plot(ax2)
ax2.axis("scaled")

new_sq.translate(5, 5)
new_sq.reflect(30)
print(new_sq)
new_sq.plot(ax2)
ax2.axis("scaled")

fig2.show()
plt.show()
