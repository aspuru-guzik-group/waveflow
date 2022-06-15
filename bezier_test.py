impot bezier
import numpy as np
nodes = np.asfortranarray([
    [0.0, 0.625, 1.0]
])
curve = curve.Curve(nodes, degree=2)

print(curve.evaluate(0.75))

