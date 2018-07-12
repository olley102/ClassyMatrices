# classymatrices

#### Important note for using
Check that you have the following modules available:
- `math`
- `random`
- `fractions` (optional: only used in Mat.fracinverse())
- `copy`
- `matplotlib.pyplot`

## Matrix operations and plotting

Matrix operations and plotting are featured in the classymatrices.py file. The module can be imported, for example:

```
import classymatrices as cm
id = cm.Mat.identity(2)
```

### Classes

- **`Mat`**: Matrix class. Includes methods such as `identity`, `matmul`, `inverse` and `det`.
- **`Transformation2D`**: Transformation matrix class for 2D plot. Includes `applyunitsq` method.
- **`Point2D`**: Point class for 2D plot. Includes `x`, `y`, `alter` and `plot` methods.
- **`Shape2D`**: Shape class for 2D plot. Includes methods such as `unitsquare`, `transform` and `plot`.

### Example use

Example use can be found in the following files.

- LinearRegExample.ipynb
- ShapeCanvasDemo.py
- ShapeTransformationsDemo.py
