import math
import random
from fractions import Fraction
import copy
import matplotlib.pyplot as plt


"""
For plotting methods, pyplot is required locally
with fig and ax set up before use.

Example:
  # required imports
  import classymatrices as cm
  import matplotlib.pyplot as plt
  
  # figure setup
  fig, ax = plt.subplots()
  
  # cm and display
  shape = cm.Shape2D.unitsquare()
  shape.plot(ax)
  ax.axis("scaled")
"""


class Mat(object):
  """Matrix

  :var m (int): vertical size
  :var n (int): horizontal size
  :var values (2D list): values in matrix
  :var is_float (bool): matrix contains floats
  """

  m = 1
  n = 1
  values = []
  is_float = False

  def __init__(self, vals:list):
    """Initializes matrix.

    :param vals (2D list): values in matrix.
  
    :raises TypeError: if vals contains values of type neither int nor float.
    """

    vals_float = False  # False for int, True for float

    m_temp = len(vals)
    n_temp = len(vals[0])

    for row in vals:
      if len(row) != n_temp:
        raise TypeError
      for num in row:
        if type(num) == float:
          vals_float = True
        elif type(num) != int:
          raise TypeError(
            "vals contains values of type neither int nor float"
          )
    
    self.values = vals[:]
    self.m = m_temp
    self.n = n_temp

    if vals_float:
      self.makefloat()

  def __repr__(self):
    return self.__str__()
  
  def __str__(self):
    text = ""
    for row in self.values:
      text += "{}\n".format(row)
    return text

  def zeros(vert_num, hor_num):
    """Initializes Mat of zeros.

    :param vert_num: vertical size.
    :param hor_num: horizontal size.
    :return: The matrix of zeros.
    """
    
    return Mat([[0]*hor_num for i in range(vert_num)])
 
  def ones(vert_num, hor_num):
    """Initializes Mat of ones.

    :param vert_num: vertical size.
    :param hor_num: horizontal size.
    :return: The matrix of ones.
    """
    
    return Mat([[1]*hor_num for i in range(vert_num)])

  def identity(num):
    """Initializes identity Mat.
    Ones lie through center diagonal from top-left to bottom-right.
    Rest are zeros.

    :param num: size in either direction of square matrix.
    :return: The identity matrix.
    """
    
    temp_mat = Mat.zeros(num, num)
    for row_num in range(num):
      temp_mat.values[row_num][row_num] = 1
    return temp_mat

  def randuniform(vert_num, hor_num, start=0, stop=1):
    """Initializes Mat of uniformly random values.
    Range is inclusive.

    :param vert_num: vertical size.
    :param hor_num: horizontal size.
    :param start: minimum of range. Default: 0
    :param stop: maximum of range. Default: 1
    :return: The matrix of randoms.
    """
    
    new_values = [[random.uniform(start, stop) for i in range(hor_num)]
      for j in range(vert_num)
    ]
    return Mat(new_values)

  def randint(vert_num, hor_num, start, stop):
    """Initializes Mat of random ints.
    Range is inclusive.

    :param vert_num: vertical size.
    :param hor_num: horizontal size.
    :param start: minimum of range.
    :param stop: maximum of range.
    :return: The matrix of randoms.
    """
    
    new_values = [[random.randint(start, stop) for i in range(hor_num)]
      for j in range(vert_num)
    ]
    return Mat(new_values)

  def _check_float(obj):
    """Checks if matrix contains float.
    Modifies obj.is_float.
    
    :param obj: matrix of type Mat or a child of Mat.
    :return: True: if contains float.
    :return: False: otherwise.
    :raises TypeError: if vals contains values of type neither int nor float.
    """

    for row in obj.values:

      for val in row:
        if type(val) == float:
          obj.is_float = True
          return True
        elif type(val) != int:
          raise TypeError(
            "vals contains values of type neither int nor float"
          )
    
    obj.is_float = False
    return False

  def makefloat(self):  # makes elements all float
    """Makes matrix values type float.
    Modifies self in-place.
    """
    
    self.is_float = True
    for row_num in range(self.m):
        self.values[row_num] = list(map(
          float, self.values[row_num]
        ))
  
  def _add(val1, val2):  # used for map
    return val1 + val2
  
  def _mul(val1, val2):
    return val1 * val2

  def __add__(self, other):
    """Adds values in other to self.
    Other can be int or float for simplicity.

    :param other: Mat, child of Mat, int or float.
    :return: Mat or child of Mat.
    :raises TypeError: if self and other are same type but different size.
    :raises TypeError: if other is of invalid type.
    """
    
    temp_mat = self.copy()

    if type(other) == type(temp_mat):

      if temp_mat.m == other.m and temp_mat.n == other.n:
        for row_num in range(len(temp_mat.values)):
          temp_mat.values[row_num] = list(map(
            Mat._add,
            temp_mat.values[row_num], other.values[row_num]
          ))
      else:
        raise TypeError(
          "self and other are same type but different size"
        )

    elif type(other) in (int, float):

      for row_num in range(len(temp_mat.values)):
        temp_mat.values[row_num] = list(map(
          lambda x: x + other,
          temp_mat.values[row_num]
        ))

    else:
      raise TypeError("other is of invalid type")
    
    if Mat._check_float(temp_mat):
      temp_mat.makefloat()
    
    return temp_mat

  def __sub__(self, other):
    return self + other * -1

  def power(self, other):
    """Raises to the power of other, element-wise.

    :param other: power to raise to.
    :return: Mat or child or Mat.
    :raises TypeError: if other is of type neither int nor float.
    """
    
    if type(other) not in (int, float):
      raise TypeError(
        "other is of type neither int nor float."
      )
    temp_mat = self.copy()

    for row_num in range(len(temp_mat.values)):
      temp_mat.values[row_num] = list(map(
        lambda x: x ** other,
        temp_mat.values[row_num]
      ))

    return temp_mat

  def _check_type(obj):
    classes = (
      "Mat",
      "Transformation2D",
      "Point2D"
    )
    return obj.__class__.__name__ in classes

  def _make_type(vals, new_type):
    if new_type == type(Point2D(0, 0)):
      return Point2D(vals[0][0], vals[1][0])
    
    return new_type(vals)

  def _check_mul_compat(obj, other):
    return obj.n == other.m and \
      Mat._check_type(obj) and Mat._check_type(other)

  def matmul(self, other):
    """Matrix multiplication.

    :param other: Mat or child of Mat to multiply with.
    :return: Mat or child of Mat.
    :raises ValueError: if self and other are not compatible for matmul.
    """
    
    if other.__class__.__name__ == "Point2D" or \
      (
        self.__class__.__name__ == "Transformation2D" and
        not (other.m == other.n == 2)
      ):
      saved_type = type(other)
    else:
      saved_type = type(self)  # to return correct type
    
    new_values = [[0]*other.n for i in range(self.m)]
    other_tr = other.transpose()

    if Mat._check_mul_compat(self, other):

      for row_num in range(self.m):
        for col_num in range(other.n):
          new_values[row_num][col_num] = sum(map(
            Mat._mul,
            self.values[row_num], other_tr.values[col_num]
          ))

    else:
      raise ValueError(
        "self and other are not compatible for matmul"
      )

    return Mat._make_type(new_values, saved_type)

  def __mul__(self, other):
    """Multiplies self with other.

    :param other: Mat, child of Mat, int or float to multiply with.
    :return: Mat or child of Mat.
    """
    temp_mat = None

    if type(other) == int or type(other) == float:
      temp_mat = self.simplemul(other)

      if Mat._check_float(temp_mat):
        temp_mat.makefloat()

    else:
      temp_mat = self.matmul(other)

    return temp_mat

  def simplemul(self, other):
    """Multiply by scalar.

    :param other: int, float or similar to multiply with.
    :return: Mat or child of Mat.
    """
    
    temp_mat = self.copy()
    for row_num in range(len(temp_mat.values)):
      temp_mat.values[row_num] = list(map(
        lambda x: x * other,
        temp_mat.values[row_num]
      ))
    return temp_mat

  def hadprod(self, other):
    """Hadamard product i.e. element-wise mul

    :param other: Mat or child of Mat to multiply with.
    :return: Mat or child of Mat.
    :raises ValueError: if self and other have different size.
    """
    
    if type(self) != type(other) or \
      self.m != other.m or self.n != other.n:
      raise ValueError(
        "self and other have different size."
      )
    temp_mat = self.copy()
    
    for row_num in range(temp_mat.m):
      temp_mat.values[row_num] = list(map(
        Mat._mul,
        temp_mat.values[row_num], other.values[row_num]
      ))
    return temp_mat

  def dotprod(self, other):
    """Dot product.
    p = x'y where x=self, y=other

    :param other: Mat or child of Mat.
    :return: int or float.
    :raises TypeError: if types are different.
    :raises TypeError: if sizes are different.
    :raises TypeError: if self or other are not vectors.
    """
    
    if type(self) != type(other) or \
      self.m != other.m or self.n != other.n or \
      (self.m != 1 and self.n != 1):  # check for vectors
      raise TypeError("invalid types or sizes")
    
    temp_mat = self.copy()
    other_temp = other.copy()

    if self.m == 1:
      other_temp = other_temp.transpose()
    else:
      temp_mat = temp_mat.transpose()

    temp_mat = temp_mat.matmul(other_temp)
    
    return temp_mat.values[0][0]

  def cofactors(self):
    """Creates Mat of cofactors."""

    if self.m < 2:
      return None

    if self.m == 2:
      temp_mat = self.copy()
      temp_mat.values[0][0], temp_mat.values[1][1] = \
        temp_mat.get(1, 1), temp_mat.get(0, 0)
      temp_mat.values[0][1], temp_mat.values[1][0] = \
        -temp_mat.get(1, 0), -temp_mat.get(0, 1)
      return temp_mat

    temp_mat = Mat.zeros(self.m, self.m)
    for row_num in range(self.m):
      for col_num in range(self.m):
        square = self.deepcopy()
        square.delcol(col_num)
        square.delrow(row_num)
        temp_mat.values[row_num][col_num] = \
          square.det() * (-1)**(row_num + col_num)
    return temp_mat

  def inverse(self):
    """Creates inverse Mat."""
    the_det = self.det()  # includes size validation
    if self.det() == 0:
      return
    
    the_adj = self.cofactors().transpose()

    return the_adj * (1/the_det)

  def fracinverse(self):  # returns Mat of Fraction
    """Creates inverse Mat of type Fraction."""
    the_det = self.det()
    if self.det() == 0:
      return
    
    the_adj = self.cofactors().transpose()

    return the_adj.simplemul(Fraction(1, the_det))

  def det(self):
    """Calculates determinant."""

    if self.m != self.n:
      raise TypeError
    if self.m == 1:
      return self.values[0][0]
    if self.m == 2:
      return self.values[0][0] * self.values[1][1] \
        - self.values[0][1] * self.values[1][0]
    
    dets = []
    for col_num in range(self.m):
      new_square = self.deepcopy()
      new_square.delrow(0)
      new_square.delcol(col_num)
      dets.append(new_square.det())
    
    return sum((-1)**i * dets[i] * self.values[0][i]
      for i in range(self.m)
    )

  def copy(self):
    saved_type = type(self)
    new_values = self.values[:]
    return Mat._make_type(new_values, saved_type)

  def deepcopy(self):
    saved_type = type(self)
    new_values = copy.deepcopy(self.values)
    return Mat._make_type(new_values, saved_type)

  def get(self, key1, key2):
    """Gets a value in matrix.

    :param key1: vertical offset from top.
    :param key2: horizontal offset from left.
    :return: The value.
    """

    if key2 == "all":
      return self.values[key1]
    return self.values[key1][key2]

  def alter(self, key1, key2, val):
    """Changes a value in matrix.

    :param key1: vertical offset from top.
    :param key2: horizontal offset from left.
    :param val: value to set.
    """

    self.values[key1][key2] = val
    if Mat._check_float(self):
      self.makefloat()

  def delcol(self, col_num):
    """Deletes column col_num."""
    for row_num in range(self.m):
      del self.values[row_num][col_num]
    self.n -= 1
    return
  
  def delrow(self, row_num):
    """Deletes row row_num"""
    del self.values[row_num]
    self.m -= 1
    return

  def insert(self, vals:list, pos:int = 0, orientation:str = "r"):
    """Insert row or column in matrix.

    :param vals: values in row or column.
    :param pos: index of matrix at which to insert new values.
    :param orientation: 'r' for row or 'c' for column.
    """

    if orientation == "r":
      if len(vals) != self.n or pos < 0 or pos > self.m:
        raise TypeError
      self.values.insert(pos, vals)
      self.m += 1
    
    elif orientation == "c":
      if len(vals) != self.m or pos < 0 or pos > self.n:
        raise TypeError
      for row_num in range(self.n):
        self.values[row_num].insert(pos, vals[row_num])
      self.n += 1
    
    else:
      raise TypeError

    if Mat._check_float(self):
      self.makefloat()
  
  def transpose(self):
    """Create transpose matrix.
    Flips matrix about center diagonal.
    """

    saved_type = type(self)
    if self.__class__.__name__ == "Point2D":
      saved_type = Mat
    new_values = list(map(list, zip(*self.values)))
    return saved_type(new_values)


class Transformation2D(Mat):
  """Transformation matrix class for 2D plot."""

  def __init__(self, vals:list):
    """Initializes transformation matrix.

    :param vals: list of lists of values.
    :raises ValueError: if matrix is not square.
    """

    super().__init__(vals)

    if not (self.m == self.n == 2):
      raise ValueError(
        "matrix is not square"
      )

  def applyunitsq(self):
    shape = Shape2D.unitsquare()
    shape.transform(self)
    return shape


class Point2D(Mat):
  """Point class for 2D plot.

  :var plotsize: size of point when plotted.
  """

  plotsize = 5

  def __init__(self, x, y, plotsize=5):
    """Initializes point.

    :param x: value in horizontal axis.
    :param y: value in vertical axis.
    :param plotsize: size of point when plotted.
    """
    super().__init__([[x], [y]])
    self.plotsize = plotsize

  def x(self):
    return self.get(0, 0)

  def y(self):
    return self.get(1, 0)

  def alter(self, x, y):
    super().alter(0, 0, x)
    super().alter(1, 0, y)

  def plot(self, ax):
    """Plot point using Matplotlib.pyplot.

    :param ax: axis initialized using pyplot.
    """
    ax.plot(self.x(), self.y(), "o", ms=self.plotsize)


class Shape2D(object):
  """Shape class for 2D plot.

  :var points: list of Point2D vertices.
  :var anchor: index of point from which shape is transformed.
  """

  points = []
  anchor = 0

  def __init__(self, points:list):
    """Initializes shape.

    :param points: list of Point2D vertices.
    """

    self.points = points

  def __repr__(self):
    ret_str = "Shape2D points:\n"

    for p in self.points:
      ret_str += p.transpose().__repr__()
    
    return ret_str

  def unitsquare():
    """Initializes unit square.
    i.e. square at origin with length 1.

    :return: The unit square.
    """

    return Shape2D([
      Point2D(0, 0),
      Point2D(1, 0),
      Point2D(1, 1),
      Point2D(0, 1)
    ])

  def transform(self, t_mat):
    """Applies transformation to self.

    :param t_mat: transformation matrix.
    """

    for p_num, p in enumerate(self.points):
      self.points[p_num] = t_mat * p

  def translate(self, x, y):
    """Applies translation to self.

    :param x: distance horizontally.
    :param y: distance vertically.
    """

    if len(self.points) == 1:
      saved_x = self.points[0].x()
      saved_y = self.points[0].y()
      self.points[0].alter(x + saved_x, y + saved_y)
      return

    t_point = Point2D(x, y)  # translation vector

    for p_num in range(len(self.points)):
      self.points[p_num] += t_point

  def setorigin(self):
    """Sets anchor point to origin."""
    t_point = self.points[0]  # translation vector
    self.translate(-t_point.x(), -t_point.y())

  def _get_trig(degrees):
    # for more accurate trig with rationals
    rad = math.radians(degrees)
    sine = 0
    cosine = 0
    rem = degrees % 360

    if rem in (30, 150):
      sine = 0.5
      cosine = math.cos(rad)
    elif rem in (60, 300):
      sine = math.sin(rad)
      cosine = 0.5
    elif rem in (210, 330):
      sine = -0.5
      cosine = math.cos(rad)
    elif rem in (120, 240):
      sine = math.sin(rad)
      cosine = -0.5
    elif rem == 0:
      sine = 0
      cosine = 1
    elif rem == 90:
      sine = 1
      cosine = 0
    elif rem == 180:
      sine = 0
      cosine = -1
    elif rem == 270:
      sine = -1
      cosine = 0
    else:
      sine = math.sin(rad)
      cosine = math.cos(rad)
    
    return sine, cosine

  def reflect(self, degrees):
    """Reflects self in a line defined by degrees.

    :param degrees: angle between line through anchor and
      line parallel to x axis
    """
    
    sin, cos = Shape2D._get_trig(2*degrees)

    saved_anchor = self.points[self.anchor]
    self.setorigin()

    t_mat = Transformation2D([
      [cos, sin],
      [sin, -cos]
    ])
    self.transform(t_mat)

    self.translate(saved_anchor.x(), saved_anchor.y())

  def rotate(self, degrees):
    """Applies rotation to self about anchor.

    :param degrees: anticlockwise angle from line parallel to x axis.
    """

    sin, cos = Shape2D._get_trig(degrees)

    saved_anchor = self.points[self.anchor]
    self.setorigin()

    t_mat = Transformation2D([
      [cos, -sin],
      [sin, cos]
    ])
    self.transform(t_mat)
    
    self.translate(saved_anchor.x(), saved_anchor.y())

  def enlarge(self, x, y):
    """Applies enlargement to self.

    :param x: factor horizontally.
    :param y: factor vertically.
    :return: None if self is a single point.
    """

    if len(self.points) == 1:
      self.points[0].plotsize *= x
      return

    saved_anchor = self.points[self.anchor]
    self.setorigin()

    t_mat = Transformation2D([
      [x, 0],
      [0, y]
    ])
    self.transform(t_mat)
    self.translate(saved_anchor.x(), saved_anchor.y())

  def copy(self):
    return Shape2D(self.points[:])

  def deepcopy(self):
    new_points = []
    for p in self.points:
      new_points.append(p.deepcopy())
    return Shape2D(new_points)

  def plot(self, ax):
    """Plots shape using Matplotlib.pyplot.

    :param ax: axis initialized using pyplot.
    """
    if len(self.points) == 1:
      self.points[0].plot(ax)

    x = []
    y = []
    for p in self.points:
      x.append(p.x())
      y.append(p.y())

    ax.fill(x, y)
