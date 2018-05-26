class Mat(object):
  m = 1  # vertical length
  n = 1  # horizontal length
  values = []
  is_float = False


  def __init__(self, vals:list):
    # False for int, True for float:
    vals_float = False

    m_temp = len(vals)
    n_temp = len(vals[0])

    for row in vals:
      if len(row) != n_temp:
        raise TypeError
      for num in row:
        if type(num) == float:
          vals_float = True
        elif type(num) != int:
          raise TypeError
    
    self.values = vals[:]
    self.m = m_temp
    self.n = n_temp

    if vals_float:
      self.makefloat()


  def __repr__(self):
    text = "\n"
    for row in self.values:
      text += "{}\n".format(row)
    return text  


  def zeros(vert_num, hor_num):  # no need for self here
    return Mat([[0]*hor_num for i in range(vert_num)])
  

  def ones(vert_num, hor_num):
    return Mat([[1]*hor_num for i in range(vert_num)])


  def identity(num):
    temp_mat = Mat.zeros(num, num)
    for row_num in range(num):
      temp_mat.values[row_num][row_num] = 1
    return temp_mat


  def randuniform(vert_num, hor_num, start=0, stop=1):  # returns float
    import random
    new_values = [[random.uniform(start, stop) for i in range(hor_num)]
      for j in range(vert_num)
    ]
    return Mat(new_values)


  def randint(vert_num, hor_num, start, stop):  # inclusive
    import random
    new_values = [[random.randint(start, stop) for i in range(hor_num)]
      for j in range(vert_num)
    ]
    return Mat(new_values)


  def _check_float(obj):  # no need for self in private method
    if obj.is_float:
      return True

    for row in obj.values:

      for val in row:
        if type(val) == float:
          obj.is_float = True
          return True
        elif type(val) != int:
          raise TypeError
    
    return False


  def makefloat(self):  # makes elements all float
    self.is_float = True
    for row_num in range(self.m):
        self.values[row_num] = list(map(
          float, self.values[row_num]
        ))
    return
  

  def _add(val1, val2):  # used for map
    return val1 + val2
  

  def _mul(val1, val2):
    return val1 * val2


  def __add__(self, other):
    temp_mat = self.copy()

    if type(other) == type(temp_mat):

      if temp_mat.m == other.m and temp_mat.n == other.n:
        for row_num in range(len(temp_mat.values)):
          temp_mat.values[row_num] = list(map(
            Mat._add,
            temp_mat.values[row_num], other.values[row_num]
          ))
      else:
        raise TypeError

    elif type(other) in (int, float):

      for row_num in range(len(temp_mat.values)):
        temp_mat.values[row_num] = list(map(
          lambda x: x + other,
          temp_mat.values[row_num]
        ))

    else:
      raise TypeError
    
    if Mat._check_float(temp_mat):
      temp_mat.makefloat()
    
    return temp_mat


  def __sub__(self, other):
    return self + other * -1


  def power(self, other):
    if type(other) not in (int, float):
      raise TypeError
    temp_mat = self.copy()

    for row_num in range(len(temp_mat.values)):
      temp_mat.values[row_num] = list(map(
        lambda x: x ** other,
        temp_mat.values[row_num]
      ))

    return temp_mat


  def _check_mul_compat(obj, other):
    return obj.n == other.m and type(obj) == type(other)


  def matmul(self, other):
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
      raise TypeError

    return Mat(new_values)


  def __mul__(self, other):
    temp_mat = None

    if type(other) == int or type(other) == float:
      temp_mat = self.simplemul(other)

      if Mat._check_float(temp_mat):
        temp_mat.makefloat()

    else:
      temp_mat = self.matmul(other)

    return temp_mat


  def simplemul(self, other):  # allows other to be Fraction
    temp_mat = self.copy()
    for row_num in range(len(temp_mat.values)):
      temp_mat.values[row_num] = list(map(
        lambda x: x * other,
        temp_mat.values[row_num]
      ))
    return temp_mat


  def dotmul(self, other):
    if type(self) != type(other) or \
      self.m != other.m or self.n != other.n:
      raise TypeError
    temp_mat = self.copy()
    
    for row_num in range(temp_mat.m):
      temp_mat.values[row_num] = list(map(
        Mat._mul,
        temp_mat.values[row_num], other.values[row_num]
      ))
    return temp_mat


  def cofactors(self):
    if self.m <= 2:
      return self

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
    the_det = self.det()  # includes dimension validation
    if self.det() == 0:
      return
    
    the_adj = self.cofactors().transpose()

    return the_adj * (1/the_det)


  def fracinverse(self):  # returns Mat of Fraction
    the_det = self.det()
    if self.det() == 0:
      return
    
    from fractions import Fraction
    the_adj = self.cofactors().transpose()

    return the_adj.simplemul(Fraction(1, the_det))


  def det(self):
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
    new_values = self.values[:]
    return Mat(new_values)


  def deepcopy(self):
    import copy
    new_values = copy.deepcopy(self.values)
    return Mat(new_values)


  def get(self, key1, key2):
    if key2 == "all":
      return self.values[key1]
    return self.values[key1][key2]


  def alter(self, key1, key2, val):
    self.values[key1][key2] = val
    if Mat._check_float(self):
      self.makefloat()
    return


  def delcol(self, col_num):
    for row_num in range(self.m):
      del self.values[row_num][col_num]
    self.n -= 1
    return
  

  def delrow(self, row_num):
    del self.values[row_num]
    self.m -= 1
    return


  def insert(self, vals:list, pos:int = 0, orientation:str = "r"):
    """
    pos - index of matrix at which to insert new values
    orientation - 'r' for row or 'c' for column
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
    
    return
  

  def transpose(self):
    new_values = list(map(list, zip(*self.values)))
    return Mat(new_values)
