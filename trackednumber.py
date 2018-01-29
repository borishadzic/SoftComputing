class TrackedNumber:

    __blue = False
    __green = False

    def __init__(self, coords):
      self.__coords = coords
  
    def update_position(self, coords):
      self.__coords = coords

    def get_bottom_right(self):
      return (self.__coords[0] + self.__coords[2], 
              self.__coords[1] + self.__coords[3])

    def get_top_left(self):
      return (self.__coords[0], self.__coords[1])

    def setPassedBlue(self, value):
        self.__blue = value

    def passedBlue(self):
      return self.__blue

    def setPassedGreen(self, value):
        self.__green = value  

    def passedGreen(self):
      return self.__green

