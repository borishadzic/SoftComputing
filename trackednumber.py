class TrackedNumber:

    def __init__(self, coords):
      self.__coords = coords
      self.__blue = False
      self.__green = False

    def update_position(self, coords):
      self.__coords = coords

    def get_bottom_right(self):
      return (self.__coords[0] + self.__coords[2], 
              self.__coords[1] + self.__coords[3])

    def get_top_left(self):
      return (self.__coords[0], self.__coords[1])

    def set_passed_blue(self, value):
        self.__blue = value

    def passed_blue(self):
      return self.__blue

    def set_passed_green(self, value):
        self.__green = value  

    def passed_green(self):
      return self.__green

