import cv2
import numpy as np

from scipy.spatial import distance
from trackednumber import TrackedNumber

class Line:
    def __init__(self, line):
        coords = line[0]
        self.__p1 = (coords[0], coords[1])
        self.__p2 = (coords[2], coords[3])
  
    def get_first_point(self):
        """
        Vraca koordinate prve (leve) tacke linije.
        Primer (26, 78)
        """
        return self.__p1
    
    def get_second_point(self):
        """
        Vraca koordinate druge (desne) tacke linije.
        Primer (60, 20)
        """
        return self.__p2

    def has_passed(self, number : TrackedNumber):
        """
        Proverava udaljenost proslednjenog broja od samog broja.
        Za udaljenost manju od 5 racuna se da je broj prosao liniju.
        U obzir uzima samo brojeve koji se nalazi izmedju x koordinata
        same linije.
        """
        dot = number.get_bottom_right()

        if self.__p1[0] < dot[0] and dot[0] < self.__p2[0]:
            p1 = np.array(self.__p1)
            p2 = np.array(self.__p2)
            p3 = np.array(dot)
            dist = float(np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1))

            return dist <= 7
        else:
            return False

def get_lines(frame):
    """
    Glavna metoda koja poziva sve ostale. Vraca 
    dve linije linije pronadjene hough transformacijom.
    Prva linija je plava, druga je zelena
    """
    blue_output, green_output = separate_lines(frame)

    blue_lines = find_all_lines(blue_output)
    blue_line = longest_line(blue_lines)

    green_lines = find_all_lines(green_output)
    green_line = longest_line(green_lines)

    return Line(blue_line), Line(green_line)

def find_all_lines(img):
    """
    Metoda koja pomocu hough transformacije pronalazi sve prave linije koje 
    se nalaze na slici
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bin = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)[1]

    img_edges = cv2.Canny(img_bin, 100, 250)
    img_blur = cv2.GaussianBlur(img_edges, (7, 7), 1)
    lines = cv2.HoughLinesP(img_blur, 1,  np.pi / 180, 50, 50, 150)

    return lines

def longest_line(lines):
    """
    Pomocna funkcija koja pronalazi najduzu liniju iz prosledjenih linija
    """
    l_line = lines[0]

    coords = l_line[0]
    al = (coords[0], coords[1])
    bl = (coords[2], coords[3])
    m_dist = distance.euclidean(al,bl) 

    for line in lines:
        coords = line[0]
        a = (coords[0], coords[1])
        b = (coords[2], coords[3])
        dst = distance.euclidean(a,b)

        if dst >= m_dist:
            m_dist = dst
            l_line = line

    return l_line

def draw_lines(img, lines):
    """
    Pomocna funkcija koja iscrtava linije pronadjene hough transfromacijom nad
    prosledjenom slikom
    """
    for line in lines:
        cv2.line(img, line.get_first_point(), line.get_second_point(), (0, 0, 255), 1)

def separate_lines(image):
    """
    Pomocna funkcija koja sa prosledjene slike izdvaja zelene i plave regione.
    Povratna vrednost funkcije su dve slike. Prva slika sadrzi samo plavu liniju,
    druga slika sadrzi samo zelenu liniju.

    """
    blueBoundaries = ([120, 0, 0], [255, 100, 100])  
    greenBoundaries = ([0, 130, 0], [50, 255, 50])

    lower = np.array(blueBoundaries[0], dtype = "uint8")
    upper = np.array(blueBoundaries[1], dtype = "uint8")
    mask = cv2.inRange(image, lower, upper)
    blue_output = cv2.bitwise_and(image, image, mask = mask)

    lower = np.array(greenBoundaries[0], dtype = "uint8")
    upper = np.array(greenBoundaries[1], dtype = "uint8")
    mask = cv2.inRange(image, lower, upper)
    green_output = cv2.bitwise_and(image, image, mask = mask)

    return blue_output, green_output

def main():
    cap = cv2.VideoCapture('Video/video-9.avi')
    ret, frame = cap.read()

    if ret:
        blue_output, green_output = separate_lines(frame)
        blue_line, green_line = get_lines(frame)

        draw_lines(blue_output, [blue_line])
        draw_lines(green_output, [green_line])

        cv2.imshow('Separated Lines', np.concatenate((blue_output, green_output), axis=1))
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()   

if __name__ == '__main__':
    main()



