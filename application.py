import cv2
import math
import numpy as np

from keras import models
from scipy.spatial import distance
from trackednumber import TrackedNumber
from lines import get_lines, draw_lines, Line

def findClosest(number, prevNumbers):
    """
    Pronalazi prethodni element iz niza koji najbolje odgovara
    prosledjenom broju.
    """
    closeNumbers = []

    for prevNumber in prevNumbers:
        dist = distance.euclidean(number.get_bottom_right(), 
                                  prevNumber.get_bottom_right())

        if dist < 20:
            closeNumbers.append([dist, prevNumber])

    closeNumbers = sorted(closeNumbers, key=lambda x: x[0])   

    if len(closeNumbers) > 0:
        return closeNumbers[0][1]
    else:
        return None

def update_tracked_numbers(img_bin, trackedNumbers):
    """
    Pronalazi konture sa slike, za svaku konturu pokusava da pronadje 
    element sa prethodnog frejma koji odgovara novoj konturi. Ako je element
    pronadjen njegova pozicija ce biti azurirana sa koordinatama nove konture.
    U slucaju da prethodni element nije pronadjen, kontura ce biti sacuvana
    kao novi element u listu
    """
    contours = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    for contour in contours:
        coords = cv2.boundingRect(contour)
        
        # ako je visina konture veca od 9
        if coords[3] > 9:
            number = TrackedNumber(coords)

            prevNumber = findClosest(number, trackedNumbers)

            if prevNumber is None:
                # ako nije pronadjen prethnodni broj koji odgovara novoj konturi
                # dodaj ga u listu
                trackedNumbers.append(number)       
            else:
                # ako je pronadjen prethodni broj koji odgovara novoj konturi
                # aziraraj njegovu poziciju novim kooordinatama
                prevNumber.update_position(coords)


def prepare_for_ann(img_bin, number):
    """
    Metoda koja sece deo slike sa binarne slike tako da obuhvati ceo broj.
    Iseceni deo slike na kojoj se nalazi trazeni broj se kasnije pretvara u oblik
    pogodan za predikciju u neuronskoj mrezi
    """
    # uzmi koordinate gornjeg levog i donjeg desnog coska konture
    p1 = number.get_top_left()
    p2 = number.get_bottom_right()

    # sa binarne slike iseci deo slike koji odgovara tim koordinatama
    # i jos 7 piksela sa svih strana slike
    extra = 7
    img_number = img_bin[p1[1] - extra : p2[1] + extra, 
                         p1[0] - extra - 1 : p2[0] + extra + 1]  

    # u slucaju da u okolnih sedam piksela uhvati deo drugog broja
    # promeni okolnih sedam piksela na crnu boju
    rows, cols = img_number.shape

    img_number[0: extra] = 0
    img_number[rows - extra: rows] = 0
    img_number[:, 0: extra] = 0
    img_number[:, cols-extra: cols] = 0

    img_number = cv2.GaussianBlur(img_number, (5, 5), 1)
    # pretvori ga u oblik pogodan za predikciju u neuronskoj mrezi
    resized = cv2.resize(img_number, (28, 28), interpolation = cv2.INTER_NEAREST)
    scaled = resized / 255
    flattened = scaled.flatten()

    return np.reshape(flattened, (1, 784))

def get_prediciton(model, img_number):
    """
    Pomocna metoda koja vraca prediktovanu vrednost za prosledjeni broj
    """
    predicted_result = model.predict(img_number)
    final_result = np.argmax(predicted_result)

    return final_result

def display_prediction(frame, predicted_number, number):
    """
    Pomocna metoda koja na prodledjenoj slici iscrtava prediktovanu vrednost
    """
    p1 = number.get_top_left()
    p2 = number.get_bottom_right()
    cv2.rectangle(frame, p1, p2, (0,255,0), 2)
    cv2.putText(frame, str(predicted_number), (p1[0], p1[1] - 3), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

def display_sum(frame, sum_blue, sum_green):
    """
    Pomocna metoda koja ispisuje trenutne sume. Plavom bojom je ispisana
    suma brojeva koji su prosli kroz plavu liniju. Zelenom bojom je ispisana suma
    brojeva koji su prosli kroz zelenu liniju
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(sum_blue), (10, 45), font, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, str(sum_green), (10, 75), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Ukupno ' + str(sum_blue-sum_green), (10, 105), font,  0.8, (0, 255, 255), 1, cv2.LINE_AA)

def filter_edge_numbers(number):
    """
    Pomocna metoda koja pronalazi elemnete koji se nalaze blizu desne ili donje ivice
    """
    return not (number.get_bottom_right()[1] > 470 or number.get_bottom_right()[0] > 620)

def draw_tracked_numbers(frame, trackedNumbers):
    """
    Pomocna metoda koja iscrtava elemente koji su trenutno praceni
    """
    for number in trackedNumbers:
        cv2.rectangle(frame, number.get_top_left(), number.get_bottom_right(), (0, 255, 0), 1)

def main(model, video_src, debug = False):
    cap = cv2.VideoCapture(video_src)
    
    first_frame = cap.read()[1]
    blue_line, green_line = get_lines(first_frame)

    sum_blue, sum_green = 0, 0
    trackedNumbers = []

    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        mask = cv2.inRange(frame, 
                           np.array([160, 160, 160], dtype="uint8"), 
                           np.array([255, 255, 255], dtype="uint8"))
        whiteImage = cv2.bitwise_and(frame, frame, mask = mask)
        whiteImage = cv2.cvtColor(whiteImage, cv2.COLOR_BGR2GRAY)
        img_bin = cv2.threshold(whiteImage, 1, 255, cv2.THRESH_BINARY)[1]

        update_tracked_numbers(img_bin, trackedNumbers)     
        
        pause = False

        for number in trackedNumbers:

            if not number.passed_blue():
                if blue_line.has_passed(number):
                    number.set_passed_blue(True)

                    img_number = prepare_for_ann(img_bin, number)
                    predicted_number = get_prediciton(model, img_number)
                    if debug: 
                        display_prediction(frame, predicted_number, number)
                    sum_blue += predicted_number

                    pause = True

            if not number.passed_green():
                if green_line.has_passed(number):
                    number.set_passed_green(True)

                    img_number = prepare_for_ann(img_bin, number)
                    predicted_number = get_prediciton(model, img_number)
                    if debug:
                        display_prediction(frame, predicted_number, number)
                    sum_green += predicted_number

                    pause = True

        # izbaci elemente koji su blizu desnoj ili donjoj ivici slike
        trackedNumbers = list(filter(filter_edge_numbers, trackedNumbers))
   
        if debug:
            # prikazi elemente koji su trenutno praceni
            draw_tracked_numbers(frame, trackedNumbers)        
            # ispisi trenutnu sumu
            display_sum(frame, sum_blue, sum_green)
            # iscrtaj linije
            draw_lines(frame, [blue_line, green_line])
            cv2.imshow(video_src, frame)

        if pause and debug:
            pause = False
            cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

    return sum_blue - sum_green

if __name__ == '__main__':
    model = models.load_model('model.h5')
    main(model, 'Video/video-0.avi', True)