import numpy as np
import cv2
import math

from lines import get_lines, draw_lines, Line
from keras import models
from vector import distance
from trackednumber import TrackedNumber

def findClosest(number, prevNumbers):
    """
    Pronalazi prethodni element iz niza koji odgovara 
    prosledjenom broju.
    """
    for prevNumber in prevNumbers:
        dist = distance(number.get_bottom_right(), 
                        prevNumber.get_bottom_right())

        if dist < 20:
            return prevNumber
            
    return None

def update_tracked_numbers(img_bin, trackedNumbers):
    """
    Pronalazi konture sa slike, za svaku konturu pokusava da pronadje 
    element sa prethodnog frejma koji odgovara novoj konturi. Ako je element
    pronadjen njegova pozicija ce biti azurirana sa koordinatama nove konture.
    U slucaju da prethodni element nije pronadjen, kontura ce biti sacuvana
    kao novi element u listu
    """
    contours = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

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
    Iseceni deo slike na kojoj se trazeni broj se kasnije pretvara u oblik
    pogodan za predikciju u neuronskoj mrezi
    """

    # uzmi koordinate gornjeg levog i donjeg desnog coska konture
    p1 = number.get_top_left()
    p2 = number.get_bottom_right()

    # sa binarne slike iseci deo slike koji odgovara tim koordinatama
    # i jos 7 piksela sa svih strana slike
    extra = 7
    img_number = img_bin[p1[1] - extra : p2[1] + extra, 
                         p1[0] - extra : p2[0] + extra]  
    img_number = cv2.GaussianBlur(img_number, (5, 5), 1)

    # u slucaju da u okolnih sedam piksela uhvati deo drugog broja
    # promeni okolnih sedam piksela na crnu boju
    img_number[0:7] = 0
    img_number[:,0:7] = 0
    img_number[0:7:-1] = 0
    img_number[:,0:7:-1] = 0

    # pretvori ga u oblik pogodan za predikciju u neuronskoj mrezi
    resized = cv2.resize(img_number, (28, 28), interpolation = cv2.INTER_NEAREST)
    scale = resized / 255
    mVector = scale.flatten()
    mColumn = np.reshape(mVector, (1, 784))

    return np.array(mColumn, dtype=np.float32)

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def get_result_from_alphabet(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

def get_prediciton(model, img_number):
    alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    predicted_result = model.predict(img_number)
    final_result = get_result_from_alphabet(predicted_result, alphabet)[0]

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

def main(model, video_src, canPause = False):
    cap = cv2.VideoCapture(video_src)
    
    first_frame = cap.read()[1]
    blue_line, green_line = get_lines(first_frame)

    sum_blue, sum_green = 0, 0
    trackedNumbers = []

    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_bin = cv2.threshold(img_gray, 160, 255, cv2.THRESH_BINARY)[1]

        update_tracked_numbers(img_bin, trackedNumbers)     
        
        pause = False

        for number in trackedNumbers:
            if not number.passedBlue():
                if blue_line.has_passed(number):
                    number.setPassedBlue(True)

                    img_number = prepare_for_ann(img_bin, number)
                    predicted_number = get_prediciton(model, img_number)
                    display_prediction(frame, predicted_number, number)
                    sum_blue += predicted_number

                    pause = True

            if not number.passedGreen():
                if green_line.has_passed(number):
                    number.setPassedGreen(True)

                    img_number = prepare_for_ann(img_bin, number)
                    predicted_number = get_prediciton(model, img_number)
                    display_prediction(frame, predicted_number, number)
                    sum_green += predicted_number

                    pause = True

        # display_sum(frame, sum_blue, sum_green)
        # draw_lines(frame, [blue_line, green_line])

        cv2.imshow(video_src, frame)

        if pause and canPause:
            pause = False
            cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

    return sum_blue - sum_green

if __name__ == '__main__':
    model = models.load_model('model3.h5')
    main(model, 'Video/video-2.avi', True)