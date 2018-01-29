from application import main
from keras import models

import winsound

model = models.load_model('model3.h5')
student_results = []

for i in range(0, 10):
    video_src = 'Video/video-' + str(i) + '.avi'
    result = main(model, video_src)
    student_results.append(float(result))
    print('video-' + str(i) + '.avi ', str(result))

res = []
n = 0
with open('Video/res.txt') as file:	
    data = file.read()
    lines = data.split('\n')
    for id, line in enumerate(lines):
        if(id>0):
            cols = line.split('\t')
            if(cols[0] == ''):
                continue
            cols[1] = cols[1].replace('\r', '')
            res.append(float(cols[1]))
            n += 1

diff = 0
for index, res_col in enumerate(res):
    diff += abs(res_col - student_results[index])
percentage = 100 - abs(diff/sum(res))*100

print('Procenat tacnosti:\t'+str(percentage))
print('Ukupno:\t'+str(n))

frequency = 2500
duration = 1000
winsound.Beep(frequency, duration)

    