# project-PaintX

Made a project using mediapie and CNN architecture using pytorch, that allows user to draw on air,
where webcam tracks handmotion and later predicts what the user is trying to draw.


Gesture Control:

index finger up to draw,
index and middle finger up to be idle,
index, middle and ring finger up to erase.


when being idle bring fingers to top center of screen to predict the drawing.



Datas:
Used datas from Quick! Draw dataset by google
has 7 classes : 'saw', 'crown', 'cup', 'cloud', 'pizza', 'camera', 'face' that can be predicted

used 25000 images each, to run on CNN architecture, and got an 91.1% test accuracy. 
