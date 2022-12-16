import cv2

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
  # Load the input image
  ret, image = cap.read()
  # Convert the image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Use OpenCV's Haar cascade classifier to detect faces in the image

  faces = detector.detectMultiScale(gray, 1.1, 8)

  # Loop over the faces and draw a rectangle around each one
  for (x, y, w, h) in faces:
      cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

  # Show the resulting image
  cv2.imshow('Image with Faces Detected', image)
  k = cv2.waitKey(30) & 0xff
  if k ==27:
    break

cap.release()