# import the necessary packages
from imutils.video import VideoStream
from pyzbar import pyzbar
import argparse
import datetime
import imutils
import time
import cv2
from collections import deque
import numpy as np
import urllib #for reading image from URL
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(20,GPIO.OUT)
GPIO.setup(16,GPIO.OUT)
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default="barcodes.csv",
	help="path to output CSV file containing barcodes")
args = vars(ap.parse_args())
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
#vs = VideoStream(usefswebcam=True).start()
time.sleep(2.0)

# define the lower and upper boundaries of the colors in the HSV color space
lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(97, 100, 117)} #assign new item lower['blue'] = (93, 10, 0)
upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(117,255,255)}

# define standard colors for circle around the object
colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0)}
 
# open the output CSV file for writing and initialize the set of
# barcodes found thus far
csv = open(args["output"], "w")
found = set()
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it to
	# have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=390)

	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	#for each color in dictionary check object in frame
	for key, value in upper.items():
	    # construct a mask for the color from dictionary`1, then perform
	    # a series of dilations and erosions to remove any small
            # blobs left in the mask
            kernel = np.ones((9,9),np.uint8)
            mask = cv2.inRange(hsv, lower[key], upper[key])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
            # find contours in the mask and initialize the current
            # (x, y) center of the ball
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None
             
            # only proceed if at least one contour was found
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
                # only proceed if the radius meets a minimum size. Correct this value for your obect's size
                if radius > 0.5:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                    cv2.putText(frame,key + " ball", (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
                    GPIO.output(20,1)

	# find the barcodes in the frame and decode each of the barcodes
	GPIO.output(16,0)
	barcodes = pyzbar.decode(frame)
# loop over the detected barcodes
	for barcode in barcodes:
		# extract the bounding box location of the barcode and draw
		# the bounding box surrounding the barcode on the image
		(x, y, w, h) = barcode.rect
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
		# the barcode data is a bytes object so if we want to draw it
		# on our output image we need to convert it to a string first
		barcodeData = barcode.data.decode("utf-8")
		barcodeType = barcode.type
 
		# draw the barcode data and barcode type on the image
		text = "{} ({})".format(barcodeData, barcodeType)
		cv2.putText(frame, text, (x, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		GPIO.output(16,1)
		# if the barcode text is currently not in our CSV file, write
		# the timestamp + barcode to disk and update the set
		if barcodeData not in found:
			csv.write("{},{}\n".format(datetime.datetime.now(),
				barcodeData))
			csv.flush()
			found.add(barcodeData)
			
	else:
                GPIO.output(20,0)
		

# show the output frame
	cv2.imshow("Barcode Scanner", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# close the output CSV file do a bit of cleanup
print("[INFO] cleaning up...")
csv.close()
cv2.destroyAllWindows()
vs.stop()