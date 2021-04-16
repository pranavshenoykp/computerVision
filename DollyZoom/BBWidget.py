"""
References: https://stackoverflow.com/questions/55149171/how-to-get-roi-bounding-box-coordinates-with-mouse-clicks-instead-of-guess-che
"""

import cv2
import numpy as np

class BoundingBoxWidget(object):
    def __init__(self, image):
        # self.original_image = cv2.imread(fname)
        self.original_image = image
        self.clone = self.original_image.copy()
        print("A new window has opened")
        print("Drag to create a bounding box")
        print("You can redraw a bounding box if needed")
        print("Press Q to save and exit")

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # Bounding box reference points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse button release
        elif event == cv2.EVENT_LBUTTONUP:
            self.clone = self.original_image.copy()
            self.image_coordinates.append((x,y))
            print('top left: {}, bottom right: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
            print('x,y,w,h : ({}, {}, {}, {})'.format(self.image_coordinates[0][0], self.image_coordinates[0][1], self.image_coordinates[1][0] - self.image_coordinates[0][0], self.image_coordinates[1][1] - self.image_coordinates[0][1]))

            # Draw rectangle 
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
            cv2.imshow("image", self.clone) 

        # Clear drawing boxes on right mouse button click
        # elif event == cv2.EVENT_RBUTTONDOWN:
        #     self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

    def return_coordinates(self):
        return self.image_coordinates

def Open_BB_widget(image):
    boundingbox_widget = BoundingBoxWidget(image)
    coordinates = [(0,0),(0,0)]
    while True:
        cv2.imshow('image', boundingbox_widget.show_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord('q'):
            coordinates = boundingbox_widget.return_coordinates()
            cv2.destroyAllWindows()
            return coordinates