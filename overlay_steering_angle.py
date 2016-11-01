import cv2
import os

def get_degrees(radians):
    return (radians * 180.0) / 3.14

def point_on_circle(center, radius, angle):
    """ Finding the x,y coordinates on circle, based on given angle
    """
    from math import cos, sin, pi
    #center of circle, angle in degree and radius of circle
    shift_angle = -3.14 / 2
    x = center[0] + (radius * cos(shift_angle + angle))
    y = center[1] + (radius * sin(shift_angle + angle))

    return int(x), int(y)

input_image_dir='/media/drive/Challenge 2/Test/center'
output_image_dir='/home/ubuntu/outputcv3'
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

with open('submission.1477958584.csv', 'r') as f:
  filename_angles = []
  for line in f:
    if "frame" not in line:
      parts = line.strip().split(',')
      filename_angles.append((parts[0]+".png", int(parts[0]), float(parts[1])))

center=(320, 400)
radius=50
for filename_angle in filename_angles:
  print filename_angle[0]
  cv_image = cv2.imread(os.path.join(input_image_dir, filename_angle[0]))
  cv2.circle(cv_image, center, radius, (255, 255, 255), thickness=4, lineType=8)
  x,y = point_on_circle(center, radius, filename_angle[2]) 
  cv2.circle(cv_image, (x,y), 6, (255, 0, 0), thickness=6, lineType=8)
  cv2.putText(cv_image, "angle: " + ("%.5f")%(get_degrees(filename_angle[2])), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
  cv2.imwrite(os.path.join(output_image_dir, filename_angle[0]), cv_image)

