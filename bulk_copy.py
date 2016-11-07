import shutil
import multiprocessing
import sys
import os

input_dataset = sys.argv[1]
output_dataset = sys.argv[2]
if not os.path.exists(output_dataset):
  os.makedirs(output_dataset)


input_images_dir = os.path.join(input_dataset, "images")
output_images_dir = os.path.join(output_dataset, "images")
if not os.path.exists(output_images_dir):
  os.makedirs(output_images_dir)

shutil.copyfile(os.path.join(input_dataset, "labels"), os.path.join(output_dataset, "labels"))
num_images = len(os.listdir(input_images_dir))

tasks = []
for i in range(1, num_images+1):
  tasks.append((str(i)+".png.npy", input_images_dir, output_images_dir))

def copy_file(args):
   i, input_images_dir, output_images_dir = args
   shutil.copyfile(os.path.join(input_images_dir, i), os.path.join(output_images_dir, i))

pool=multiprocessing.Pool(8)
pool.map(copy_file, tasks)
