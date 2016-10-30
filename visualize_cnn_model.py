import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from keras import backend as K
from keras.models import load_model

# Only supports convolution filters
def visualize_cnn_model(model_file, image_file, output_dir):
  model = load_model(model_file)
  print "model json:"
  print model.to_json()

  X = np.load(open(image_file, 'r'))

  #normalize image
  X = (X - (255.0/2))/255.0
  X = np.expand_dims(X, axis=0)

  num_layers = len(model.layers)

  for i in range(1, num_layers-1):
    if 'convolution' in model.layers[i].name:
      get_nth_layer_output = K.function([model.layers[0].input], [model.layers[i].output])
      layer_output = get_nth_layer_output([X])[0]
      print model.layers[i].name
      print layer_output.shape
      fig = plt.figure(figsize=(30.0, 15.0))
      for filter_num in range(0, layer_output.shape[3]):
        fig.add_subplot(layer_output.shape[3]/8, 8, filter_num)
        plt.imshow(layer_output[0, :, :, filter_num])
      plt.savefig(os.path.join(output_dir, model.layers[i].name + ".png"))

if __name__ == '__main__':
  if (len(sys.argv) <= 3):
     print "python visualize_cnn_model.py <model.h5 file> <image.jpg.npy file> <output dir>"
     sys.exit()
  model_file = sys.argv[1]
  image_file = sys.argv[2]
  output_file = sys.argv[3]
  visualize_cnn_model(model_file, image_file, output_file)

