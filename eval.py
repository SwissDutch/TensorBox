import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
import argparse
import subprocess
import scipy as scp
from scipy.misc import imread, imread, imsave, imresize

import time
import sys
import logging
from random import shuffle


#from train import build_lstm_forward, build_overfeat_forward
from train import build_overfeat_forward
from utils import googlenet_load, train_utils
from utils.annolist import AnnotationLib as al
from utils.stitch_wrapper import stitch_rects
from utils.train_utils import add_rectangles, rescale_boxes

flags = tf.app.flags
FLAGS = flags.FLAGS

reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

tf.app.flags.DEFINE_string('hypes', './hypes/default.json',
                           """HYPES""")

tf.app.flags.DEFINE_string('run', None,
                           """Run to Analyse.""")

def run_eval(H, checkpoint_dir , hypes_file, output_path):
  """Do Evaluation with full epoche of data.

  Args:
    H: Hypes
    checkpoint_dir: directory with checkpoint files
    output_path: path to save results
  """

  #Load GT
  true_idl = H['data']['test_idl']
  true_annos = al.parse(true_idl)

  # define output files
  pred_file = 'val_%s.idl' % os.path.basename(hypes_file).replace('.json', '')
  pred_idl = os.path.join(output_path, pred_file)
  true_file = 'true_%s.idl' % os.path.basename(hypes_file).replace('.json', '')
  true_idl_scaled = os.path.join(output_path, true_file)

  data_folder = os.path.dirname(os.path.realpath(true_idl))

  #Load Graph Model
  tf.reset_default_graph()
  googlenet = googlenet_load.init(H)
  x_in = tf.placeholder(tf.float32, name='x_in')
  if H['arch']['use_lstm']:
    lstm_forward = build_lstm_forward(H, tf.expand_dims(x_in, 0), 
                                 googlenet, 'test', reuse=None)
    pred_boxes, pred_logits, pred_confidences = lstm_forward
  else:
    overfeat_forward = build_overfeat_forward(H, tf.expand_dims(x_in, 0),
                                              googlenet, 'test')
    pred_boxes, pred_logits, pred_confidences = overfeat_forward

  start_time = time.time()  
  saver = tf.train.Saver()
  with tf.Session() as sess:
    logging.info("Starting Evaluation")
    sess.run(tf.initialize_all_variables())

    # Restore Checkpoints
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      logging.info(ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)

    annolist = al.AnnoList()
    trueanno = al.AnnoList()

    #shuffle true_annos to randomize plottet Images
    #shuffle(true_annos)

    for i in range(len(true_annos)):
      true_anno = true_annos[i]
      img = imread( os.path.join(data_folder, true_anno.imageName))

      # Rescale Boxes
      trueanno.append(rescale_boxes(img.shape, true_annos[i],
                                    H["arch"]["image_height"],
                                    H["arch"]["image_width"]))
      # Rescale Images
      img = imresize(img, (H["arch"]["image_height"],
                           H["arch"]["image_width"]), interp='cubic')

      feed = {x_in: img}
      (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes,
                                                       pred_confidences],
                                                      feed_dict=feed)
      pred_anno = al.Annotation()
      pred_anno.imageName = true_anno.imageName
      new_img, rects = add_rectangles([img], np_pred_confidences,
                                      np_pred_boxes, H["arch"],
                                      use_stitching=True,
                                      rnn_len=H['arch']['rnn_len'],
                                      min_conf=0.3)
  
      pred_anno.rects = rects
      annolist.append(pred_anno)

      if i % 20 == 0:
      # Draw every 20th Image; 
      # plotted Image is randomized due to shuffling
        duration = time.time() - start_time
        duration = float(duration)*1000/20
        out_img = os.path.join(output_path, 'test_%i.png'%i)
        scp.misc.imsave(out_img, new_img)
        logging.info('Step %d: Duration %.3f ms'
                                   % (i, duration))
        start_time = time.time()

  annolist.save(pred_idl)
  trueanno.save(true_idl_scaled)

  # write results to disk
  iou_threshold = 0.5
  rpc_cmd = './utils/annolist/doRPC.py --minOverlap %f %s %s' % (iou_threshold, true_idl_scaled,
                                                                 pred_idl)
  rpc_output = subprocess.check_output(rpc_cmd, shell=True)
  txt_file = [line for line in rpc_output.split('\n') if line.strip()][-1]
  output_png = os.path.join(output_path, "roc.png")
  plot_cmd = './utils/annolist/plotSimple.py %s --output %s' % (txt_file, output_png)
  plot_output = subprocess.check_output(plot_cmd, shell=True)


def main(_):
  '''
  Parse command line arguments, load data and create output folder.

  Output will be stored in the rundir/output. The last checkpoint
  in rundir is loaded for evaluation.
    '''

  if FLAGS.run is None:
    logging.error("No Checkpoint dir is provided!") 
    logging.error("Usage: eval.py --run=path/to/checkpointdir --hypes=HYPES")
    exit(1)

  # Get and create output_path
  output_path = os.path.realpath(os.path.join(FLAGS.run,
                                              "eval")) 
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  #Load Hypes

  with open(FLAGS.hypes, 'r') as f:
    H = json.load(f)

  #run evaluation
  run_eval(H,FLAGS.run, FLAGS.hypes, output_path)

  logging.info("Evaluation Complete. Results are saved in: %s",
               output_path)



if __name__ == '__main__':
  tf.app.run()
