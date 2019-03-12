from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os
import re
import csv
import tensorflow as tf
import nltk

from caption_generator import CaptionGenerator
from model import ShowAndTellModel
from vocabulary import Vocabulary
from pyrouge import Rouge

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("model_path", r"C:\Users\praneeth\Documents\Studies\Big_Data_Analytics\Tutorials_SourceCode\Tutorial 6 Source Code\medium-show-and-tell-caption-generator-master\model\show-and-tell.pb", "Model graph def path")
tf.flags.DEFINE_string("vocab_file", r"C:\Users\praneeth\Documents\Studies\Big_Data_Analytics\Tutorials_SourceCode\Tutorial 6 Source Code\medium-show-and-tell-caption-generator-master\etc\word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", r"C:\Users\praneeth\Documents\Studies\Big_Data_Analytics\Tutorials_SourceCode\Tutorial 6 Source Code\medium-show-and-tell-caption-generator-master\imgs",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
                       
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def main(_):
    model = ShowAndTellModel(FLAGS.model_path)
    vocab = Vocabulary(FLAGS.vocab_file)
    filenames = _load_filenames()
    generator = CaptionGenerator(model, vocab)
    value=0
    expected = [['a' for i in range(4)] for j in range(len(filenames))]
    for filename in filenames:
        with tf.gfile.GFile(filename, "rb") as f:
            image = f.read()
        captions = generator.beam_search(image)
        print("Captions for image %s:" % os.path.basename(filename))  
        generated = []
        for i, caption in enumerate(captions):
            # Ignore begin and end tokens <S> and </S>.
            sentence = [vocab.id_to_token(w) for w in caption.sentence[1:-1]]
            sentence = " ".join(sentence)
            generated.append(sentence)
            print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
        expected[value][0]=generated[0]
        expected[value][1]=generated[1]
        expected[value][2]=generated[2]
        expected[value][3]=generated[3]
        value+=1
    with open(r'C:\Users\praneeth\Documents\Studies\Big_Data_Analytics\Lab2\Data.csv') as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      data = list(csv_reader)
      Truedata = []
      for index in range(0,len(data)):
        Truedata.append(data[index][1])
    BLEUscore=0    
    r = Rouge()
    for i in range(len(Truedata))   :
       for j in range(4):
          hypothesis = Truedata[i].split()
          reference = expected[i][j].split()
          #there may be several references
          BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
          print ("True data is : %s" % Truedata[i])
          print ("Caption data is : %s" % expected[i][j])
          print ("Rouge Precision is : %f, Recall is : %f, f_score is : %f" % r.rouge_l([expected[i][j]], [Truedata[i]]))
          print ("Bleu score is: ",BLEUscore)    
    
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]    

def _load_filenames():
    filenames = []
    directory = FLAGS.input_files
    for file_pattern in sorted(os.listdir(directory), key=natural_keys):
       filename = os.fsdecode(file_pattern)
       filename = os.path.join(FLAGS.input_files, filename)
       if filename.endswith(".jpg"): 
         filenames.extend(tf.gfile.Glob(filename))
    logger.info("Running caption generation on %d files matching %s",
                len(filenames), FLAGS.input_files)
    return filenames

if __name__ == "__main__":
    tf.app.run()
