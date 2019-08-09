import tensorflow as tf
import numpy as np
import os

from model_para    import *
from cnn_bl_conf   import *
from dnn_bl01_conf import *
from dnn_bl02_conf import *
from add_bl_conf   import *


#======================================================================================================#

class model_conf(object):

    def __init__( self):

        self.model_para = model_para()

        # ============================== Fed Input
        #input data
        self.input_layer_val     = tf.placeholder(tf.float32, [None, self.model_para.n_freq, self.model_para.n_time, self.model_para.n_chan], name="input_layer_val")
        self.input_layer_val_f01 = self.input_layer_val[:,0:128,  :,:] #mel
        self.input_layer_val_f02 = self.input_layer_val[:,128:256,:,:] #gam
        self.input_layer_val_f03 = self.input_layer_val[:,256:384,:,:] #cqt
        
        #expected class
        self.expected_classes    = tf.placeholder(tf.float32, [None, self.model_para.n_class], name="expected_classes")

        #run mode
        self.mode                = tf.placeholder(tf.bool, name="running_mode")

        #============================== NETWORK CONFIGURATION

        # Call Batchnorm
        with tf.device('/gpu:0'), tf.variable_scope("fist_batch_norm_01")as scope:
             self.input_layer_val_01 = tf.contrib.layers.batch_norm(self.input_layer_val_f01, 
                                                                    is_training = self.mode, 
                                                                    decay = 0.9,
                                                                    zero_debias_moving_mean=True
                                                                   )

        with tf.device('/gpu:0'), tf.variable_scope("fist_batch_norm_02")as scope:
             self.input_layer_val_02 = tf.contrib.layers.batch_norm(self.input_layer_val_f02, 
                                                                    is_training = self.mode, 
                                                                    decay = 0.9,
                                                                    zero_debias_moving_mean=True
                                                                   )

        with tf.device('/gpu:0'), tf.variable_scope("fist_batch_norm_03")as scope:
             self.input_layer_val_03 = tf.contrib.layers.batch_norm(self.input_layer_val_f03, 
                                                                    is_training = self.mode, 
                                                                    decay = 0.9,
                                                                    zero_debias_moving_mean=True
                                                                   )

        # Call CNN and Get CNN output
        with tf.device('/gpu:0'), tf.variable_scope("cnn_01")as scope:
            self.cnn_ins_01 = cnn_bl_conf(self.input_layer_val_01, self.mode)
        with tf.device('/gpu:0'), tf.variable_scope("cnn_02")as scope:
            self.cnn_ins_02 = cnn_bl_conf(self.input_layer_val_02, self.mode)
        with tf.device('/gpu:0'), tf.variable_scope("cnn_03")as scope:
            self.cnn_ins_03 = cnn_bl_conf(self.input_layer_val_03, self.mode)

        self.cnn_ins_01_output = self.cnn_ins_01.final_output
        self.cnn_ins_02_output = self.cnn_ins_02.final_output
        self.cnn_ins_03_output = self.cnn_ins_03.final_output

        # Call ADD and Get Add output
        self.add_ins_01 = add_bl_conf(self.cnn_ins_01_output, self.cnn_ins_02_output, self.cnn_ins_03_output)

        self.add_ins_01_output     = self.add_ins_01.final_output
        self.add_ins_01_output_dim = self.add_ins_01.final_output_dim

        # Call DNN and Get DNN output
        self.dnn_bl02_ins_01 = dnn_bl02_conf(self.cnn_ins_01_output, self.add_ins_01_output_dim, self.mode)
        self.dnn_bl02_ins_02 = dnn_bl02_conf(self.cnn_ins_02_output, self.add_ins_01_output_dim, self.mode)
        self.dnn_bl02_ins_03 = dnn_bl02_conf(self.cnn_ins_03_output, self.add_ins_01_output_dim, self.mode)
 
        self.output_layer_b1 = self.dnn_bl02_ins_01.final_output  
        self.output_layer_b2 = self.dnn_bl02_ins_02.final_output
        self.output_layer_b3 = self.dnn_bl02_ins_03.final_output

        self.dnn_bl01_ins_01 = dnn_bl01_conf(self.add_ins_01_output, self.add_ins_01_output_dim, self.mode)

        self.output_layer      = self.dnn_bl01_ins_01.final_output
        self.prob_output_layer = tf.nn.softmax(self.output_layer)

        self.wanted_data = self.add_ins_01_output

        #print self.output_layer.get_shape()           #n x nClassa
        #exit()

        ### ======================================== LOSS FUNCTION AND ACCURACY =========================
        ### loss function
        with tf.device('/gpu:0'), tf.variable_scope("loss") as scope:
  
            # l2 loss  
            l2_loss = self.model_para.l2_lamda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

            # main loss
            losses     = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.expected_classes, logits=self.output_layer)

            # brand loss
            losses_b1  = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.expected_classes, logits=self.output_layer_b1)
            losses_b2  = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.expected_classes, logits=self.output_layer_b2)
            losses_b3  = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.expected_classes, logits=self.output_layer_b3)
 
            # final loss
            self.loss = (tf.reduce_mean(losses_b1) + tf.reduce_mean(losses_b2) + tf.reduce_mean(losses_b3))/3 + tf.reduce_mean(losses) + l2_loss

        ### Calculate Accuracy  
        with tf.device('/gpu:0'), tf.name_scope("accuracy") as scope:
            self.correct_prediction    = tf.equal(tf.argmax(self.output_layer,1),    tf.argmax(self.expected_classes,1))
            self.correct_prediction_b1 = tf.equal(tf.argmax(self.output_layer_b1,1), tf.argmax(self.expected_classes,1))
            self.correct_prediction_b2 = tf.equal(tf.argmax(self.output_layer_b2,1), tf.argmax(self.expected_classes,1))
            self.correct_prediction_b3 = tf.equal(tf.argmax(self.output_layer_b3,1), tf.argmax(self.expected_classes,1))

            self.accuracy_org = tf.reduce_mean(tf.cast(self.correct_prediction,"float", name="accuracy" ))
            self.accuracy_b1  = tf.reduce_mean(tf.cast(self.correct_prediction_b1,"float", name="accuracy_b1" ))
            self.accuracy_b2  = tf.reduce_mean(tf.cast(self.correct_prediction_b2,"float", name="accuracy_b2" ))
            self.accuracy_b3  = tf.reduce_mean(tf.cast(self.correct_prediction_b3,"float", name="accuracy_b3" ))

            self.accuracy     = (self.accuracy_org + self.accuracy_b1 + self.accuracy_b2 + self.accuracy_b3)/4
