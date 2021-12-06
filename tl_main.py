import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
from core import effnetv2_configs
from core import effnetv2_model
from core import autoaugment
import argparse


import os
import PIL
from PIL import Image
import pathlib
import os
import shutil
import glob

"""
How to use
# dataset/<dataset_name>/: path
# dataset/<dataset_name>/train/<class_names>/: train files
# dataset/<dataset_name>/test/imgs/: files for classifying
# dataset/<dataset_name>/test_classified/: classified result
"""

BATCH_SIZE = 16
INPUT_SIZE = 224
LR = 5e-3
EPOCH = 100
AUGMETATION_TYPE = "AUTO" #AUTO/RANDOM/NO

"""
PATH SETUP
"""
parser = argparse.ArgumentParser(description='Effinet v2 TL')
parser.add_argument('--path', type=str, default='', required=True, help="path of model to use")
args = parser.parse_args()

data_dir = args.path +"train/"
data_test_dir = args.path +"test/"
data_cls_dir = args.path +"test_classified/"

data_dir = pathlib.Path(data_dir)
data_test_dir = pathlib.Path(data_test_dir)

"""
data processing
"""
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=99,
  image_size=(INPUT_SIZE, INPUT_SIZE), batch_size=1)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  seed=99,
  subset="validation",
  image_size=(INPUT_SIZE, INPUT_SIZE))

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_test_dir,
  shuffle=False,
  label_mode=None,
  image_size=(INPUT_SIZE, INPUT_SIZE),
  batch_size=BATCH_SIZE)

test_path = test_ds.file_paths

"""
Check dataset
"""
class_names = train_ds.class_names
print("class name: ", class_names)
print("class number: ", len(class_names))

for cls_name in class_names:
  if os.path.isdir(data_cls_dir+cls_name):
    files = glob.glob(data_cls_dir+cls_name+"/*")
    for f in files:
        os.remove(f)
  else:
    os.mkdir(data_cls_dir+cls_name)

"""
data Augmetation & normalize
"""
train_ds = train_ds.unbatch()
if AUGMETATION_TYPE=="RANDOM":
  print("#Augmentation Method: Random Augmentation")
  train_ds = train_ds.map(lambda x, y: (autoaugment.distort_image_with_randaugment(x,2,15), y)).batch(BATCH_SIZE)  
elif AUGMETATION_TYPE=="AUTO":
  print("#Augmentation Method: Auto Augmentation")
  train_ds = train_ds.map(lambda x, y: (autoaugment.distort_image_with_autoaugment(x,"v0"), y)).batch(BATCH_SIZE)
else:
  train_ds=train_ds.batch(BATCH_SIZE)

# normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255) #0~1
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./128., offset=-1) #-1~1
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x: (normalization_layer(x)))

image_batch, labels_batch = next(iter(train_ds))
first_image = image_batch[0]
print("min & max of image: ",np.min(first_image), np.max(first_image))

"""
cache: load on memery after first epoch 
prefetch: prefetch
"""
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

for image_batch, labels_batch in train_ds:
  print("image batch shape: ", image_batch.shape)
  print("label batch shape: ", labels_batch.shape)
  break

"""
core model
"""
eff_model = effnetv2_model.get_model('efficientnetv2-s', include_top=False, training=True)
# eff_model.summary() #for checking summary of Efficientnet v2
initializer = tf.keras.initializers.GlorotNormal()

model = tf.keras.models.Sequential([
  tf.keras.layers.InputLayer(input_shape=[INPUT_SIZE, INPUT_SIZE, 3]),
  eff_model,
  # tf.keras.layers.Dropout(rate=0.2),
  # tf.keras.layers.Dense(2048, activation='relu'),
  tf.keras.layers.Dropout(rate=0.2),
  tf.keras.layers.Dense(16,kernel_initializer=initializer, activation='relu'),

  tf.keras.layers.Dense(len(class_names),kernel_initializer=initializer)
])

eff_model.trainable = False
model.build()
model.summary()


"""
Loss & optimizer
"""
optimizer = keras.optimizers.SGD(learning_rate=LR)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()


"""
step functions
"""
@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
    logits = model(x, training=True)
    loss_value = loss_fn(y, logits)
    loss_value += sum(model.losses)
  grads = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(grads, model.trainable_weights))
  train_acc_metric.update_state(y, logits)
  return loss_value

@tf.function
def val_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)
    return val_logits

@tf.function
def test_step(x):
    val_logits = model(x, training=False)
    return val_logits


"""
train process
"""

for epoch in range(EPOCH):
  print("\nStart of epoch %d" % (epoch,))
  start_time = time.time()
  for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
    loss_value = train_step(x_batch_train, y_batch_train)
    if step % 10 == 0:
      print(
        "Training loss (for one batch) at step %d: %.4f"
        % (step, float(loss_value))
      )
      
  train_acc = train_acc_metric.result()
  print("Training acc over epoch: %.4f" % (float(train_acc),))
  train_acc_metric.reset_states()
  for x_batch_val, y_batch_val in val_ds:
    val_logits = val_step(x_batch_val, y_batch_val)

  val_acc = val_acc_metric.result()
  val_acc_metric.reset_states()
  print("Validation acc: %.4f" % (float(val_acc)))
  print("Time taken: %.2fs" % (time.time() - start_time))

"""
Classify test data
"""
idx = 0
for x_batch_test in test_ds:
    test_logits = test_step(x_batch_test)
    for logit in test_logits:
      index_max = np.argmax(np.array(logit))
      shutil.copy2(test_path[idx],data_cls_dir+class_names[index_max])
      idx+=1
print("!!!!Images are classified on %s"%data_cls_dir)
  

