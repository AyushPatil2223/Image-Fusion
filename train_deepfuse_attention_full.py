# train_deepfuse_attention_full.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import cv2

# -----------------------
# CONFIGURATION
# -----------------------
BASE_DIR = r"C:\Users\ayush\Downloads\alignimages\preprocessed_ordered\resized_128"
HR_DIR = os.path.join(BASE_DIR, "HR")
LR_DIR = os.path.join(BASE_DIR, "LR")
IR_DIR = os.path.join(BASE_DIR, "IR")

OUT_DIR = r"C:\Users\ayush\Downloads\alignimages\fused_outputs_attention"
os.makedirs(OUT_DIR, exist_ok=True)

# Mixed precision for faster training
#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')


# Allow GPU memory growth (prevents out-of-memory errors)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


# Image size (smaller size → faster training, less memory)
IMG_SIZE = (128, 128)

# Batch size (small for low GPU memory, mixed precision helps speed)
BATCH_SIZE = 4

# Number of epochs (keep same, enough for fusion)
EPOCHS = 50  

# Learning rate (same as before, good starting point)
LEARNING_RATE = 1e-4  

# Data pipeline optimization
AUTOTUNE = tf.data.AUTOTUNE  



# -----------------------
# LOAD DATA
# -----------------------
def load_images(folder, single_channel=False):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    imgs = []
    for f in files:
        img = np.load(os.path.join(folder, f)).astype(np.float32)
        if single_channel and img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        imgs.append(img)
    return np.array(imgs, dtype=np.float32)

HR_imgs = load_images(HR_DIR)
LR_imgs = load_images(LR_DIR)
IR_imgs = load_images(IR_DIR, single_channel=True)

dataset = tf.data.Dataset.from_tensor_slices((HR_imgs, LR_imgs, IR_imgs))
dataset = dataset.shuffle(buffer_size=20).batch(BATCH_SIZE).prefetch(AUTOTUNE)
print(f"Loaded {len(HR_imgs)} HR, {len(LR_imgs)} LR, {len(IR_imgs)} IR images")

# -----------------------
# MODEL COMPONENTS
# -----------------------
def build_encoder(input_shape):
    inp = Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(inp)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    return Model(inp, x)

def build_decoder(input_shape, out_channels=3):
    inp = Input(shape=input_shape)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inp)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(out_channels, 1, padding='same', activation='sigmoid')(x)
    return Model(inp, x)

def attention_block(x):
    # --- Channel Attention ---
    channels = x.shape[-1]

    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)

    shared_mlp_1 = layers.Dense(channels // 8, activation='relu')
    shared_mlp_2 = layers.Dense(channels)

    out_avg = shared_mlp_2(shared_mlp_1(avg_pool))
    out_max = shared_mlp_2(shared_mlp_1(max_pool))

    channel_att = layers.Add()([out_avg, out_max])
    channel_att = layers.Activation('sigmoid')(channel_att)
    channel_att = layers.Reshape((1, 1, channels))(channel_att)

    x = layers.Multiply()([x, channel_att])

    # --- Spatial Attention ---
    # Use Keras Lambda layers instead of tf.reduce_mean/max
    avg_pool_sp = layers.Lambda(lambda z: tf.reduce_mean(z, axis=-1, keepdims=True))(x)
    max_pool_sp = layers.Lambda(lambda z: tf.reduce_max(z, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=-1)([avg_pool_sp, max_pool_sp])
    spatial_att = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(concat)

    x = layers.Multiply()([x, spatial_att])
    return x


# -----------------------
# BUILD MODEL
# -----------------------
encoder_rgb = build_encoder((IMG_SIZE[1], IMG_SIZE[0], 3))
encoder_ir = build_encoder((IMG_SIZE[1], IMG_SIZE[0], 1))
decoder = build_decoder((IMG_SIZE[1], IMG_SIZE[0], 64))

inp_hr = Input(shape=(IMG_SIZE[1], IMG_SIZE[0], 3))
inp_lr = Input(shape=(IMG_SIZE[1], IMG_SIZE[0], 3))
inp_ir = Input(shape=(IMG_SIZE[1], IMG_SIZE[0], 1))

f_hr = encoder_rgb(inp_hr)
f_lr = encoder_rgb(inp_lr)
f_ir = encoder_ir(inp_ir)

f_hr_att = attention_block(f_hr)
f_lr_att = attention_block(f_lr)
f_ir_att = attention_block(f_ir)

f_fused = layers.Add()([f_hr_att, f_lr_att, f_ir_att])
out = decoder(f_fused)

model = Model(inputs=[inp_hr, inp_lr, inp_ir], outputs=out)
opt = tf.keras.optimizers.Adam(LEARNING_RATE)
mse = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=opt, loss=mse)
model.summary()

# -----------------------
# TRAINING LOOP
# -----------------------
for epoch in range(1, EPOCHS+1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    epoch_loss = []
    for step, (hr, lr, ir) in enumerate(dataset):
        with tf.GradientTape() as tape:
            fused = model([hr, lr, ir], training=True)
            loss = mse(hr, fused)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss.append(tf.reduce_mean(loss).numpy())
        if step % 20 == 0:
            print(f"  Step {step}, Loss: {tf.reduce_mean(loss).numpy():.6f}")

    print(f"Epoch {epoch} mean loss: {np.mean(epoch_loss):.6f}")

    # Save first 3 fused images per epoch
    fused_pred = fused.numpy()
    for k in range(min(3, fused_pred.shape[0])):
        out_img = (fused_pred[k]*255.0).astype(np.uint8)
        fname = os.path.join(OUT_DIR, f"epoch{epoch:03d}_sample{k+1}.png")
        cv2.imwrite(fname, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

# -----------------------
# SAVE MODEL
# -----------------------
model.save(os.path.join(OUT_DIR, "deepfuse_attention_final.h5"))
print("Training finished. Model saved in:", OUT_DIR)
