import numpy as np
import tensorflow as tf

# --- Cấu hình ---
INPUT_H = 32
INPUT_W = 32
INPUT_C = 64
OUTPUT_F = 128
KERNEL_H = 3
KERNEL_W = 3
STRIDE = 1
PADDING = 1

# --- Đọc dữ liệu IFM ---
ifm = np.loadtxt('c:\\Code c\\Learning\\ifm.txt', dtype=np.int8)
ifm = ifm.reshape((INPUT_H, INPUT_W, INPUT_C))  # HWC

# --- Đọc dữ liệu Weights ---
weights = np.loadtxt('c:\\Code c\\Learning\\weights.txt', dtype=np.int8)
weights = weights.reshape((OUTPUT_F, KERNEL_H, KERNEL_W, INPUT_C))  # OIHW

# --- Chuyển weights về chiều TensorFlow: HWIO ---
weights_tf = np.transpose(weights, (1, 2, 3, 0))  # OIHW -> HWIO

# --- Chuẩn bị input cho TensorFlow ---
ifm_tf = np.expand_dims(ifm, axis=0)  # shape: (1, H, W, C)

# --- Thực hiện phép nhân chập ---
ofm = tf.nn.conv2d(
    input=tf.cast(ifm_tf, tf.float32),
    filters=tf.cast(weights_tf, tf.float32),
    strides=[1, STRIDE, STRIDE, 1],
    padding='SAME'
)

# --- CHUẨN HÓA GIỐNG C CODE ---
ofm_np = ofm.numpy()[0]  # shape: (OUTPUT_H, OUTPUT_W, OUTPUT_F)
ofm_np = np.round(ofm_np).astype(np.int32)         # Làm tròn về số nguyên
ofm_np = np.right_shift(ofm_np, 8)                # Dịch phải 8 bit
ofm_np = np.clip(ofm_np, -128, 127).astype(np.int8)  # Giới hạn về int8

# --- Xuất ra file để so sánh ---
ofm_flat = ofm_np.reshape(-1)
with open('c:\\Code c\\Learning\\ofm_golden.txt', 'w') as f:
    for v in ofm_flat:
        f.write(f"{v}\n")

print("Đã xuất file ofm_golden.txt để so sánh với ofm_output.txt!")