import numpy as np
import tensorflow as tf
import os

# --- Cấu hình giống hệt layer2.c ---
INPUT_H = 32
INPUT_W = 32
INPUT_C = 128   # Số kênh đầu vào (từ output của layer1)
OUTPUT_F = 64   # Số filter của lớp PWC
STRIDE = 1
KERNEL_SIZE = 1 # Pointwise Convolution có kernel 1x1

# --- Đường dẫn tệp ---
ifm_path = 'c:\\Code c\\FUSED BLOCK\\ofm_output.txt'
weights_path = 'c:\\Code c\\FUSED BLOCK\\weights_pwc.txt'
output_golden_path = 'c:\\Code c\\FUSED BLOCK\\ofm_golden_pwc.txt'

print(f"Đọc IFM từ: {ifm_path}")
if not os.path.exists(ifm_path):
    raise FileNotFoundError(f"Tệp đầu vào không tồn tại: {ifm_path}")

print(f"Đọc Weights từ: {weights_path}")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Tệp trọng số không tồn tại: {weights_path}")

# --- 1. Đọc dữ liệu IFM (Output của layer1) ---
# layer2.c đọc dữ liệu theo cấu trúc HWC
ifm = np.loadtxt(ifm_path, dtype=np.int8)
ifm = ifm.reshape((INPUT_H, INPUT_W, INPUT_C))  # Reshape về HWC

# --- 2. Đọc dữ liệu Weights ---
# layer2.c nạp weight theo thứ tự (OUTPUT_F, INPUT_C)
weights = np.loadtxt(weights_path, dtype=np.int8)
weights = weights.reshape((OUTPUT_F, INPUT_C)) # Reshape về (F, C)

# --- 3. Chuẩn bị dữ liệu cho TensorFlow ---
# a. Chuyển weights về chiều của TensorFlow: (H, W, C_in, C_out)
# Từ (F, C) -> (C, F) -> (1, 1, C, F)
weights_tf = np.transpose(weights, (1, 0)) # (F, C) -> (C, F)
weights_tf = np.expand_dims(weights_tf, axis=0)
weights_tf = np.expand_dims(weights_tf, axis=0) # Shape cuối cùng: (1, 1, INPUT_C, OUTPUT_F)

# b. Chuẩn bị IFM cho TensorFlow: (N, H, W, C)
ifm_tf = np.expand_dims(ifm, axis=0)  # Shape: (1, H, W, C)

print(f"Kích thước IFM cho TensorFlow: {ifm_tf.shape}")
print(f"Kích thước Weights cho TensorFlow: {weights_tf.shape}")

# --- 4. Thực hiện phép nhân chập (Pointwise Convolution) ---
# Cast sang float32 để tính toán, giống như tích lũy trong int32 rồi mới chuẩn hóa
ofm = tf.nn.conv2d(
    input=tf.cast(ifm_tf, tf.float32),
    filters=tf.cast(weights_tf, tf.float32),
    strides=[1, STRIDE, STRIDE, 1],
    padding='SAME' # PWC không làm thay đổi kích thước
)

# --- 5. CHUẨN HÓA GIỐNG HỆT C CODE ---
ofm_np = ofm.numpy()[0]  # Lấy kết quả và bỏ chiều batch, shape: (H, W, F)

# a. Dịch phải 8 bit (tương đương chia cho 2^8 = 256)
# Phép dịch bit trên số có dấu trong C tương đương với phép chia sàn (floor division) trong Python
ofm_np = np.floor(ofm_np / 256).astype(np.int32)

# b. Giới hạn giá trị trong khoảng int8
ofm_np = np.clip(ofm_np, -128, 127).astype(np.int8)

print(f"Kích thước OFM sau khi tính toán: {ofm_np.shape}")

# --- 6. Xuất ra file để so sánh ---
# Dữ liệu được ghi theo thứ tự HWC, giống cách C code tính toán chỉ số
ofm_flat = ofm_np.flatten()
with open(output_golden_path, 'w') as f:
    for v in ofm_flat:
        f.write(f"{v}\n")

print(f"Thành công! Đã xuất file golden '{output_golden_path}' để so sánh.")