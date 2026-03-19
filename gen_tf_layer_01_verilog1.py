import numpy as np
import tensorflow as tf
import argparse

# --- CÁC HÀM ĐỌC/GHI FILE TỪ HEX ---

def read_int_array_file(filename, dtype=np.int32):
    with open(filename, "r") as file:
        lines = file.readlines()
    data = []
    for line in lines:
        s = line.strip()
        if s:
            # Đọc chuỗi Hex thành số Unsigned, sau đó ép về Signed bằng Mask
            val = int(s.split()[-1], 16)
            if dtype == np.int8:
                if val > 0x7F:
                    val -= 0x100
            elif dtype == np.int32:
                if val > 0x7FFFFFFF:
                    val -= 0x100000000
            data.append(val)
    return np.array(data, dtype=dtype)

def read_zp_file(filename):
    try:
        with open(filename, "r") as file:
            content = file.read().strip()
            # Lấy chuỗi cuối phòng trường hợp có text đi kèm
            val_str = content.split()[-1]
            
        val = int(val_str, 16)
        if val > 0x7F:
            val -= 0x100
            
        return np.int8(val)
    except Exception as e:
        print(f"Cảnh báo: Không đọc được file {filename}, dùng ZP=0. Lỗi: {e}")
        return np.int8(0)

def read_hex_file_weight(filename, shape):
    with open(filename, "r") as file:
        lines = file.readlines()    
    
    raw_data = []
    for x in lines:
        val = int(x.strip(), 16)
        if val > 0x7F:
            val -= 0x100
        raw_data.append(val)
        
    data = np.array(raw_data, dtype=np.int8)

    H, W, C, F = shape
    reshaped_data = np.zeros((H, W, C, F), dtype=np.int8)
    index = 0
    for f in range(F):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    reshaped_data[h, w, c, f] = data[index]
                    index += 1
    return reshaped_data

def read_hex_file(filename, shape):
    with open(filename, "r") as file:
        lines = file.readlines()    
    
    raw_data = []
    for x in lines:
        val = int(x.strip(), 16)
        if val > 0x7F:
            val -= 0x100
        raw_data.append(val)
        
    data = np.array(raw_data, dtype=np.int8)
    
    H, W, C = shape
    reshaped_data = np.zeros((H, W, C), dtype=np.int8)
    index = 0
    for h in range(H):
        for w in range(W):
            for c in range(C):
                reshaped_data[h, w, c] = data[index]
                index += 1
    return reshaped_data

def write_hex_file(filename, data):
    H, W, C = data.shape
    with open(filename, "w") as file:
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    int_value = int(round(data[h, w, c]))
                    # Format dưới dạng Hex 2 ký tự (int8) mask bù 2
                    hex_value = int_value & 0xFF 
                    file.write(f"{hex_value:02X}\n")

# --- CÁC HÀM XỬ LÝ SCALE ---

def SaturatingRoundingDoublingHighMul(a, b):
    a_64 = int(a)
    b_64 = int(b)
    ab_64 = a_64 * b_64
    
    nudge = 1 << 30
    result = (ab_64 + nudge) >> 31
    return result

def RoundingRightShift(x, shift):
    if shift <= 0:
        return x
        
    nudge = 1 << (shift - 1)
    return (x + nudge) >> shift

def MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift):
    left_shift = shift if shift > 0 else 0
    right_shift = -shift if shift < 0 else 0
    
    x_shifted = x * (1 << left_shift)
    high_mul = SaturatingRoundingDoublingHighMul(x_shifted, quantized_multiplier)
    result = RoundingRightShift(high_mul, right_shift)
    
    return result

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifm_height", type=int, required=True)
    parser.add_argument("--ifm_width", type=int, required=True)
    parser.add_argument("--ifm_channel", type=int, required=True)
    parser.add_argument("--weight_filter", type=int, required=True)
    parser.add_argument("--padding1", type=int, default=1)  # Padding P
    parser.add_argument("--stride1", type=int, default=1)   # Stride S
    args = parser.parse_args()

    # 1. TÍNH TOÁN KÍCH THƯỚC OFM
    output_feature_height = (args.ifm_height - 3 + 2 * args.padding1) // args.stride1 + 1
    output_feature_width = (args.ifm_width - 3 + 2 * args.padding1) // args.stride1 + 1
    output_feature_channel = args.weight_filter

    # 2. KHAI BÁO CÁC ĐƯỜNG DẪN FILE ĐẦU VÀO VÀ ĐẦU RA TỪ FOLDER HEX
    input_file = "golden_verilog_hex/op004_CONV_2D_ifm_values.hex"
    zp_in_file = "golden_verilog_hex/op004_CONV_2D_ifm_zp.hex" 
    
    weight_file = "golden_verilog_hex/op004_CONV_2D_weight_values.hex"
    eff_bias_file = "golden_verilog_hex/op004_CONV_2D_effective_bias.hex"
    
    m_file = "golden_verilog_hex/op004_CONV_2D_multiplier.hex"
    n_file = "golden_verilog_hex/op004_CONV_2D_shift.hex"
    
    zp_out_file = "golden_verilog_hex/op004_CONV_2D_ofm_zp.hex"
    output_file = "golden_verilog_hex/op004_CONV_2D_ofm_verilog.hex"

    # 3. ĐỌC DỮ LIỆU TỪ CÁC FILE ĐẦU VÀO
    # Đọc input map & zero point
    input_data = read_hex_file(input_file, (args.ifm_height, args.ifm_width, args.ifm_channel))
    zp_in = read_zp_file(zp_in_file)
    print(f"Info: ZP input = {zp_in}")

    # Đọc trọng số
    weight_data_flat = read_hex_file_weight(weight_file, (3, 3, args.ifm_channel, args.weight_filter))
    weight_data = weight_data_flat.reshape(3, 3, args.ifm_channel, args.weight_filter)
    
    # Đọc effective bias từ file HEX mới
    with open(eff_bias_file, "r") as f_bias:
        raw_bias = []
        for line in f_bias:
            s_val = line.strip()
            if not s_val:
                continue
            
            # Giải mã Hex thành int32 (Cấu trúc tương tự)
            val = int(s_val, 16)
            if val > 0x7FFFFFFF:
                val -= 0x100000000
            raw_bias.append(val)
            
        effective_bias = np.array(raw_bias, dtype=np.int32)

    # Đọc hệ số Requantize (Multiplier & Shift) & Output Zero Point
    m_from_file = read_int_array_file(m_file, dtype=np.int32)
    n_from_file = read_int_array_file(n_file, dtype=np.int8)
    zp_ofm = read_zp_file(zp_out_file)
    print(f"Info: OFM ZP: {zp_ofm}")

    # 4. CHUẨN BỊ PADDING CHO IFM
    if args.padding1 > 0:
        pad_h_total = max((output_feature_height - 1) * args.stride1 + 3 - args.ifm_height, 0)
        pad_w_total = max((output_feature_width - 1) * args.stride1 + 3 - args.ifm_width, 0)
        
        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left
        
        print(f"Padding info: Top={pad_top}, Bottom={pad_bottom}, Left={pad_left}, Right={pad_right}")
        
        input_data_padded = np.pad(
            input_data,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=zp_in
        )
    else:
        input_data_padded = input_data

    padded_height, padded_width, _ = input_data_padded.shape

    # 5. KHỞI TẠO VÀ CHẠY MÔ HÌNH TENSORFLOW ĐỂ TÍNH TÍCH CHẬP THÔ
    input_layer = tf.keras.layers.Input(shape=(padded_height, padded_width, args.ifm_channel))
    conv_layer = tf.keras.layers.Conv2D(filters=args.weight_filter,
                                        kernel_size=(3, 3),
                                        strides=(args.stride1, args.stride1),
                                        padding='valid', 
                                        activation=None)(input_layer)
    
    model = tf.keras.Model(inputs=input_layer, outputs=conv_layer)
    model.layers[1].set_weights([weight_data.astype(np.float32), effective_bias.astype(np.float32)])

    output_data = model.predict(input_data_padded.reshape(1, padded_height, padded_width, args.ifm_channel).astype(np.float32))
    output_data = output_data.reshape(output_feature_height, output_feature_width, output_feature_channel)
    
    # 6. THỰC HIỆN QUÁ TRÌNH REQUANTIZE
    H_out, W_out, C_out = output_data.shape
    output_final = np.zeros((H_out, W_out, C_out), dtype=np.int8)
    
    for f in range(C_out):
        multiplier_m = int(m_from_file[f])
        shift_n = int(n_from_file[f])

        if f == 0:
            print(f"Debug Filter 0: M={multiplier_m}, n={shift_n}")

        acc_data = output_data[:, :, f].flatten()
        res_scaled_flat = np.zeros_like(acc_data)
        
        for i in range(len(acc_data)):
             res_scaled_flat[i] = MultiplyByQuantizedMultiplier(acc_data[i], multiplier_m, shift_n)
             
        res_scaled = res_scaled_flat.reshape(H_out, W_out)    

        res_final = res_scaled + zp_ofm
        res_final = np.clip(res_final, -128, 127)
        
        output_final[:, :, f] = res_final.astype(np.int8)

    # 7. GHI KẾT QUẢ RA FILE HEX
    write_hex_file(output_file, output_final)
    print(f"Kết quả đã được ghi vào {output_file}")