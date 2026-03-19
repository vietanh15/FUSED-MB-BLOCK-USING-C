#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


// ================== Định nghĩa cấu hình ==================
#define INPUT_H 32
#define INPUT_W 32
#define INPUT_C 128   // Số kênh đầu vào là output của layer1
#define OUTPUT_F 64   // Số filter PWC (tùy bài toán, ví dụ giảm còn 64)
#define STRIDE 1

#define NUM_PES 4
#define DSP_PER_PE 16
#define DRAM_BUS_WIDTH_BITS 128
#define DRAM_BUS_WIDTH_BYTES (DRAM_BUS_WIDTH_BITS / 8)

#define IFM_SIZE (INPUT_H * INPUT_W * INPUT_C)
#define WEIGHTS_SIZE (OUTPUT_F * INPUT_C)
#define OFM_SIZE (INPUT_H * INPUT_W * OUTPUT_F)

#define IFM_BRAM_DEPTH (IFM_SIZE / DRAM_BUS_WIDTH_BYTES)
#define WEIGHT_BRAM_FILTER_SETS (OUTPUT_F / NUM_PES)
#define SINGLE_FILTER_CHUNKS (INPUT_C / DRAM_BUS_WIDTH_BYTES)
#define WEIGHT_BRAM_DEPTH (WEIGHT_BRAM_FILTER_SETS * SINGLE_FILTER_CHUNKS)
#define OFM_BRAM_DEPTH (OFM_SIZE / DRAM_BUS_WIDTH_BYTES)

// ================== Khai báo bộ nhớ ==================
int8_t dram[IFM_SIZE + WEIGHTS_SIZE];
int8_t ifm_bram[IFM_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe0[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe1[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe2[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe3[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];

void* weight_bram_pointers[NUM_PES] = {
    weight_bram_pe0, weight_bram_pe1, weight_bram_pe2, weight_bram_pe3
};

int8_t ofm_bram[OFM_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
// BRAM để lưu trữ Identity (dữ liệu từ ifm.txt) cho kết nối dư
// Kích thước là 32x32x64, giống hệt OFM, nên dùng chung OFM_BRAM_DEPTH
int8_t identity_bram[OFM_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int32_t pe_accumulators[NUM_PES];

// ================== Hàm chức năng ==================

/**
 * @brief Nạp dữ liệu từ một file text HWC thô (như ifm.txt) vào một mảng BRAM.
 *        Hàm này đọc dữ liệu theo thứ tự HWC và sắp xếp vào BRAM.
 * @param filename Tên file để đọc.
 * @param bram_ptr Con trỏ tới mảng BRAM đích.
 * @param h Height của tensor
 * @param w Width của tensor
 * @param c Channels của tensor
 */
void load_hwc_to_bram(const char* filename, int8_t (*bram_ptr)[DRAM_BUS_WIDTH_BYTES], int h, int w, int c) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Loi: Khong the mo file %s\n", filename);
        exit(1);
    }

    int total_elements = h * w * c;
    for (int i = 0; i < total_elements; i++) {
        int row = i / DRAM_BUS_WIDTH_BYTES;
        int col = i % DRAM_BUS_WIDTH_BYTES;
        if (fscanf(file, "%hhd", &bram_ptr[row][col]) != 1) {
            printf("Loi: Khong the doc du lieu tu file %s tai phan tu thu %d\n", filename, i);
            fclose(file);
            exit(1);
        }
    }
    fclose(file);
}

void load_file_to_dram(const char* filename, int8_t* dram_ptr, int size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Loi: Khong the mo file %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%hhd", &dram_ptr[i]) != 1) {
            printf("Loi: Khong the doc du lieu tu file %s\n", filename);
            fclose(file);
            exit(1);
        }
    }
    fclose(file);
}

void load_file_to_ifm_bram(const char* filename, int8_t (*bram_ptr)[DRAM_BUS_WIDTH_BYTES], int depth, int width) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Loi: Khong the mo file %s\n", filename);
        exit(1);
    }
    for (int row = 0; row < depth; row++) {
        for (int col = 0; col < width; col++) {
            if (fscanf(file, "%hhd", &bram_ptr[row][col]) != 1) {
                printf("Loi: Khong the doc du lieu tu file %s tai hang %d, cot %d\n", filename, row, col);
                fclose(file);
                exit(1);
            }
        }
    }
    fclose(file);
}

void load_bram(const int8_t* src_dram, int dram_offset, int size, int8_t (*dest_bram)[DRAM_BUS_WIDTH_BYTES], int bram_row) {
    memcpy(dest_bram[bram_row], &src_dram[dram_offset], size);
}

void compute_pe(int MAC, int addr_BRAM_IFM, int addr_BRAM_Weight, int index_pe, int valid, int32_t* data_output) {
    static const int8_t zero_buffer[DRAM_BUS_WIDTH_BYTES] = {0};
    const int8_t* ifm_data_ptr = (addr_BRAM_IFM == -1) ? zero_buffer : &ifm_bram[addr_BRAM_IFM][0];
    int8_t (*current_weight_bram)[DRAM_BUS_WIDTH_BYTES] = weight_bram_pointers[index_pe];
    const int8_t* weight_data_ptr = &current_weight_bram[addr_BRAM_Weight][0];
    int32_t partial_sum = 0;
    for (int dsp = 0; dsp < MAC; dsp++) {
        partial_sum += ifm_data_ptr[dsp] * weight_data_ptr[dsp];
    }
    *data_output = partial_sum;
}

/**
 * @brief Thực hiện phép cộng dư song song cho NUM_PES PE, mô phỏng băng thông 128-bit.
 *        Đọc các byte identity cần thiết, cộng với các giá trị tích lũy, và ghi kết quả.
 * @param accumulators Con trỏ tới mảng NUM_PES giá trị int32 từ các bộ tích lũy.
 * @param identity_bram_row Địa chỉ hàng trong BRAM của dữ liệu identity (đọc 16 byte).
 * @param dest_ofm_row Con trỏ tới hàng đích trong OFM BRAM để ghi kết quả (ghi 16 byte).
 */
void compute_parallel_residual_add(const int32_t* accumulators, int identity_bram_row, int8_t* dest_ofm_row) {
    // Đọc 1 dòng (16 byte) từ identity BRAM, mô phỏng băng thông 128-bit
    const int8_t* identity_data_ptr = identity_bram[identity_bram_row];

    // Vòng lặp mô phỏng NUM_PES bộ cộng vector hoạt động song song
    for (int pe = 0; pe < NUM_PES; pe++) {
        // 1. Lấy kết quả từ Accumulator của PE
        int32_t pwc_output = accumulators[pe];

        // 2. Lấy dữ liệu Input gốc (Identity) cho PE tương ứng
        int8_t identity_data = identity_data_ptr[pe];

        // 3. Thực hiện phép cộng (căn chỉnh)
        int32_t res_add_result = pwc_output + ((int32_t)identity_data << 8);

        // 4. Khối Post-Processing (Shift + Saturate)
        int32_t final_val = res_add_result >> 8;
        if (final_val > 127) final_val = 127;
        if (final_val < -128) final_val = -128;

        // 5. Ghi kết quả cuối cùng vào vị trí tương ứng trong hàng OFM đích
        dest_ofm_row[pe] = (int8_t)final_val;
    }
}

void write_ofm_bram_to_file(const char* filename, int8_t ofm_bram[OFM_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES]) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Loi: Khong the mo file %s de ghi OFM!\n", filename);
        exit(1);
    }
    for (int row = 0; row < OFM_BRAM_DEPTH; row++) {
        for (int col = 0; col < DRAM_BUS_WIDTH_BYTES; col++) {
            fprintf(file, "%d ", ofm_bram[row][col]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

// ================== Hàm main ==================
int main() {
    printf("Bat dau mo phong PWC Fused voi Residual Add...\n");

    // 1a. Nạp Input gốc (ifm.txt) vào identity_bram để thực hiện phép cộng dư
    // Kích thước là 32x32x64, khớp với OUTPUT_F
    load_hwc_to_bram("ifm.txt", identity_bram, INPUT_H, INPUT_W, OUTPUT_F);
    printf("Da nap Identity (ifm.txt) vao BRAM cho ket noi du.\n");

    // 1b. Nạp IFM (output của DWC) trực tiếp từ file vào BRAM
    load_file_to_ifm_bram("ofm_output.txt", ifm_bram, IFM_BRAM_DEPTH, DRAM_BUS_WIDTH_BYTES);
    printf("Da nap IFM (ofm_output.txt) vao BRAM.\n");

    // 2. Nạp Weight từ file vào DRAM (vẫn dùng vùng nhớ DRAM cho weight)
    load_file_to_dram("weights_pwc.txt", dram + IFM_SIZE, WEIGHTS_SIZE);

    // 3. Nạp IFM vào BRAM -> Đã thực hiện ở bước 1, không cần làm lại.

    // 4. Nạp Weight vào BRAM của từng PE theo kiểu interleaved
    for (int f_group = 0; f_group < WEIGHT_BRAM_FILTER_SETS; f_group++) {
        for (int chunk = 0; chunk < SINGLE_FILTER_CHUNKS; chunk++) {
            for (int pe = 0; pe < NUM_PES; pe++) {
                int filter_idx = f_group * NUM_PES + pe;
                int dram_offset = IFM_SIZE + filter_idx * INPUT_C + chunk * DRAM_BUS_WIDTH_BYTES;
                int bram_row = f_group * SINGLE_FILTER_CHUNKS + chunk;
                load_bram(dram, dram_offset, DRAM_BUS_WIDTH_BYTES, (int8_t (*)[DRAM_BUS_WIDTH_BYTES])weight_bram_pointers[pe], bram_row);
            }
        }
    }

    // 5. Tính toán PWC và lưu OFM vào BRAM
    int channel_chunks = INPUT_C / DSP_PER_PE;
    for (int oh = 0; oh < INPUT_H; oh++) {
        for (int ow = 0; ow < INPUT_W; ow++) {
            for (int f_group = 0; f_group < WEIGHT_BRAM_FILTER_SETS; f_group++) {
                memset(pe_accumulators, 0, sizeof(pe_accumulators));
                // PWC: Không còn vòng lặp kh, kw, chỉ còn lặp theo channel_chunks
                for (int c_chunk = 0; c_chunk < channel_chunks; c_chunk++) {
                    int ifm_bram_row = (oh * INPUT_W + ow) * channel_chunks + c_chunk;
                    int weight_bram_row = f_group * SINGLE_FILTER_CHUNKS + c_chunk;
                    for (int pe = 0; pe < NUM_PES; pe++) {
                        int32_t partial_sum_result;
                        compute_pe(DSP_PER_PE, ifm_bram_row, weight_bram_row, pe, channel_chunks, &partial_sum_result);
                        pe_accumulators[pe] += partial_sum_result;
                    }
                }
                // --- KHỐI MÔ PHỎNG "RESIDUAL ADDER UNIT" SONG SONG ---
                // Tính toán địa chỉ hàng BRAM chung cho NUM_PES kết quả PE
                int base_filter_idx = f_group * NUM_PES;
                int base_flat_idx = (oh * INPUT_W + ow) * OUTPUT_F + base_filter_idx;
                int bram_row = base_flat_idx / DRAM_BUS_WIDTH_BYTES;

                // Gọi hàm thực hiện cộng dư song song cho NUM_PES PE.
                // Hàm này đọc 1 dòng (16 byte) từ identity_bram và ghi 1 dòng vào ofm_bram.
                compute_parallel_residual_add(pe_accumulators, bram_row, ofm_bram[bram_row]);
            }
        }
    }

    printf("Hoan thanh! Ket qua PWC da luu vao BRAM.\n");

    // Xuất OFM ra file
    write_ofm_bram_to_file("ofm_pwc_residual.txt", ofm_bram);
    printf("Da xuat file ofm_pwc_residual.txt!\n");
    return 0;
}