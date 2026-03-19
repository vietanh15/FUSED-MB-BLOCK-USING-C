#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>

// ================== Lớp 1: DWC (Từ layer1withpingpongmux.c) ==================
#define L1_INPUT_H 32
#define L1_INPUT_W 32
#define L1_INPUT_C 64
#define L1_OUTPUT_F 128
#define KERNEL_H 3
#define KERNEL_W 3
#define STRIDE 1
#define PADDING 1

// ================== Lớp 2: PWC + Residual (Từ layer2.c) ==================
#define L2_INPUT_C L1_OUTPUT_F // 128
#define L2_OUTPUT_F 64

// ================== Cấu hình chung ==================
#define L1_NUM_PES 16
#define L2_NUM_PES 4
#define DSP_PER_PE 16 // Giả định L1 và L2 dùng chung cấu hình này
#define DRAM_BUS_WIDTH_BITS 128
#define DRAM_BUS_WIDTH_BYTES (DRAM_BUS_WIDTH_BITS / 8)

// --- Cấu hình bộ nhớ L1 ---
#define L1_IFM_SIZE (L1_INPUT_H * L1_INPUT_W * L1_INPUT_C)
#define L1_WEIGHTS_SIZE (L1_OUTPUT_F * KERNEL_H * KERNEL_W * L1_INPUT_C)
#define L1_OFM_SIZE (L1_INPUT_H * L1_INPUT_W * L1_OUTPUT_F)
#define L1_IFM_CHUNKS_PER_IMG_ROW (L1_INPUT_W * L1_INPUT_C / DRAM_BUS_WIDTH_BYTES)
#define L1_IFM_BUFFER_IMG_ROWS 6
#define L1_IFM_BRAM_TOTAL_DEPTH (L1_IFM_BUFFER_IMG_ROWS * L1_IFM_CHUNKS_PER_IMG_ROW)
#define L1_WEIGHT_BRAM_FILTER_SETS (L1_OUTPUT_F / L1_NUM_PES)
#define L1_SINGLE_FILTER_CHUNKS ((KERNEL_H * KERNEL_W * L1_INPUT_C) / DRAM_BUS_WIDTH_BYTES)
#define L1_WEIGHT_BRAM_DEPTH (L1_WEIGHT_BRAM_FILTER_SETS * L1_SINGLE_FILTER_CHUNKS)
#define L1_OFM_BRAM_DEPTH (L1_OFM_SIZE / DRAM_BUS_WIDTH_BYTES)

// --- Cấu hình bộ nhớ L2 ---
#define L2_WEIGHTS_SIZE (L2_OUTPUT_F * L2_INPUT_C)
#define L2_OFM_SIZE (L1_INPUT_H * L1_INPUT_W * L2_OUTPUT_F)
#define L2_IDENTITY_SIZE (L1_INPUT_H * L1_INPUT_W * L2_OUTPUT_F) // Residual add có cùng kích thước OFM của L2
#define L2_WEIGHT_BRAM_FILTER_SETS (L2_OUTPUT_F / L2_NUM_PES)
#define L2_SINGLE_FILTER_CHUNKS (L2_INPUT_C / DRAM_BUS_WIDTH_BYTES)
#define L2_WEIGHT_BRAM_DEPTH (L2_WEIGHT_BRAM_FILTER_SETS * L2_SINGLE_FILTER_CHUNKS)
#define L2_OFM_BRAM_DEPTH (L2_OFM_SIZE / DRAM_BUS_WIDTH_BYTES)
#define L2_IDENTITY_BRAM_DEPTH (L2_IDENTITY_SIZE / DRAM_BUS_WIDTH_BYTES)

// ================== Khai báo bộ nhớ (Cấp phát tĩnh) ==================
// --- DRAM chung cho tất cả --- 
int8_t dram[L1_IFM_SIZE + L1_WEIGHTS_SIZE + L2_WEIGHTS_SIZE];

// --- Bộ nhớ Lớp 1 ---
int8_t l1_ifm_bram[L1_IFM_BRAM_TOTAL_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l1_ofm_bram[L1_OFM_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES]; // Output của L1, Input của L2
int8_t l1_weight_bram_pe0[L1_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l1_weight_bram_pe1[L1_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l1_weight_bram_pe2[L1_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l1_weight_bram_pe3[L1_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l1_weight_bram_pe4[L1_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l1_weight_bram_pe5[L1_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l1_weight_bram_pe6[L1_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l1_weight_bram_pe7[L1_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l1_weight_bram_pe8[L1_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l1_weight_bram_pe9[L1_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l1_weight_bram_pe10[L1_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l1_weight_bram_pe11[L1_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l1_weight_bram_pe12[L1_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l1_weight_bram_pe13[L1_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l1_weight_bram_pe14[L1_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l1_weight_bram_pe15[L1_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];

void* l1_weight_bram_pointers[L1_NUM_PES] = {
    l1_weight_bram_pe0, l1_weight_bram_pe1, l1_weight_bram_pe2, l1_weight_bram_pe3,
    l1_weight_bram_pe4, l1_weight_bram_pe5, l1_weight_bram_pe6, l1_weight_bram_pe7,
    l1_weight_bram_pe8, l1_weight_bram_pe9, l1_weight_bram_pe10, l1_weight_bram_pe11,
    l1_weight_bram_pe12, l1_weight_bram_pe13, l1_weight_bram_pe14, l1_weight_bram_pe15
};
int32_t l1_pe_accumulators[L1_NUM_PES];

// --- Bộ nhớ Lớp 2 ---
int8_t l2_identity_bram[L2_IDENTITY_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l2_ofm_bram[L2_OFM_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l2_weight_bram_pe0[L2_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l2_weight_bram_pe1[L2_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l2_weight_bram_pe2[L2_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t l2_weight_bram_pe3[L2_WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
// ... (Giả sử L2 chỉ dùng 4 PE cho đơn giản, nếu cần 16 thì phải khai báo đủ)
void* l2_weight_bram_pointers[L2_NUM_PES] = {
    l2_weight_bram_pe0, l2_weight_bram_pe1, l2_weight_bram_pe2, l2_weight_bram_pe3
};
int32_t l2_pe_accumulators[L2_NUM_PES];


// --- Biến đồng bộ hóa Pipeline ---
volatile int pixel_ready_for_l2[L1_INPUT_H][L1_INPUT_W];


// ================== Hàm chức năng chung ==================
void load_file_to_dram(const char* filename, int8_t* dram_ptr, int size) {
    FILE* file = fopen(filename, "r");
    if (!file) { printf("Loi: Khong the mo file %s\n", filename); exit(1); }
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%hhd", &dram_ptr[i]) != 1) {
            printf("Loi: Doc file %s that bai\n", filename); fclose(file); exit(1);
        }
    }
    fclose(file);
}

void load_bram(const int8_t* src_dram, int dram_offset, int size, int8_t (*dest_bram)[DRAM_BUS_WIDTH_BYTES], int bram_row) {
    memcpy(dest_bram[bram_row], &src_dram[dram_offset], size);
}

void write_ofm_bram_to_file(const char* filename, int8_t (*ofm_bram)[DRAM_BUS_WIDTH_BYTES], int depth) {
    FILE* file = fopen(filename, "w");
    if (!file) { printf("Loi: Khong the mo file %s de ghi\n", filename); exit(1); }
    for (int row = 0; row < depth; row++) {
        for (int col = 0; col < DRAM_BUS_WIDTH_BYTES; col++) {
            fprintf(file, "%d ", ofm_bram[row][col]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

// ================== Hàm chức năng Lớp 1 (DWC) ==================
void l1_load_bank(int bank_id, int start_ih, int8_t (*ifm_bram)[DRAM_BUS_WIDTH_BYTES], const int8_t* dram) {
    #define BANK_SIZE 3
    printf(">>> [L1-LOAD] Nap BANK %s voi du lieu tu hang %d den %d.\n", bank_id == 0 ? "PING" : "PONG", start_ih, start_ih + BANK_SIZE - 1);
    for (int i = 0; i < 3; i++) {
        int current_ih = start_ih + i;
        if (current_ih < L1_INPUT_H) {
            int dram_start_offset = current_ih * L1_INPUT_W * L1_INPUT_C;
            int bram_start_row = (bank_id * 3 + i) * L1_IFM_CHUNKS_PER_IMG_ROW;
            for (int chunk = 0; chunk < L1_IFM_CHUNKS_PER_IMG_ROW; chunk++) {
                load_bram(dram, dram_start_offset + chunk * DRAM_BUS_WIDTH_BYTES, DRAM_BUS_WIDTH_BYTES, ifm_bram, bram_start_row + chunk);
            }
        }
    }
}

void l1_compute_pe(int MAC, const int8_t* ifm_data_ptr, int addr_BRAM_Weight, int index_pe, int32_t* data_output) {
    int8_t (*current_weight_bram)[DRAM_BUS_WIDTH_BYTES] = l1_weight_bram_pointers[index_pe];
    const int8_t* weight_data_ptr = &current_weight_bram[addr_BRAM_Weight][0];
    int32_t partial_sum = 0;
    for (int i = 0; i < MAC; i++) {
        partial_sum += ifm_data_ptr[i] * weight_data_ptr[i];
    }
    *data_output = partial_sum;
}

// ================== Hàm chức năng Lớp 2 (PWC) ==================
void load_hwc_to_bram(const char* filename, int8_t (*bram_ptr)[DRAM_BUS_WIDTH_BYTES], int h, int w, int c) {
    FILE* file = fopen(filename, "r");
    if (!file) { printf("Loi: Khong the mo file %s\n", filename); exit(1); }
    int total_elements = h * w * c;
    for (int i = 0; i < total_elements; i++) {
        int row = i / DRAM_BUS_WIDTH_BYTES;
        int col = i % DRAM_BUS_WIDTH_BYTES;
        if (fscanf(file, "%hhd", &bram_ptr[row][col]) != 1) {
            printf("Loi: Doc file %s that bai\n", filename); fclose(file); exit(1);
        }
    }
    fclose(file);
}

void l2_compute_pe(int MAC, int addr_BRAM_IFM, int addr_BRAM_Weight, int index_pe, int32_t* data_output) {
    const int8_t* ifm_data_ptr = &l1_ofm_bram[addr_BRAM_IFM][0]; // L2 đọc từ OFM của L1
    int8_t (*current_weight_bram)[DRAM_BUS_WIDTH_BYTES] = l2_weight_bram_pointers[index_pe];
    const int8_t* weight_data_ptr = &current_weight_bram[addr_BRAM_Weight][0];
    int32_t partial_sum = 0;
    for (int dsp = 0; dsp < MAC; dsp++) {
        partial_sum += ifm_data_ptr[dsp] * weight_data_ptr[dsp];
    }
    *data_output = partial_sum;
}

void l2_compute_residual_add(const int32_t* accumulators, int identity_bram_row, int8_t* dest_ofm_row) {
    const int8_t* identity_data_ptr = l2_identity_bram[identity_bram_row];
    for (int pe = 0; pe < L2_NUM_PES; pe++) { // Giả sử L2 chỉ dùng 4 PE
        int32_t pwc_output = accumulators[pe];
        int8_t identity_data = identity_data_ptr[pe];
        int32_t res_add_result = pwc_output + ((int32_t)identity_data << 8);
        int32_t final_val = res_add_result >> 8;
        if (final_val > 127) final_val = 127;
        if (final_val < -128) final_val = -128;
        dest_ofm_row[pe] = (int8_t)final_val;
    }
}


// ================== Hàm main ==================
int main() {
    static const int8_t zero_buffer[DRAM_BUS_WIDTH_BYTES] = {0};
    printf("CHUONG TRINH MO PHONG PIPELINE L1(DWC) + L2(PWC)\n");
    printf("==================================================\n");

    // --- 1. NẠP DỮ LIỆU VÀO DRAM ---
    printf("Bat dau nap du lieu...\n");
    load_file_to_dram("ifm.txt", dram, L1_IFM_SIZE);
    load_file_to_dram("weights.txt", dram + L1_IFM_SIZE, L1_WEIGHTS_SIZE);
    load_file_to_dram("weights_pwc.txt", dram + L1_IFM_SIZE + L1_WEIGHTS_SIZE, L2_WEIGHTS_SIZE);
    printf("Da nap xong IFM, Weights L1, Weights L2 vao DRAM.\n");

    // --- 2. NẠP DỮ LIỆU CỐ ĐỊNH VÀO BRAM ---
    // Nạp weight L1 vào BRAM của từng PE
    for (int f_group = 0; f_group < L1_WEIGHT_BRAM_FILTER_SETS; f_group++) {
        for (int chunk = 0; chunk < L1_SINGLE_FILTER_CHUNKS; chunk++) {
            for (int pe = 0; pe < L1_NUM_PES; pe++) {
                int filter_idx = f_group * L1_NUM_PES + pe;
                int dram_offset = L1_IFM_SIZE + filter_idx * (KERNEL_H * KERNEL_W * L1_INPUT_C) + chunk * DRAM_BUS_WIDTH_BYTES;
                int bram_row = f_group * L1_SINGLE_FILTER_CHUNKS + chunk;
                load_bram(dram, dram_offset, DRAM_BUS_WIDTH_BYTES, (int8_t (*)[DRAM_BUS_WIDTH_BYTES])l1_weight_bram_pointers[pe], bram_row);
            }
        }
    }
    printf("Da nap weights L1 vao BRAM.\n");

    // Nạp weight L2 vào BRAM của từng PE
    for (int f_group = 0; f_group < L2_WEIGHT_BRAM_FILTER_SETS; f_group++) {
        for (int chunk = 0; chunk < L2_SINGLE_FILTER_CHUNKS; chunk++) {
            for (int pe = 0; pe < L2_NUM_PES; pe++) { // Giả sử L2 chỉ dùng 4 PE
                int filter_idx = f_group * L2_NUM_PES + pe;
                int dram_offset = L1_IFM_SIZE + L1_WEIGHTS_SIZE + filter_idx * L2_INPUT_C + chunk * DRAM_BUS_WIDTH_BYTES;
                int bram_row = f_group * L2_SINGLE_FILTER_CHUNKS + chunk;
                load_bram(dram, dram_offset, DRAM_BUS_WIDTH_BYTES, (int8_t (*)[DRAM_BUS_WIDTH_BYTES])l2_weight_bram_pointers[pe], bram_row);
            }
        }
    }
    printf("Da nap weights L2 vao BRAM.\n");
    
    // Nạp dữ liệu cho kết nối dư (Residual)
    load_hwc_to_bram("ifm.txt", l2_identity_bram, L1_INPUT_H, L1_INPUT_W, L2_OUTPUT_F);
    printf("Da nap Identity (ifm.txt) cho L2 vao BRAM.\n");

    // Khởi tạo mảng cờ đồng bộ
    memset((void*)pixel_ready_for_l2, 0, sizeof(pixel_ready_for_l2));

    printf("\nBat dau tinh toan pipeline...\n");

    #pragma omp parallel sections
    {
        // ======================= SECTION 1: PRODUCER (Lớp 1 - DWC) =======================
        #pragma omp section
        {
            // --- Logic Ping-Pong của Lớp 1 ---
            #define PING_BANK 0
            #define PONG_BANK 1
            #define BANK_SIZE 3
            int bank_start_ih[2] = {-1, -1};
            int active_bank = PING_BANK;
            int next_block_start_ih = BANK_SIZE;

            printf("\n[L1] Khoi dong: Nap truoc BANK PING.\n");
            l1_load_bank(PING_BANK, 0, l1_ifm_bram, dram);
            bank_start_ih[PING_BANK] = 0;
            
            int oh = 0;
            while (oh < L1_INPUT_H) {
                int start_ih_for_active_bank = bank_start_ih[active_bank];
                int max_ih_in_active_bank = (start_ih_for_active_bank != -1) ? (start_ih_for_active_bank + BANK_SIZE - 1) : -1;
                if (max_ih_in_active_bank >= L1_INPUT_H) max_ih_in_active_bank = L1_INPUT_H - 1;

                int compute_end_oh = (max_ih_in_active_bank - (KERNEL_H - 1) + PADDING) / STRIDE;
                if (compute_end_oh >= L1_INPUT_H) compute_end_oh = L1_INPUT_H - 1;
                if (max_ih_in_active_bank == -1) compute_end_oh = L1_INPUT_H - 1;
                if (compute_end_oh < oh) compute_end_oh = oh;

                int inactive_bank = 1 - active_bank;
                volatile bool compute_finished_using_old_bank = false;

                #pragma omp parallel sections
                {
                    #pragma omp section
                    {
                        if (oh <= compute_end_oh) {
                            printf(">>> [L1-COMPUTE] Tinh toan tu oh=%d den oh=%d su dung BANK %s\n", oh, compute_end_oh, active_bank == PING_BANK ? "PING" : "PONG");
                        }
                        for (int current_oh = oh; current_oh <= compute_end_oh; current_oh++) {
                            for (int ow = 0; ow < L1_INPUT_W; ow++) {
                                // --- Tính toán 1 pixel (oh, ow) cho L1 ---
                                for (int f_group = 0; f_group < L1_WEIGHT_BRAM_FILTER_SETS; f_group++) {
                                    memset(l1_pe_accumulators, 0, sizeof(l1_pe_accumulators));
                                    for (int kh = 0; kh < KERNEL_H; kh++) {
                                        for (int kw = 0; kw < KERNEL_W; kw++) {
                                            int c_chunks = L1_INPUT_C / DSP_PER_PE;
                                            for (int c_chunk = 0; c_chunk < c_chunks; c_chunk++) {
                                                int ih = current_oh * STRIDE + kh - PADDING;
                                                int iw = ow * STRIDE + kw - PADDING;
                                                const int8_t* ifm_data_ptr = zero_buffer;
                                                if (ih >= 0 && ih < L1_INPUT_H && iw >= 0 && iw < L1_INPUT_W) {
                                                    int target_bank = -1, target_slot = -1;
                                                    int start_ih_active = bank_start_ih[active_bank];
                                                    if (start_ih_active != -1 && ih >= start_ih_active && ih < start_ih_active + BANK_SIZE) {
                                                        target_bank = active_bank;
                                                        target_slot = ih - start_ih_active;
                                                    } else {
                                                        int start_ih_inactive = bank_start_ih[inactive_bank];
                                                        if (start_ih_inactive != -1 && ih >= start_ih_inactive && ih < start_ih_inactive + BANK_SIZE) {
                                                            target_bank = inactive_bank;
                                                            target_slot = ih - start_ih_inactive;
                                                        }
                                                    }
                                                    if (target_bank != -1) {
                                                        int bram_base = target_bank * BANK_SIZE * L1_IFM_CHUNKS_PER_IMG_ROW;
                                                        int slot_start = target_slot * L1_IFM_CHUNKS_PER_IMG_ROW;
                                                        int offset = (iw * L1_INPUT_C + c_chunk * DSP_PER_PE) / DRAM_BUS_WIDTH_BYTES;
                                                        ifm_data_ptr = &l1_ifm_bram[bram_base + slot_start + offset][0];
                                                    }
                                                }
                                                int weight_bram_row = f_group * L1_SINGLE_FILTER_CHUNKS + (kh * KERNEL_W + kw) * c_chunks + c_chunk;
                                                for (int pe = 0; pe < L1_NUM_PES; pe++) {
                                                    int32_t partial_sum;
                                                    l1_compute_pe(DSP_PER_PE, ifm_data_ptr, weight_bram_row, pe, &partial_sum);
                                                    l1_pe_accumulators[pe] += partial_sum;
                                                }
                                            }
                                        }
                                    }
                                    // Ghi kết quả L1 vào l1_ofm_bram
                                    for (int pe = 0; pe < L1_NUM_PES; pe++) {
                                        int filter_idx = f_group * L1_NUM_PES + pe;
                                        int flat_idx = (current_oh * L1_INPUT_W + ow) * L1_OUTPUT_F + filter_idx;
                                        int bram_row = flat_idx / DRAM_BUS_WIDTH_BYTES;
                                        int bram_offset = flat_idx % DRAM_BUS_WIDTH_BYTES;
                                        int32_t shifted = l1_pe_accumulators[pe] >> 8;
                                        if (shifted > 127) shifted = 127;
                                        if (shifted < -128) shifted = -128;
                                        l1_ofm_bram[bram_row][bram_offset] = (int8_t)shifted;
                                    }
                                }
                                // --- ĐÁNH DẤU PIXEL SẴN SÀNG CHO L2 ---
                                printf("    [L1-COMPUTE] Hoan thanh pixel (%d, %d). Bao hieu cho L2.\n", current_oh, ow);
                                pixel_ready_for_l2[current_oh][ow] = 1; 
                            }
                        }
                        compute_finished_using_old_bank = true;
                    }
                    #pragma omp section
                    {
                        while (!compute_finished_using_old_bank) {} // Busy-wait
                        if (next_block_start_ih < L1_INPUT_H) {
                            l1_load_bank(inactive_bank, next_block_start_ih, l1_ifm_bram, dram);
                        }
                    }
                }
                if (next_block_start_ih < L1_INPUT_H) {
                    bank_start_ih[inactive_bank] = next_block_start_ih;
                }
                oh = compute_end_oh + 1;
                next_block_start_ih += BANK_SIZE;
                active_bank = 1 - active_bank;
            }
            printf("[L1] Hoan thanh toan bo tinh toan.\n");
        }

        // ======================= SECTION 2: CONSUMER (Lớp 2 - PWC) =======================
        #pragma omp section
        {
            printf("[L2] Bat dau cho du lieu tu L1...\n");
            int pixels_processed = 0;
            while (pixels_processed < L1_INPUT_H * L1_INPUT_W) {
                for (int oh = 0; oh < L1_INPUT_H; oh++) {
                    for (int ow = 0; ow < L1_INPUT_W; ow++) {
                        // --- KIỂM TRA CỜ ĐỒNG BỘ ---
                        if (pixel_ready_for_l2[oh][ow] == 1) {
                            printf("    [L2-COMPUTE] Nhan duoc pixel (%d, %d). Bat dau tinh toan PWC.\n", oh, ow);
                            
                            // --- Tính toán PWC cho pixel (oh, ow) ---
                            int c_chunks = L2_INPUT_C / DSP_PER_PE;
                            for (int f_group = 0; f_group < L2_WEIGHT_BRAM_FILTER_SETS; f_group++) {
                                memset(l2_pe_accumulators, 0, sizeof(l2_pe_accumulators));
                                for (int c_chunk = 0; c_chunk < c_chunks; c_chunk++) {
                                    int ifm_bram_row = (oh * L1_INPUT_W + ow) * c_chunks + c_chunk;
                                    int weight_bram_row = f_group * L2_SINGLE_FILTER_CHUNKS + c_chunk;
                                    for (int pe = 0; pe < L2_NUM_PES; pe++) { // Giả sử L2 dùng 4 PE
                                        int32_t partial_sum;
                                        l2_compute_pe(DSP_PER_PE, ifm_bram_row, weight_bram_row, pe, &partial_sum);
                                        l2_pe_accumulators[pe] += partial_sum;
                                    }
                                }
                                // --- Cộng dư và ghi kết quả cuối cùng ---
                                int base_filter_idx = f_group * L2_NUM_PES;
                                int base_flat_idx = (oh * L1_INPUT_W + ow) * L2_OUTPUT_F + base_filter_idx;
                                int bram_row = base_flat_idx / DRAM_BUS_WIDTH_BYTES;
                                l2_compute_residual_add(l2_pe_accumulators, bram_row, l2_ofm_bram[bram_row]);
                            }
                            
                            // Đánh dấu đã xử lý xong để không chạy lại
                            pixel_ready_for_l2[oh][ow] = 2; 
                            pixels_processed++;
                        }
                    }
                }
            }
            printf("[L2] Hoan thanh toan bo tinh toan.\n");
        }
    }

    printf("\nHoan thanh toan bo pipeline!\n");

    // Xuất OFM cuối cùng ra file
    write_ofm_bram_to_file("ofm_pipeline_final.txt", l2_ofm_bram, L2_OFM_BRAM_DEPTH);
    printf("Da xuat file ofm_pipeline_final.txt!\n");
    return 0;
}