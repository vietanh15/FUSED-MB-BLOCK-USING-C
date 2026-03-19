#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>

// ================== Định nghĩa cấu hình (Đồng bộ với file tham chiếu) ==================
#define INPUT_H 32
#define INPUT_W 32
#define INPUT_C 64

#define OUTPUT_H 32
#define OUTPUT_W 32
#define OUTPUT_F 128

#define KERNEL_H 3
#define KERNEL_W 3
#define STRIDE 1
#define PADDING 1

#define NUM_PES 16
#define DSP_PER_PE 16
#define DRAM_BUS_WIDTH_BITS 128
#define DRAM_BUS_WIDTH_BYTES (DRAM_BUS_WIDTH_BITS / 8)

// --- Cấu hình cho Ping-Pong Buffer ---
#define IFM_BUFFER_IMG_ROWS 6 // Tổng số hàng ảnh trong BRAM (Ping: 3, Pong: 3)
#define IFM_CHUNKS_PER_IMG_ROW (INPUT_W * INPUT_C / DRAM_BUS_WIDTH_BYTES) // Số BRAM row cho 1 hàng ảnh
#define IFM_BRAM_TOTAL_DEPTH (IFM_BUFFER_IMG_ROWS * IFM_CHUNKS_PER_IMG_ROW) // Tổng độ sâu BRAM
 
#define IFM_SIZE (INPUT_H * INPUT_W * INPUT_C)
#define WEIGHTS_SIZE (OUTPUT_F * KERNEL_H * KERNEL_W * INPUT_C)
#define OFM_SIZE (OUTPUT_H * OUTPUT_W * OUTPUT_F)
#define WEIGHT_BRAM_FILTER_SETS (OUTPUT_F / NUM_PES)
#define SINGLE_FILTER_CHUNKS ((KERNEL_H * KERNEL_W * INPUT_C) / DRAM_BUS_WIDTH_BYTES)
#define WEIGHT_BRAM_DEPTH (WEIGHT_BRAM_FILTER_SETS * SINGLE_FILTER_CHUNKS)
#define OFM_BRAM_DEPTH (OFM_SIZE / DRAM_BUS_WIDTH_BYTES)

// ================== Khai báo bộ nhớ (Cấp phát tĩnh) ==================
int8_t dram[IFM_SIZE + WEIGHTS_SIZE];
int8_t ifm_bram[IFM_BRAM_TOTAL_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe0[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe1[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe2[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe3[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe4[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe5[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe6[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe7[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe8[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe9[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe10[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe11[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe12[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe13[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe14[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int8_t weight_bram_pe15[WEIGHT_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];

void* weight_bram_pointers[NUM_PES] = {
    weight_bram_pe0, weight_bram_pe1, weight_bram_pe2, weight_bram_pe3,
    weight_bram_pe4, weight_bram_pe5, weight_bram_pe6, weight_bram_pe7,
    weight_bram_pe8, weight_bram_pe9, weight_bram_pe10, weight_bram_pe11,
    weight_bram_pe12, weight_bram_pe13, weight_bram_pe14, weight_bram_pe15
};

int8_t ofm_bram[OFM_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
int32_t pe_accumulators[NUM_PES];

// ================== Hàm chức năng ==================
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

void load_bram(const int8_t* src_dram, int dram_offset, int size, int8_t (*dest_bram)[DRAM_BUS_WIDTH_BYTES], int bram_row) {
    memcpy(dest_bram[bram_row], &src_dram[dram_offset], size);
}

void compute_pe_flexible(int MAC, const int8_t* ifm_data_ptr, int addr_BRAM_Weight, int index_pe, int32_t* data_output) {
    int8_t (*current_weight_bram)[DRAM_BUS_WIDTH_BYTES] = weight_bram_pointers[index_pe];
    const int8_t* weight_data_ptr = &current_weight_bram[addr_BRAM_Weight][0];
    int32_t partial_sum = 0;
    for (int i = 0; i < MAC; i++) {
        partial_sum += ifm_data_ptr[i] * weight_data_ptr[i];
    }
    *data_output = partial_sum;
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

void load_bank(int bank_id, int start_ih, int8_t (*ifm_bram)[DRAM_BUS_WIDTH_BYTES], const int8_t* dram) {
    #define BANK_SIZE 3
    printf(">>> [LOAD]    Nap BANK %s voi du lieu tu hang %d den %d.\n", bank_id == 0 ? "PING" : "PONG", start_ih, start_ih + BANK_SIZE - 1);
    for (int i = 0; i < 3; i++) {
        int current_ih = start_ih + i;
        if (current_ih < INPUT_H) {
            printf("        [LOAD]    - Nap hang ih=%d\n", current_ih);
            int dram_start_offset = current_ih * INPUT_W * INPUT_C;
            int bram_start_row = (bank_id * 3 + i) * IFM_CHUNKS_PER_IMG_ROW;
            for (int chunk = 0; chunk < IFM_CHUNKS_PER_IMG_ROW; chunk++) {
                load_bram(dram, dram_start_offset + chunk * DRAM_BUS_WIDTH_BYTES, DRAM_BUS_WIDTH_BYTES, ifm_bram, bram_start_row + chunk);
            }
        }
    }
}

// ================== Hàm main ==================
int main() {
    static const int8_t zero_buffer[DRAM_BUS_WIDTH_BYTES] = {0};
    printf("CHUONG TRINH MO PHONG CONV LAYER 1 VOI PING-PONG (MUX + INTERLOCK)\n");
    printf("===================================================================\n");

    // 1. Nạp dữ liệu từ file vào DRAM
    load_file_to_dram("ifm.txt", dram, IFM_SIZE);
    load_file_to_dram("weights.txt", dram + IFM_SIZE, WEIGHTS_SIZE);

    // 2. Nạp weight vào BRAM của từng PE (chỉ làm 1 lần)
    for (int f_group = 0; f_group < WEIGHT_BRAM_FILTER_SETS; f_group++) {
        for (int chunk = 0; chunk < SINGLE_FILTER_CHUNKS; chunk++) {
            for (int pe = 0; pe < NUM_PES; pe++) {
                int filter_idx = f_group * NUM_PES + pe;
                int dram_offset = IFM_SIZE + filter_idx * (KERNEL_H * KERNEL_W * INPUT_C) + chunk * DRAM_BUS_WIDTH_BYTES;
                int bram_row = f_group * SINGLE_FILTER_CHUNKS + chunk;
                load_bram(dram, dram_offset, DRAM_BUS_WIDTH_BYTES, (int8_t (*)[DRAM_BUS_WIDTH_BYTES])weight_bram_pointers[pe], bram_row);
            }
        }
    }

    // 3. Khởi tạo các biến quản lý Ping-Pong
    #define PING_BANK 0
    #define PONG_BANK 1
    #define BANK_SIZE 3

    int bank_start_ih[2];
    bank_start_ih[PING_BANK] = -1;
    bank_start_ih[PONG_BANK] = -1;

    int active_bank = PING_BANK;
    int next_block_start_ih = BANK_SIZE;

    // Nạp bank đầu tiên (PING) để PE có thể bắt đầu ngay
    printf("\nKhoi dong: Nap truoc BANK PING.\n");
    load_bank(PING_BANK, 0, ifm_bram, dram);
    bank_start_ih[PING_BANK] = 0;
    
    // 4. Tính toán và lưu OFM với logic Ping-Pong và MUX+Interlock song song
    int channel_chunks = INPUT_C / DSP_PER_PE;
    printf("\nBat dau tinh toan tich chap song song...\n");

    int oh = 0;
    while (oh < OUTPUT_H) {
        int start_ih_for_active_bank = bank_start_ih[active_bank];
        int max_ih_in_active_bank = -1;
        if (start_ih_for_active_bank != -1) {
            int theoretical_max_ih = start_ih_for_active_bank + BANK_SIZE - 1;
            max_ih_in_active_bank = (theoretical_max_ih < INPUT_H) ? theoretical_max_ih : INPUT_H - 1;
        }

        int compute_end_oh = (max_ih_in_active_bank - (KERNEL_H - 1) + PADDING) / STRIDE;
        if (compute_end_oh >= OUTPUT_H) compute_end_oh = OUTPUT_H - 1;
        if (max_ih_in_active_bank == -1) compute_end_oh = OUTPUT_H - 1;
        if (compute_end_oh < oh) compute_end_oh = oh;

        int inactive_bank = 1 - active_bank;
        
        // --- MUX+Interlock: Khai báo biến cờ để khóa/mở khóa LOAD ---
        volatile bool compute_finished_using_old_bank = false;

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // --- TASK 1: COMPUTE (Đọc từ active_bank và inactive_bank nếu cần) ---
                if (oh <= compute_end_oh) {
                    printf(">>> [COMPUTE] Tinh toan tu oh=%d den oh=%d su dung BANK %s (+ doc bank cu via MUX)\n", oh, compute_end_oh, active_bank == PING_BANK ? "PING" : "PONG");
                }
                for (int current_oh = oh; current_oh <= compute_end_oh; current_oh++) {
                    printf("    [COMPUTE] Tinh toan cho oh=%d, can cac ih: %d, %d, %d\n", current_oh, current_oh - 1, current_oh, current_oh + 1);
                    for (int ow = 0; ow < OUTPUT_W; ow++) {
                        for (int f_group = 0; f_group < WEIGHT_BRAM_FILTER_SETS; f_group++) {
                            memset(pe_accumulators, 0, sizeof(pe_accumulators));
                            for (int kh = 0; kh < KERNEL_H; kh++) {
                                for (int kw = 0; kw < KERNEL_W; kw++) {
                                    for (int c_chunk = 0; c_chunk < channel_chunks; c_chunk++) {
                                        int ih = current_oh * STRIDE + kh - PADDING;
                                        int iw = ow * STRIDE + kw - PADDING;

                                        const int8_t* ifm_data_ptr = zero_buffer;

                                        if (ih >= 0 && ih < INPUT_H && iw >= 0 && iw < INPUT_W) {
                                            int target_bank = -1, target_slot = -1;
                                            
                                            // --- MUX LOGIC ---
                                            // 1. Tìm trong active_bank (ưu tiên)
                                            int start_ih_active = bank_start_ih[active_bank];
                                            if (start_ih_active != -1 && ih >= start_ih_active && ih < start_ih_active + BANK_SIZE) {
                                                target_bank = active_bank;
                                                target_slot = ih - start_ih_active;
                                            }
                                            
                                            // 2. Nếu không có, tìm trong inactive_bank (dữ liệu cũ, an toàn nhờ interlock)
                                            if(target_bank == -1) {
                                                int start_ih_inactive = bank_start_ih[inactive_bank];
                                                if (start_ih_inactive != -1 && ih >= start_ih_inactive && ih < start_ih_inactive + BANK_SIZE) {
                                                    target_bank = inactive_bank;
                                                    target_slot = ih - start_ih_inactive;
                                                }
                                            }

                                            if (target_bank != -1) {
                                                int bram_base_row_for_bank = target_bank * BANK_SIZE * IFM_CHUNKS_PER_IMG_ROW;
                                                int slot_start_bram_row = target_slot * IFM_CHUNKS_PER_IMG_ROW;
                                                int offset_in_row = (iw * INPUT_C + c_chunk * DSP_PER_PE) / DRAM_BUS_WIDTH_BYTES;
                                                ifm_data_ptr = &ifm_bram[bram_base_row_for_bank + slot_start_bram_row + offset_in_row][0];
                                            }
                                        }

                                        int weight_bram_row = f_group * SINGLE_FILTER_CHUNKS + (kh * KERNEL_W + kw) * channel_chunks + c_chunk;
                                        for (int pe = 0; pe < NUM_PES; pe++) {
                                            int32_t partial_sum_result;
                                            compute_pe_flexible(DSP_PER_PE, ifm_data_ptr, weight_bram_row, pe, &partial_sum_result);
                                            pe_accumulators[pe] += partial_sum_result;
                                        }
                                    }
                                }
                            }
                            for (int pe = 0; pe < NUM_PES; pe++) {
                                int current_filter_idx = f_group * NUM_PES + pe;
                                int output_flat_idx = (current_oh * OUTPUT_W + ow) * OUTPUT_F + current_filter_idx;
                                int bram_row = output_flat_idx / DRAM_BUS_WIDTH_BYTES;
                                int bram_offset = output_flat_idx % DRAM_BUS_WIDTH_BYTES;
                                int32_t shifted = pe_accumulators[pe] >> 8;
                                if (shifted > 127) shifted = 127;
                                if (shifted < -128) shifted = -128;
                                ofm_bram[bram_row][bram_offset] = (int8_t)shifted;
                            }
                        }
                    }
                }
                 // --- INTERLOCK: Mở khóa cho LOAD ---
                printf(">>> [COMPUTE] Hoan thanh, mo khoa cho LOAD.\n");
                compute_finished_using_old_bank = true;
            }

            #pragma omp section
            {
                // --- TASK 2: LOAD (Bị khóa cho đến khi COMPUTE xong) ---
                printf(">>> [LOAD]    Dang cho COMPUTE mo khoa...\n");
                while (!compute_finished_using_old_bank) {
                    // Busy-wait mô phỏng pipeline stall
                }
                printf(">>> [LOAD]    Da duoc mo khoa! Bat dau nap.\n");

                if (next_block_start_ih < INPUT_H) {
                    load_bank(inactive_bank, next_block_start_ih, ifm_bram, dram);
                }
            }
        }
        // --- KẾT THÚC XỬ LÝ SONG SONG ---

        if (next_block_start_ih < INPUT_H) {
            bank_start_ih[inactive_bank] = next_block_start_ih;
        }
        oh = compute_end_oh + 1;
        next_block_start_ih += BANK_SIZE;
        active_bank = 1 - active_bank;
        if (oh < OUTPUT_H) {
            printf("\n--- SWITCH! Chuan bi tinh toan tren BANK %s ---\n", active_bank == PING_BANK ? "PING" : "PONG");
        }
    }

    printf("Hoan thanh! Ket qua OFM da luu vao BRAM.\n");

    // Xuất OFM ra file
    write_ofm_bram_to_file("ofm_output_mux.txt", ofm_bram);
    printf("Da xuat file ofm_output_mux.txt!\n");
    return 0;
}