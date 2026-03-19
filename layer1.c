#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ================== Định nghĩa cấu hình ==================
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

#define IFM_SIZE (INPUT_H * INPUT_W * INPUT_C)
#define WEIGHTS_SIZE (OUTPUT_F * KERNEL_H * KERNEL_W * INPUT_C)
#define OFM_SIZE (OUTPUT_H * OUTPUT_W * OUTPUT_F)

#define IFM_BRAM_DEPTH (IFM_SIZE / DRAM_BUS_WIDTH_BYTES)
#define WEIGHT_BRAM_FILTER_SETS (OUTPUT_F / NUM_PES)
#define SINGLE_FILTER_CHUNKS ((KERNEL_H * KERNEL_W * INPUT_C) / DRAM_BUS_WIDTH_BYTES)
#define WEIGHT_BRAM_DEPTH (WEIGHT_BRAM_FILTER_SETS * SINGLE_FILTER_CHUNKS)
#define OFM_BRAM_DEPTH (OFM_SIZE / DRAM_BUS_WIDTH_BYTES)

// ================== Khai báo bộ nhớ ==================
int8_t dram[IFM_SIZE + WEIGHTS_SIZE];
int8_t ifm_bram[IFM_BRAM_DEPTH][DRAM_BUS_WIDTH_BYTES];
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

void compute_pe(int MAC, int addr_BRAM_IFM, int addr_BRAM_Weight, int index_pe, int valid, int32_t* data_output) {
    static const int8_t zero_buffer[DRAM_BUS_WIDTH_BYTES] = {0};
    const int8_t* ifm_data_ptr = (addr_BRAM_IFM == -1) ? zero_buffer : &ifm_bram[addr_BRAM_IFM][0];
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

// ================== Hàm main ==================
int main() {
    printf("Bat dau mo phong...\n");

    // 1. Nạp dữ liệu từ file vào DRAM
    load_file_to_dram("ifm.txt", dram, IFM_SIZE);
    load_file_to_dram("weights.txt", dram + IFM_SIZE, WEIGHTS_SIZE);

    // 2. Nạp IFM vào BRAM
    int ifm_chunks = IFM_SIZE / DRAM_BUS_WIDTH_BYTES;
    for (int i = 0; i < ifm_chunks; i++) {
        load_bram(dram, i * DRAM_BUS_WIDTH_BYTES, DRAM_BUS_WIDTH_BYTES, ifm_bram, i);
    }

    // 3. Nạp weight vào BRAM của từng PE theo kiểu interleaved
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

    // 4. Tính toán và lưu OFM vào BRAM
    int channel_chunks = INPUT_C / DSP_PER_PE;
    for (int oh = 0; oh < OUTPUT_H; oh++) {
        for (int ow = 0; ow < OUTPUT_W; ow++) {
            for (int f_group = 0; f_group < WEIGHT_BRAM_FILTER_SETS; f_group++) {
                memset(pe_accumulators, 0, sizeof(pe_accumulators));
                for (int kh = 0; kh < KERNEL_H; kh++) {
                    for (int kw = 0; kw < KERNEL_W; kw++) {
                        for (int c_chunk = 0; c_chunk < channel_chunks; c_chunk++) {
                            int ih = oh * STRIDE + kh - PADDING;
                            int iw = ow * STRIDE + kw - PADDING;
                            int ifm_bram_row = (ih < 0 || ih >= INPUT_H || iw < 0 || iw >= INPUT_W) ? -1 : (ih * INPUT_W + iw) * channel_chunks + c_chunk;
                            int weight_bram_row = f_group * SINGLE_FILTER_CHUNKS + (kh * KERNEL_W + kw) * channel_chunks + c_chunk;
                            for (int pe = 0; pe < NUM_PES; pe++) {
                                int32_t partial_sum_result;
                                compute_pe(DSP_PER_PE, ifm_bram_row, weight_bram_row, pe, channel_chunks, &partial_sum_result);
                                pe_accumulators[pe] += partial_sum_result;
                            }
                        }
                    }
                }
                // Ghi kết quả vào BRAM OFM theo thứ tự HWC, chuẩn hóa bằng phép dịch bit
                for (int pe = 0; pe < NUM_PES; pe++) {
                    int current_filter_idx = f_group * NUM_PES + pe;
                    int output_flat_idx = (oh * OUTPUT_W + ow) * OUTPUT_F + current_filter_idx;
                    int bram_row = output_flat_idx / DRAM_BUS_WIDTH_BYTES;
                    int bram_offset = output_flat_idx % DRAM_BUS_WIDTH_BYTES;
                    int32_t shifted = pe_accumulators[pe] >> 8; // dịch phải 8 bit
                    if (shifted > 127) shifted = 127;
                    if (shifted < -128) shifted = -128;
                    ofm_bram[bram_row][bram_offset] = (int8_t)shifted;
                }
            }
        }
    }

    printf("Hoan thanh! Ket qua OFM da luu vao BRAM.\n");

    // Xuất OFM ra file
    write_ofm_bram_to_file("ofm_output.txt", ofm_bram);
    printf("Da xuat file ofm_output.txt!\n");
    return 0;
}
