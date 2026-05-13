#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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
    printf("========== LAYER1.C - SEQUENTIAL IMPLEMENTATION ==========\n");
    printf("\n");

    // =============== THÔNG SỐ LÝ THUYẾT ===============
    int theoretical_load_ifm_rows = IFM_SIZE / DRAM_BUS_WIDTH_BYTES;
    int theoretical_weight_per_pe = SINGLE_FILTER_CHUNKS * WEIGHT_BRAM_FILTER_SETS;
    int theoretical_load_cycles = theoretical_load_ifm_rows + (theoretical_weight_per_pe * NUM_PES);
    // Khi 16 PE chạy song song: (32×32×8 outer loops) × (3×3×4 inner loops) = 294,912 chu kì
    // Công thức: 32*32*8 * 3*3*4 = 294,912 (PE không đếm tuần tự nữa)
    int theoretical_compute_cycles_sequential = 4718592; // Nếu PE chạy tuần tự
    int theoretical_compute_cycles_parallel = 294912;    // Nếu 16 PE chạy song song (4718592 / 16)
    
    printf("[THEORY] Configuration:\n");
    printf("  - Input:  %dx%dx%d = %d bytes\n", INPUT_H, INPUT_W, INPUT_C, IFM_SIZE);
    printf("  - Weights: %dx%dx%dx%d = %d bytes\n", OUTPUT_F, KERNEL_H, KERNEL_W, INPUT_C, WEIGHTS_SIZE);
    printf("  - Output: %dx%dx%d = %d bytes\n\n", OUTPUT_H, OUTPUT_W, OUTPUT_F, OFM_SIZE);

    printf("[THEORY] Memory Layout:\n");
    printf("  - IFM BRAM depth: %d rows (16 bytes/row)\n", theoretical_load_ifm_rows);
    printf("  - Weight BRAM/PE: %d rows, Total: %d rows\n", theoretical_weight_per_pe, theoretical_weight_per_pe * NUM_PES);
    printf("  - OFM BRAM depth: %d rows\n\n", OFM_BRAM_DEPTH);

    printf("[THEORY] Performance (cycles):\n");
    printf("  - Load IFM:      %d cycles\n", theoretical_load_ifm_rows);
    printf("  - Load Weights:  %d cycles\n", theoretical_weight_per_pe * NUM_PES);
    printf("  - Total Load:    %d cycles\n", theoretical_load_cycles);
    printf("  - Compute (Sequential PE):  %d cycles (16 PE tuần tự)\n", theoretical_compute_cycles_sequential);
    printf("  - Compute (Parallel 16 PE): %d cycles (nếu 16 cores lý tưởng)\n", theoretical_compute_cycles_parallel);
    printf("  - TOTAL (Sequential Code):  %d cycles\n\n", theoretical_load_cycles + theoretical_compute_cycles_sequential);
    printf("  [NOTE] Thực thi: Parallel OpenMP trên %d cores, nhưng loop tính 16 PE\n", NUM_PES);
    printf("======================================================\n\n");

    clock_t time_start_all = clock();

    // 1. Nạp dữ liệu từ file vào DRAM
    printf("[PROFILE] Phase 1: Loading from file to DRAM...\n");
    clock_t time_file_start = clock();
    load_file_to_dram("ifm.txt", dram, IFM_SIZE);
    load_file_to_dram("weights.txt", dram + IFM_SIZE, WEIGHTS_SIZE);
    clock_t time_file_end = clock();
    double file_load_time = (double)(time_file_end - time_file_start) / CLOCKS_PER_SEC * 1000;
    printf("  Elapsed time: %.3f ms\n\n", file_load_time);

    // 2. Nạp IFM vào BRAM
    printf("[PROFILE] Phase 2: Loading IFM to BRAM...\n");
    clock_t time_bram_ifm_start = clock();
    int ifm_chunks = IFM_SIZE / DRAM_BUS_WIDTH_BYTES;
    long long load_iterations = 0;
    for (int i = 0; i < ifm_chunks; i++) {
        load_bram(dram, i * DRAM_BUS_WIDTH_BYTES, DRAM_BUS_WIDTH_BYTES, ifm_bram, i);
        load_iterations++;
    }
    clock_t time_bram_ifm_end = clock();
    double ifm_bram_time = (double)(time_bram_ifm_end - time_bram_ifm_start) / CLOCKS_PER_SEC * 1000;
    printf("  IFM rows loaded: %d\n", ifm_chunks);
    printf("  Elapsed time: %.3f ms\n\n", ifm_bram_time);

    // 3. Nạp weight vào BRAM của từng PE theo kiểu interleaved
    printf("[PROFILE] Phase 3: Loading Weights to BRAM (per PE)...\n");
    clock_t time_bram_weight_start = clock();
    for (int f_group = 0; f_group < WEIGHT_BRAM_FILTER_SETS; f_group++) {
        for (int chunk = 0; chunk < SINGLE_FILTER_CHUNKS; chunk++) {
            for (int pe = 0; pe < NUM_PES; pe++) {
                int filter_idx = f_group * NUM_PES + pe;
                int dram_offset = IFM_SIZE + filter_idx * (KERNEL_H * KERNEL_W * INPUT_C) + chunk * DRAM_BUS_WIDTH_BYTES;
                int bram_row = f_group * SINGLE_FILTER_CHUNKS + chunk;
                load_bram(dram, dram_offset, DRAM_BUS_WIDTH_BYTES, (int8_t (*)[DRAM_BUS_WIDTH_BYTES])weight_bram_pointers[pe], bram_row);
                load_iterations++;
            }
        }
    }
    clock_t time_bram_weight_end = clock();
    double weight_bram_time = (double)(time_bram_weight_end - time_bram_weight_start) / CLOCKS_PER_SEC * 1000;
    printf("  Weight rows per PE: %d\n", SINGLE_FILTER_CHUNKS * WEIGHT_BRAM_FILTER_SETS);
    printf("  Elapsed time: %.3f ms\n\n", weight_bram_time);

    // 4. Tính toán và lưu OFM vào BRAM
    printf("[PROFILE] Phase 4: Compute convolution...\n");
    clock_t time_compute_start = clock();
    int channel_chunks = INPUT_C / DSP_PER_PE;
    long long compute_loop_iterations = 0;
    
    for (int oh = 0; oh < OUTPUT_H; oh++) {
        for (int ow = 0; ow < OUTPUT_W; ow++) {
            for (int f_group = 0; f_group < WEIGHT_BRAM_FILTER_SETS; f_group++) {
                memset(pe_accumulators, 0, sizeof(pe_accumulators));
                for (int kh = 0; kh < KERNEL_H; kh++) {
                    for (int kw = 0; kw < KERNEL_W; kw++) {
                        for (int c_chunk = 0; c_chunk < channel_chunks; c_chunk++) {
                            // ===== THỰC SỰ SONG SONG: 16 PE chạy đồng thời trên 16 cores =====
                            int ih = oh * STRIDE + kh - PADDING;
                            int iw = ow * STRIDE + kw - PADDING;
                            int ifm_bram_row = (ih < 0 || ih >= INPUT_H || iw < 0 || iw >= INPUT_W) ? -1 : (ih * INPUT_W + iw) * channel_chunks + c_chunk;
                            int weight_bram_row = f_group * SINGLE_FILTER_CHUNKS + (kh * KERNEL_W + kw) * channel_chunks + c_chunk;
                            
                            #pragma omp parallel for num_threads(NUM_PES)
                            for (int pe = 0; pe < NUM_PES; pe++) {
                                int32_t partial_sum_result;
                                compute_pe(DSP_PER_PE, ifm_bram_row, weight_bram_row, pe, channel_chunks, &partial_sum_result);
                                #pragma omp critical
                                pe_accumulators[pe] += partial_sum_result;
                            }
                            // Đếm 16 PE (thực tế tuần tự từ qua góc nhìn outer loop)
                            #pragma omp atomic
                            compute_loop_iterations += NUM_PES;
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
    clock_t time_compute_end = clock();
    double compute_time = (double)(time_compute_end - time_compute_start) / CLOCKS_PER_SEC * 1000;
    printf("  Output shape: %dx%dx%d = %d elements\n", OUTPUT_H, OUTPUT_W, OUTPUT_F, OUTPUT_H*OUTPUT_W*OUTPUT_F);
    printf("  Elapsed time: %.3f ms\n\n", compute_time);

    // 5. Xuất OFM ra file
    printf("[PROFILE] Phase 5: Writing OFM to file...\n");
    clock_t time_write_start = clock();
    write_ofm_bram_to_file("ofm_output.txt", ofm_bram);
    clock_t time_write_end = clock();
    double write_time = (double)(time_write_end - time_write_start) / CLOCKS_PER_SEC * 1000;
    printf("  Elapsed time: %.3f ms\n\n", write_time);

    clock_t time_end_all = clock();
    double total_time = (double)(time_end_all - time_start_all) / CLOCKS_PER_SEC * 1000;

    // =============== TÍNH CYCLES THỰC TẾ ===============
    long long total_theoretical_load_cycles = theoretical_load_ifm_rows + (theoretical_weight_per_pe * NUM_PES);
    
    // Approximate CPU frequency từ compute phase
    double actual_compute_cycles = (double)compute_loop_iterations;
    double cpu_freq_estimate = (actual_compute_cycles * 1000000) / (compute_time * 1000);
    
    printf("======================================================\n");
    printf("[RESULT] Total execution time: %.3f ms\n", total_time);
    printf("  - File load:    %.3f ms (%.1f%%)\n", file_load_time, file_load_time/total_time*100);
    printf("  - IFM BRAM:     %.3f ms (%.1f%%)\n", ifm_bram_time, ifm_bram_time/total_time*100);
    printf("  - Weight BRAM:  %.3f ms (%.1f%%)\n", weight_bram_time, weight_bram_time/total_time*100);
    printf("  - Compute:      %.3f ms (%.1f%%)\n", compute_time, compute_time/total_time*100);
    printf("  - Write output: %.3f ms (%.1f%%)\n", write_time, write_time/total_time*100);
    
    printf("\n======================================================\n");
    printf("[FORMULA] Symbolic Cycle Calculation:\n");
    printf("  Load cycles = IFM_rows + (Weight_per_PE × NUM_PES)\n");
    printf("              = %d + (%d × %d)\n", theoretical_load_ifm_rows, theoretical_weight_per_pe, NUM_PES);
    printf("              = %d + %d\n", theoretical_load_ifm_rows, theoretical_weight_per_pe * NUM_PES);
    printf("              = %d cycles\n\n", theoretical_load_cycles);
    
    printf("  Compute cycles (Sequential PE) = OH × OW × F_group × KH × KW × C_chunk × NUM_PES\n");
    printf("                                  = %d × %d × %d × %d × %d × %d × %d\n", 
           OUTPUT_H, OUTPUT_W, WEIGHT_BRAM_FILTER_SETS, KERNEL_H, KERNEL_W, channel_chunks, NUM_PES);
    printf("                                  = %d cycles\n\n", theoretical_compute_cycles_sequential);
    
    printf("  Compute cycles (Parallel 16 PE) = OH × OW × F_group × KH × KW × C_chunk\n");
    printf("                                   = %d × %d × %d × %d × %d × %d\n", 
           OUTPUT_H, OUTPUT_W, WEIGHT_BRAM_FILTER_SETS, KERNEL_H, KERNEL_W, channel_chunks);
    printf("                                   = %d cycles (if perfect 16× speedup)\n\n", theoretical_compute_cycles_parallel);
    
    printf("======================================================\n");
    printf("[CYCLES] Theoretical vs Actual (TRUE PARALLEL with OpenMP):\n");
    printf("  - Theoretical Load cycles:    %lld cycles\n", total_theoretical_load_cycles);
    printf("  - Theoretical Compute (Sequential Model): %d cycles\n", theoretical_compute_cycles_sequential);
    printf("  - Theoretical Total (Sequential Model):   %d cycles\n\n", theoretical_load_cycles + theoretical_compute_cycles_sequential);
    
    printf("  [ACTUAL EXECUTION]\n");
    printf("  - Actual Load iterations:     %lld iterations\n", load_iterations);
    printf("  - Actual Compute loop count:  %lld iterations (16 PE × outer loops)\n", compute_loop_iterations);
    printf("  - Actual Compute time:        %.3f ms (16 cores parallel)\n", compute_time);
    printf("  - Actual Total time:          %.3f ms\n\n", total_time);
    
    printf("  - Ratio vs Sequential Theory: %.2f x\n", actual_compute_cycles / theoretical_compute_cycles_sequential);
    printf("  - Compute speedup (vs sequential): %.2f x\n", (theoretical_compute_cycles_sequential / 16.0) / compute_time);
    printf("  - Load efficiency:            %.1f%% (Theory vs Actual)\n\n", 
           (theoretical_load_cycles / (double)load_iterations) * 100);
    
    printf("======================================================\n");

    return 0;
}
