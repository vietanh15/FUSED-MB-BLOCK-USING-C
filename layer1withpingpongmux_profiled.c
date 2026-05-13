#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
#include <time.h>

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
    for (int i = 0; i < 3; i++) {
        int current_ih = start_ih + i;
        if (current_ih < INPUT_H) {
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
    printf("========== LAYER1 PING-PONG MUX - OPTIMIZED ==========\n");
    printf("\n");

    // =============== THÔNG SỐ LÝ THUYẾT ===============
    int bank_size = 3;
    int num_banks = (INPUT_H + bank_size - 1) / bank_size;
    int theoretical_load_per_bank = bank_size * IFM_CHUNKS_PER_IMG_ROW;
    int theoretical_ifm_load_total = IFM_SIZE / DRAM_BUS_WIDTH_BYTES;
    int theoretical_weight_per_pe = SINGLE_FILTER_CHUNKS * WEIGHT_BRAM_FILTER_SETS;
    int theoretical_compute_cycles = 4718592;
    int theoretical_compute_cycles_parallel = 294912;  // 16 PE song song
    
    printf("[THEORY] Configuration:\n");
    printf("  - Input:  %dx%dx%d = %d bytes\n", INPUT_H, INPUT_W, INPUT_C, IFM_SIZE);
    printf("  - Weights: %dx%dx%dx%d = %d bytes\n", OUTPUT_F, KERNEL_H, KERNEL_W, INPUT_C, WEIGHTS_SIZE);
    printf("  - Output: %dx%dx%d = %d bytes\n\n", OUTPUT_H, OUTPUT_W, OUTPUT_F, OFM_SIZE);

    printf("[THEORY] Ping-Pong Memory Layout:\n");
    printf("  - Bank size: %d rows (3 image rows)\n", bank_size * IFM_CHUNKS_PER_IMG_ROW);
    printf("  - Total banks: %d\n", num_banks);
    printf("  - IFM BRAM depth: %d rows (6 rows buffer: 3 PING + 3 PONG)\n", IFM_BRAM_TOTAL_DEPTH);
    printf("  - Weight BRAM/PE: %d rows, Total: %d rows\n", theoretical_weight_per_pe, theoretical_weight_per_pe * NUM_PES);
    printf("  - OFM BRAM depth: %d rows\n\n", OFM_BRAM_DEPTH);

    printf("[THEORY] Memory Savings (vs Sequential):\n");
    printf("  - Sequential IFM BRAM: 4096 rows\n");
    printf("  - Ping-Pong IFM BRAM:  %d rows\n", IFM_BRAM_TOTAL_DEPTH);
    printf("  - Savings: %.1f%%\n\n", (1.0 - (double)IFM_BRAM_TOTAL_DEPTH/4096) * 100);

    printf("[THEORY] Performance (cycles):\n");
    printf("  - Load per bank:       %d cycles\n", theoretical_load_per_bank);
    printf("  - Num banks to load:   %d\n", num_banks);
    printf("  - IFM load total:      %d cycles (tail bank is partial)\n", theoretical_ifm_load_total);
    printf("  - Compute (Sequential PE): %d cycles (16 PE tuần tự)\n", theoretical_compute_cycles);
    printf("  - Compute (Parallel 16 PE): %d cycles (nếu 16 cores lý tưởng)\n", theoretical_compute_cycles_parallel);
    printf("  - With full overlap:   %d cycles (compute dominates)\n", theoretical_compute_cycles_parallel);
    printf("  - Theoretical speedup: ~0.18%% (compute-bound)\n\n");
    printf("  [NOTE] Thực thi: Parallel OpenMP trên %d cores\n", NUM_PES);
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

    // 2. Nạp weight vào BRAM của từng PE (chỉ làm 1 lần)
    printf("[PROFILE] Phase 2: Loading Weights to BRAM (per PE)...\n");
    clock_t time_bram_weight_start = clock();
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
    clock_t time_bram_weight_end = clock();
    double weight_bram_time = (double)(time_bram_weight_end - time_bram_weight_start) / CLOCKS_PER_SEC * 1000;
    printf("  Weight rows per PE: %d\n", SINGLE_FILTER_CHUNKS * WEIGHT_BRAM_FILTER_SETS);
    printf("  Elapsed time: %.3f ms\n\n", weight_bram_time);

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
    printf("[PROFILE] Phase 3: Pre-load initial bank (PING)...\n");
    clock_t time_initial_load = clock();
    load_bank(PING_BANK, 0, ifm_bram, dram);
    clock_t time_initial_load_end = clock();
    bank_start_ih[PING_BANK] = 0;
    double initial_load_time = (double)(time_initial_load_end - time_initial_load) / CLOCKS_PER_SEC * 1000;
    printf("  Elapsed time: %.3f ms\n\n", initial_load_time);
    
    // 4. Tính toán và lưu OFM với logic Ping-Pong và MUX+Interlock song song
    printf("[PROFILE] Phase 4: Compute with Ping-Pong buffering...\n");
    clock_t time_compute_total = clock();
    int channel_chunks = INPUT_C / DSP_PER_PE;
    double time_compute_only = 0;
    double time_load_only = 0;
    long long compute_loop_iterations = 0;
    long long load_iterations = theoretical_load_per_bank;

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
                clock_t comp_start = clock();
                for (int current_oh = oh; current_oh <= compute_end_oh; current_oh++) {
                    for (int ow = 0; ow < OUTPUT_W; ow++) {
                        for (int f_group = 0; f_group < WEIGHT_BRAM_FILTER_SETS; f_group++) {
                            memset(pe_accumulators, 0, sizeof(pe_accumulators));
                            for (int kh = 0; kh < KERNEL_H; kh++) {
                                for (int kw = 0; kw < KERNEL_W; kw++) {
                                    for (int c_chunk = 0; c_chunk < channel_chunks; c_chunk++) {
                                        int ih = current_oh * STRIDE + kh - PADDING;
                                        int iw = ow * STRIDE + kw - PADDING;

                                        const int8_t* zero_buffer = (const int8_t*)calloc(DRAM_BUS_WIDTH_BYTES, 1);
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
                                        
                                        // ===== THỰC SỰ SONG SONG: 16 PE chạy đồng thời trên 16 cores =====
                                        #pragma omp parallel for num_threads(NUM_PES)
                                        for (int pe = 0; pe < NUM_PES; pe++) {
                                            int32_t partial_sum_result;
                                            compute_pe_flexible(DSP_PER_PE, ifm_data_ptr, weight_bram_row, pe, &partial_sum_result);
                                            #pragma omp critical
                                            pe_accumulators[pe] += partial_sum_result;
                                        }
                                        // Đếm 16 PE (thực tế tuần tự từ qua góc nhìn outer loop)
                                        #pragma omp critical
                                        compute_loop_iterations += NUM_PES;
                                        free((void*)zero_buffer);
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
                clock_t comp_end = clock();
                time_compute_only += (double)(comp_end - comp_start) / CLOCKS_PER_SEC * 1000;
                compute_finished_using_old_bank = true;
            }

            #pragma omp section
            {
                // --- TASK 2: LOAD (Bị khóa cho đến khi COMPUTE xong) ---
                while (!compute_finished_using_old_bank) {
                    // Busy-wait
                }

                clock_t load_start = clock();
                if (next_block_start_ih < INPUT_H) {
                    int rows_to_load = INPUT_H - next_block_start_ih;
                    if (rows_to_load > BANK_SIZE) rows_to_load = BANK_SIZE;
                    load_iterations += rows_to_load * IFM_CHUNKS_PER_IMG_ROW;
                    load_bank(inactive_bank, next_block_start_ih, ifm_bram, dram);
                }
                clock_t load_end = clock();
                time_load_only += (double)(load_end - load_start) / CLOCKS_PER_SEC * 1000;
            }
        }

        if (next_block_start_ih < INPUT_H) {
            bank_start_ih[inactive_bank] = next_block_start_ih;
        }
        oh = compute_end_oh + 1;
        next_block_start_ih += BANK_SIZE;
        active_bank = 1 - active_bank;
    }
    clock_t time_compute_total_end = clock();
    double compute_phase_total = (double)(time_compute_total_end - time_compute_total) / CLOCKS_PER_SEC * 1000;

    // 5. Xuất OFM ra file
    printf("[PROFILE] Phase 5: Writing OFM to file...\n");
    clock_t time_write_start = clock();
    write_ofm_bram_to_file("ofm_output_mux.txt", ofm_bram);
    clock_t time_write_end = clock();
    double write_time = (double)(time_write_end - time_write_start) / CLOCKS_PER_SEC * 1000;
    printf("  Elapsed time: %.3f ms\n\n", write_time);

    clock_t time_end_all = clock();
    double total_time = (double)(time_end_all - time_start_all) / CLOCKS_PER_SEC * 1000;

    printf("======================================================\n");
    printf("[RESULT] Total execution time: %.3f ms\n", total_time);
    printf("  - File load:       %.3f ms (%.1f%%)\n", file_load_time, file_load_time/total_time*100);
    printf("  - Weight BRAM:     %.3f ms (%.1f%%)\n", weight_bram_time, weight_bram_time/total_time*100);
    printf("  - Initial IFM buf: %.3f ms (%.1f%%)\n", initial_load_time, initial_load_time/total_time*100);
    printf("  - Compute+Load:    %.3f ms (%.1f%%)\n", compute_phase_total, compute_phase_total/total_time*100);
    printf("    - Compute only:  %.3f ms\n", time_compute_only);
    printf("    - Load only:     %.3f ms\n", time_load_only);
    printf("  - Write output:    %.3f ms (%.1f%%)\n", write_time, write_time/total_time*100);
    
    printf("\n======================================================\n");
    printf("[FORMULA] Symbolic Cycle Calculation (Ping-Pong):\n");
    printf("  Load cycles (per bank) = Bank_size × IFM_chunks\n");
    printf("                         = %d × %d = %d cycles\n", bank_size, IFM_CHUNKS_PER_IMG_ROW, theoretical_load_per_bank);
    printf("  Total IFM load         = INPUT_H × IFM_chunks\n");
    printf("                         = %d × %d = %d cycles (tail bank is partial)\n\n", 
           INPUT_H, IFM_CHUNKS_PER_IMG_ROW, theoretical_ifm_load_total);
    
    printf("  Compute cycles (Sequential PE) = OH × OW × F_group × KH × KW × C_chunk × NUM_PES\n");
    printf("  (Same as sequential layer1.c)\n");
    printf("                                  = %d cycles\n\n", theoretical_compute_cycles);
    
    printf("  Compute cycles (Parallel 16 PE) = OH × OW × F_group × KH × KW × C_chunk\n");
    printf("                                   = %d cycles (if 16× speedup)\n\n", theoretical_compute_cycles_parallel);
    
    printf("======================================================\n");
    printf("[CYCLES] Theoretical vs Actual (TRUE PARALLEL with OpenMP + Ping-Pong):\n");
    printf("  - Theoretical IFM load cycles:            %d cycles\n", theoretical_ifm_load_total);
    printf("  - Theoretical Compute (Sequential Model): %d cycles\n", theoretical_compute_cycles);
    printf("  - Theoretical Compute (Parallel Model):  %d cycles\n", theoretical_compute_cycles_parallel);
    printf("  - Theoretical Total (Parallel Model):    ~%d cycles (compute-bound)\n\n", theoretical_compute_cycles_parallel);
    
    printf("  [ACTUAL EXECUTION]\n");
    printf("  - Actual Compute loop count:  %lld iterations (16 PE x outer loops)\n", compute_loop_iterations);
    printf("  - Actual Load iterations:     %lld iterations (overlapped)\n", load_iterations);
    printf("  - Actual Compute time:        %.3f ms (16 cores parallel)\n", time_compute_only);
    printf("  - Actual Total time:          %.3f ms\n\n", total_time);
    
    printf("  - Ratio vs Sequential Theory: %.2f x\n", (double)compute_loop_iterations / theoretical_compute_cycles);
    printf("  - Compute speedup (16 cores): %.2f x\n", (theoretical_compute_cycles / 16.0) / time_compute_only);
    printf("  - Compute efficiency:         %.1f%%\n\n", (theoretical_compute_cycles / (double)compute_loop_iterations) * 100);
    
    printf("======================================================\n");

    return 0;
}
