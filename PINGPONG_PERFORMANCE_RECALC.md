# Recalculation: Layer1 Ping-Pong Performance

Ngay trong code goc `layer1withpingpong.c`, ping-pong IFM khong phai la 6 BRAM doc lap. Khai bao thuc te la:

```c
int8_t ifm_bram[IFM_BRAM_TOTAL_DEPTH][DRAM_BUS_WIDTH_BYTES];
```

Voi:

```c
IFM_BUFFER_IMG_ROWS = 6
IFM_CHUNKS_PER_IMG_ROW = INPUT_W * INPUT_C / 16 = 32 * 64 / 16 = 128
IFM_BRAM_TOTAL_DEPTH = 6 * 128 = 768 rows
```

Nghia dung: day la 1 IFM BRAM logic co 768 row 16-byte, duoc chia thanh 2 bank:

| Bank | So hang anh | Row BRAM/1 hang anh | Tong row BRAM |
|---|---:|---:|---:|
| PING | 3 | 128 | 384 |
| PONG | 3 | 128 | 384 |
| Tong | 6 | 128 | 768 |

Loi sai trong cong thuc cu la dien giai "6 hang" nhu 6 BRAM hoac nhu 6 row BRAM nho. Thuc te do la 6 hang anh, moi hang anh gom 128 row BRAM 16-byte.

## 1. Thong so layer1

Tu `layer1.c` va `layer1withpingpong.c`:

| Ky hieu | Gia tri |
|---|---:|
| `INPUT_H` | 32 |
| `INPUT_W` | 32 |
| `INPUT_C` | 64 |
| `OUTPUT_H` | 32 |
| `OUTPUT_W` | 32 |
| `OUTPUT_F` | 128 |
| `KERNEL_H x KERNEL_W` | 3 x 3 |
| `PADDING` | 1 |
| `STRIDE` | 1 |
| `NUM_PES` | 16 |
| `DSP_PER_PE` | 16 |
| `DRAM_BUS_WIDTH_BYTES` | 16 |

Suy ra:

```text
IFM_SIZE     = 32 * 32 * 64        = 65,536 bytes
WEIGHTS_SIZE = 128 * 3 * 3 * 64    = 73,728 bytes
OFM_SIZE     = 32 * 32 * 128       = 131,072 bytes

IFM rows full image = 65,536 / 16 = 4,096 BRAM rows
Chunks per image row = 32 * 64 / 16 = 128 BRAM rows
Channel chunks = 64 / 16 = 4
Filter groups = 128 / 16 = 8
Single filter chunks = 3 * 3 * 64 / 16 = 36
Weight rows per PE = 36 * 8 = 288
Weight rows total = 288 * 16 = 4,608
OFM rows = 131,072 / 16 = 8,192
```

## 2. No Ping-Pong

`layer1.c` nap toan bo IFM vao BRAM truoc khi compute.

Dung luong IFM BRAM:

```text
IFM_BRAM_DEPTH = IFM_SIZE / BUS = 65,536 / 16 = 4,096 rows
IFM capacity = 4,096 * 16 = 65,536 bytes = 64 KiB
```

Load cycles:

```text
Load_IFM = INPUT_H * INPUT_W * INPUT_C / BUS
         = 32 * 32 * 64 / 16
         = 4,096 cycles

Load_Weight = OUTPUT_F * KH * KW * INPUT_C / BUS
            = 128 * 3 * 3 * 64 / 16
            = 4,608 cycles

Load_total_no_pingpong = 4,096 + 4,608 = 8,704 cycles
```

Compute cycles neu vong `for pe` chay tuan tu:

```text
Compute_seqPE = OH * OW * (OUTPUT_F / NUM_PES) * KH * KW
                * (INPUT_C / DSP_PER_PE) * NUM_PES
              = 32 * 32 * 8 * 3 * 3 * 4 * 16
              = 4,718,592 cycles
```

Compute cycles ly tuong neu 16 PE chay song song:

```text
Compute_16PE = Compute_seqPE / 16
             = 294,912 cycles
```

Tong:

```text
Total_no_pingpong_seqPE = 8,704 + 4,718,592 = 4,727,296 cycles
Total_no_pingpong_16PE  = 8,704 +   294,912 =   303,616 cycles
```

## 3. Ping-Pong Dung Theo Code Goc

`layer1withpingpong.c` dung 2 bank, moi bank 3 hang anh:

```text
BANK_SIZE = 3 image rows
Bank_rows = BANK_SIZE * IFM_CHUNKS_PER_IMG_ROW
          = 3 * 128
          = 384 BRAM rows
```

Dung luong IFM ping-pong:

```text
IFM_BRAM_TOTAL_DEPTH = 2 * Bank_rows = 2 * 384 = 768 rows
IFM capacity = 768 * 16 = 12,288 bytes = 12 KiB
```

So voi no ping-pong:

```text
Memory_saved_IFM = 4,096 - 768 = 3,328 rows
                 = 53,248 bytes = 52 KiB
Saving_percent = 3,328 / 4,096 = 81.25%
```

### 3.1 Tong So Block IFM Can Load

Khong duoc tinh `ceil(32/3) * 384 = 11 * 384 = 4,224` cho tong byte thuc, vi block cuoi khong day 3 hang.

Code load cac block bat dau o:

```text
start_ih = 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30
```

Trong do:

```text
10 block dau day 3 hang: 10 * 3 * 128 = 3,840 rows
block cuoi co 2 hang:     2 * 128 =   256 rows
Tong IFM load thuc:                  4,096 rows
```

Cong thuc tong quat:

```text
N_full = INPUT_H / BANK_SIZE
N_tail = INPUT_H % BANK_SIZE

IFM_load_rows_pingpong =
    N_full * BANK_SIZE * IFM_CHUNKS_PER_IMG_ROW
  + N_tail * IFM_CHUNKS_PER_IMG_ROW

Voi INPUT_H=32, BANK_SIZE=3:
N_full=10, N_tail=2
IFM_load_rows_pingpong = 10*3*128 + 2*128 = 4,096 rows
```

Ket luan quan trong: ping-pong khong giam tong du lieu IFM phai doc tu DRAM. No chi giam dung luong BRAM can giu dong thoi va cho phep overlap load voi compute.

### 3.2 Load Cycles Hieu Dung Khi Overlap

Thu tu trong code:

1. Load weights truoc compute.
2. Preload PING block dau tien.
3. Compute tren active bank, dong thoi load bank con lai neu con du lieu.

Do do phan load khong overlap bat buoc la:

```text
Load_weight_pre = 4,608 cycles
Load_first_bank = 3 * 128 = 384 cycles
```

Cac IFM load con lai:

```text
IFM_remaining_load = 4,096 - 384 = 3,712 cycles
```

Neu compute cua moi stage lon hon load cua block tiep theo thi `IFM_remaining_load` bi che boi compute. Dieu nay dung voi ca model 16 PE song song:

```text
Compute_per_output_row_16PE = OW * F_group * KH * KW * C_chunk
                            = 32 * 8 * 3 * 3 * 4
                            = 9,216 cycles

Stage nho nhat co 1 output row:
9,216 cycles > max load block 384 cycles
```

Vi vay:

```text
Effective_load_pingpong = Load_weight_pre + Load_first_bank
                        = 4,608 + 384
                        = 4,992 cycles
```

Tong hieu dung:

```text
Total_pingpong_seqPE = 4,992 + 4,718,592 = 4,723,584 cycles
Total_pingpong_16PE  = 4,992 +   294,912 =   299,904 cycles
```

Speedup tren ly thuyet so voi no ping-pong:

```text
Saved_effective_cycles = 8,704 - 4,992 = 3,712 cycles

Speedup_seqPE = 4,727,296 / 4,723,584 = 1.000786x
Speedup_16PE  =   303,616 /   299,904 = 1.012377x
```

Ping-pong trong cau hinh nay chu yeu tiet kiem IFM BRAM 52 KiB. Performance chi tang nho vi compute lon hon load rat nhieu.

## 4. Luu Y Ve Halo Buffer Trong Code

`layer1withpingpong.c` co them:

```c
#define HALO_ROWS (KERNEL_H - 1)
int8_t halo_buffer[HALO_BUFFER_DEPTH][DRAM_BUS_WIDTH_BYTES];
```

Voi `HALO_ROWS=2`:

```text
HALO_BUFFER_DEPTH = 2 * 128 = 256 rows
Halo capacity = 256 * 16 = 4,096 bytes = 4 KiB
```

Neu tinh dung theo khai bao C hien tai, tong IFM-side storage la:

```text
Ping-pong IFM BRAM = 768 rows = 12 KiB
Halo buffer        = 256 rows =  4 KiB
Tong               = 1,024 rows = 16 KiB
```

Neu y tuong ban dau la "chi 1 IFM BRAM gom 6 hang anh, khong buffer IFM phu", thi `halo_buffer` can duoc xem la logic phu phai loai bo hoac thay bang interlock/MUX doc bank cu truoc khi ghi de. Neu chap nhan halo la buffer phu nho, thi dung luong tiet kiem van la:

```text
64 KiB - 16 KiB = 48 KiB = 75%
```

## 5. Doi Chieu Voi Code Da Chay

### 5.1 Build

Lenh da chay:

```powershell
gcc -O2 -fopenmp -lm -o layer1withpingpong_check.exe layer1withpingpong.c
```

Ket qua: build thanh cong.

Lenh build `layer1.c` hien tai:

```powershell
gcc -O2 -lm -o layer1_check.exe layer1.c
```

Ket qua: build that bai do:

```text
redefinition of 'time_file_end'
'channel_chunks' undeclared
```

Viec nay khong lam doi cong thuc no ping-pong, nhung can sua code neu muon build lai truc tiep tu `layer1.c` hien tai.

### 5.2 Log Ping-Pong Goc

`layer1withpingpong_check.exe` in dung lich load block:

```text
0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30
```

Block cuoi chi in:

```text
Nap hang anh 30
Nap hang anh 31
```

Dieu nay xac nhan cong thuc IFM load dung la:

```text
10 * 384 + 256 = 4,096 rows
```

khong phai:

```text
11 * 384 = 4,224 rows
```

### 5.3 Log No Ping-Pong Profiled

`layer1_profiled.exe` da chay va log:

```text
IFM rows loaded: 4096
Weight rows per PE: 288
Actual Load iterations: 8704
Actual Compute loop count: 4718592
```

Khớp voi cong thuc:

```text
Load_no_pingpong = 4096 + 4608 = 8704
Compute_seqPE = 4718592
```

### 5.4 Output Correctness Sau Khi Sua

Loi correctness ban dau nam o block cuoi `start_ih=30`. Block nay chi co 2 hang hop le (`ih=30,31`), nhung code cu lay halo theo slot co dinh:

```text
slot 1 -> ih=31
slot 2 -> ih=32 invalid
```

Vi vay `oh=31` bi mat `ih=30`, trong khi convolution 3x3 padding 1 can:

```text
oh=31 -> ih=30,31,32(padding)
```

Da sua `layer1withpingpong.c` de halo luon lay `HALO_ROWS` hang hop le cuoi cung cua bank sap bi ghi de:

```text
valid_rows_in_bank = min(BANK_SIZE, INPUT_H - bank_start_ih)
halo_source_slot = valid_rows_in_bank - HALO_ROWS + i
```

Voi block cuoi:

```text
valid_rows_in_bank = 2
halo slots = 0,1 -> ih=30,31
```

Sau khi build va chay lai:

```powershell
Compare-Object (Get-Content ofm_output.txt) (Get-Content ofm_outputl1withpingpong.txt)
```

Ket qua khong con khac biet:

```text
line_count 8192 8192
diff_lines 0
```

SHA256 sau khi sua:

```text
ofm_output.txt                040BEC54919AB337D7ED7B0CB353415E5D66F12C783BFF6882A5BCBBB32B9531
ofm_outputl1withpingpong.txt  040BEC54919AB337D7ED7B0CB353415E5D66F12C783BFF6882A5BCBBB32B9531
```

Nghia la ping-pong hien tai da khop voi no ping-pong ca ve lich load/buffer/cycle va ket qua OFM.

## 6. Cong Thuc Dung De Dung Ve Sau

### No Ping-Pong

```text
IFM_capacity_no_pp = INPUT_H * INPUT_W * INPUT_C bytes

Load_no_pp =
    INPUT_H * INPUT_W * INPUT_C / BUS
  + OUTPUT_F * KH * KW * INPUT_C / BUS

Compute_seqPE =
    OH * OW * (OUTPUT_F / NUM_PES) * KH * KW
    * (INPUT_C / DSP_PER_PE) * NUM_PES

Compute_16PE = Compute_seqPE / NUM_PES

Total_no_pp_seqPE = Load_no_pp + Compute_seqPE
Total_no_pp_16PE  = Load_no_pp + Compute_16PE
```

### Ping-Pong 1 IFM BRAM, 6 Hang Anh

```text
Chunks_per_image_row = INPUT_W * INPUT_C / BUS
Bank_rows = BANK_SIZE * Chunks_per_image_row
IFM_capacity_pp = 2 * Bank_rows * BUS

IFM_raw_load_pp = INPUT_H * INPUT_W * INPUT_C / BUS
Weight_load = OUTPUT_F * KH * KW * INPUT_C / BUS
First_bank_load = min(BANK_SIZE, INPUT_H) * Chunks_per_image_row

Effective_load_pp = Weight_load + First_bank_load
Neu compute moi stage >= load block tiep theo.

Total_pp_seqPE = Effective_load_pp + Compute_seqPE
Total_pp_16PE  = Effective_load_pp + Compute_16PE
```

Voi layer1 hien tai:

```text
IFM_capacity_pp = 2 * 3 * 128 * 16 = 12,288 bytes = 12 KiB
IFM_raw_load_pp = 4,096 cycles
Effective_load_pp = 4,608 + 384 = 4,992 cycles
Total_pp_seqPE = 4,723,584 cycles
Total_pp_16PE = 299,904 cycles
```
