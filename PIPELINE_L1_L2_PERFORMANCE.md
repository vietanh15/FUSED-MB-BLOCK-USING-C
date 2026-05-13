# Pipeline L1 + L2 Performance Formula

File code: `pinelinel1withl2.c`

Muc tieu cua ban nay:

- L1 dung ping-pong IFM 1 BRAM logic gom 6 hang anh: `3 PING + 3 PONG`.
- Khong dung `halo_buffer`.
- Khi compute can du lieu bank cu, MUX doc `inactive_bank` cu.
- Interlock chi cho phep load ghi de `inactive_bank` sau khi compute block hien tai khong con can bank cu nua.
- L2 chay PWC + residual voi 4 PE de tiet kiem tai nguyen, vi L2 nhe hon L1.
- L2 consume theo pixel L1 vua produce, khong can buffer trung gian ngoai `l1_ofm_bram` la output chinh cua L1.

## 1. Thong So

| Thong so | Gia tri |
|---|---:|
| L1 input | `32 x 32 x 64` |
| L1 output | `32 x 32 x 128` |
| L1 kernel | `3 x 3` |
| L1 PE | `16` |
| L2 input | `32 x 32 x 128` |
| L2 output | `32 x 32 x 64` |
| L2 kernel | `1 x 1` |
| L2 PE | `4` |
| DSP/PE | `16` |
| DRAM row | `16 bytes` |

## 2. L1 Ping-Pong MUX/Interlock

L1 IFM:

```text
L1_IFM_SIZE = 32 * 32 * 64 = 65,536 bytes
L1_IFM_rows_raw = 65,536 / 16 = 4,096 rows
L1_IFM_chunks_per_image_row = 32 * 64 / 16 = 128 rows
```

Ping-pong capacity:

```text
Bank_size = 3 image rows
Bank_rows = 3 * 128 = 384 BRAM rows
PING + PONG = 2 * 384 = 768 BRAM rows = 12 KiB
```

Khong co halo buffer, nen IFM storage dung:

```text
L1_IFM_storage = 768 * 16 = 12,288 bytes = 12 KiB
Saving_vs_full_IFM = 64 KiB - 12 KiB = 52 KiB = 81.25%
```

Load IFM raw van la toan bo anh:

```text
L1_IFM_load_raw = 4,096 cycles
```

Block cuoi chi co 2 hang:

```text
10 full blocks * 384 + 1 tail block * 256 = 4,096 cycles
```

Vi dung MUX/interlock khong halo:

```text
Load first bank = 384 cycles
Remaining IFM load = 4,096 - 384 = 3,712 cycles
```

## 3. L1 Compute

```text
L1_filter_groups = 128 / 16 = 8
L1_channel_chunks = 64 / 16 = 4
```

Sequential PE count:

```text
L1_compute_seqPE
= OH * OW * filter_groups * KH * KW * channel_chunks * L1_NUM_PES
= 32 * 32 * 8 * 3 * 3 * 4 * 16
= 4,718,592 cycles
```

Ideal 16-PE hardware count:

```text
L1_compute_16PE
= 32 * 32 * 8 * 3 * 3 * 4
= 294,912 cycles
```

Per-pixel L1:

```text
L1_per_pixel_16PE = 8 * 3 * 3 * 4 = 288 cycles/pixel
```

## 4. L2 PWC + Residual With 4 PE

L2 weights:

```text
L2_WEIGHTS_SIZE = 64 * 128 = 8,192 bytes
L2_weight_load = 8,192 / 16 = 512 cycles
```

L2 identity:

```text
L2_identity_size = 32 * 32 * 64 = 65,536 bytes
L2_identity_load = 65,536 / 16 = 4,096 cycles
```

L2 compute:

```text
L2_filter_groups = 64 / 4 = 16
L2_channel_chunks = 128 / 16 = 8
```

Sequential PE count:

```text
L2_compute_seqPE
= OH * OW * filter_groups * channel_chunks * L2_NUM_PES
= 32 * 32 * 16 * 8 * 4
= 524,288 cycles
```

Ideal 4-PE hardware count:

```text
L2_compute_4PE
= 32 * 32 * 16 * 8
= 131,072 cycles
```

Residual add count, 4 output channels/group:

```text
L2_residual
= OH * OW * filter_groups
= 32 * 32 * 16
= 16,384 cycles
```

L2 total ideal stream:

```text
L2_stream_4PE = L2_compute_4PE + L2_residual
              = 131,072 + 16,384
              = 147,456 cycles
```

Per-pixel L2:

```text
L2_per_pixel_4PE = 16 * 8 + 16 = 144 cycles/pixel
```

L2 voi 4 PE van nhanh hon L1:

```text
L1_per_pixel_16PE = 288 cycles
L2_per_pixel_4PE  = 144 cycles
```

Do do L2 khong tro thanh bottleneck, con tiet kiem 12 PE so voi cau hinh 16 PE.

## 5. Pipeline Formula

Fixed preload:

```text
Preload
= L1_weight_load + L2_weight_load + L2_identity_load + L1_first_bank_load
= 4,608 + 512 + 4,096 + 384
= 9,600 cycles
```

L1 stream voi MUX/interlock:

```text
L1_stream
= L1_compute_16PE + L1_remaining_IFM_load
= 294,912 + 3,712
= 298,624 cycles
```

L2 stream:

```text
L2_stream = 147,456 cycles
```

Pipeline bound:

```text
Pipeline_stream = max(L1_stream, L2_stream)
                = max(298,624, 147,456)
                = 298,624 cycles
```

Total ideal pipeline:

```text
Total_pipeline
= Preload + Pipeline_stream
= 9,600 + 298,624
= 308,224 cycles
```

Neu khong pipeline L1/L2:

```text
Total_non_pipeline
= Preload + L1_stream + L2_stream
= 9,600 + 298,624 + 147,456
= 455,680 cycles
```

Pipeline speedup ly thuyet:

```text
Speedup = 455,680 / 308,224 = 1.478x
```

## 6. Actual Run

Build:

```powershell
gcc -O2 -fopenmp -lm -o pipeline_l1_l2_omp_check.exe pinelinel1withl2.c
```

Run:

```powershell
.\pipeline_l1_l2_omp_check.exe > pipeline_l1_l2_run.txt 2>&1
```

Exit code:

```text
EXIT=0
```

Summary:

```text
[LOAD]
  L1 weight:   theory=4608, actual=4608
  L1 IFM raw:  theory=4096, actual=4096
  L2 weight:   theory=512, actual=512
  L2 identity: theory=4096, actual=4096

[COMPUTE]
  L1 seq-PE:   theory=4718592, actual=4718592
  L1 16-PE:    theory=294912, actual=294912
  L2 seq-PE:   theory=524288, actual=524288
  L2 4-PE:     theory=131072, actual=131072
  L2 residual: theory=16384, actual=16384

[PIPELINE]
  L1 pixels produced: 1024 / 1024
  L2 pixels consumed: 1024 / 1024
```

Ket luan: ly thuyet va counter thuc te trong code khop tung hang.

