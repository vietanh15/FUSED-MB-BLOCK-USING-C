# Optimized Pipeline L1 + L2 with Ping-Pong & MUX/Interlock

## 1. Key Changes from Sequential PE Count

### **Removed: Sequential PE Counting**

❌ **OLD (Sequential PE):**
```
Compute_seq_PE = OH × OW × F_group × KH × KW × C_chunk × NUM_PES
               = 32 × 32 × 8 × 3 × 3 × 4 × 16
               = 4,718,592 cycles
```

❌ **Why bad:** Counts PE loop sequentially, but PE actually runs parallel

---

### **NEW: Parallel PE Counting Only**

✅ **NEW (Parallel PE only):**
```
Compute_16PE = OH × OW × F_group × KH × KW × C_chunk
             = 32 × 32 × 8 × 3 × 3 × 4
             = 294,912 cycles

Compute_4PE  = OH × OW × F_group × C_chunk  (L2, no kernel for 1×1)
             = 32 × 32 × 16 × 8
             = 131,072 cycles
```

✅ **Why good:** Reflects ACTUAL parallel execution, no mutex overhead counting

---

## 2. Layer 2 Optimization: Add Ping-Pong + MUX/Interlock

### **Current L2 Problem:**

```
L2 Input: L1 OFM (32×32×128)
Currently: Load ENTIRE L1 OFM into L2_IFM_BRAM before compute
Problem: Huge L2_IFM_BRAM (128 KB), compute waits for full load
```

### **New L2 Solution: Ping-Pong with MUX/Interlock**

**Same strategy as L1:**
- Ping-pong 6 image rows (3 PING + 3 PONG) 
- Load overlapped with compute
- MUX reads active bank OR inactive bank if needed
- Interlock prevents write to bank while compute still using

**Memory savings:**
```
L2_IFM_BRAM (old):  32 × 128 = 4,096 rows = 64 KB
L2_IFM_BRAM (new):  6 × 128 = 768 rows = 12 KB
Saved:              52 KB (81.25% reduction!)

Total BRAM freed: L1 (52 KB) + L2 (52 KB) = 104 KB!
```

---

## 3. Formula Simplification

### **Layer 1 (DWC, 16 PE)**

```
┌─ Load Phase ──────────────────────────────┐
│ L1_weight_load = 4,608 cycles             │
│ L1_IFM_preload = 384 cycles (first bank)  │
│ L1_load_effective = 4,992 cycles          │
└───────────────────────────────────────────┘

┌─ Compute Phase ───────────────────────────┐
│ L1_compute_16PE = 294,912 cycles          │
│ (parallel execution, 16 PE on 16 cores)   │
└───────────────────────────────────────────┘

Total L1: 4,992 + 294,912 = 299,904 cycles
```

### **Layer 2 (PWC, 4 PE with Ping-Pong)**

```
┌─ Load Phase ──────────────────────────────┐
│ L2_weight_load = 512 cycles               │
│ L2_identity_load = 4,096 cycles           │
│ L2_IFM_preload = 384 cycles (first bank)  │
│ L2_load_effective = 4,992 cycles          │
└───────────────────────────────────────────┘

┌─ Compute Phase ───────────────────────────┐
│ L2_compute_4PE = 131,072 cycles           │
│ L2_residual_add = 16,384 cycles           │
│ L2_total_compute = 147,456 cycles         │
└───────────────────────────────────────────┘

Total L2: 4,992 + 147,456 = 152,448 cycles
```

### **Pipeline Formula (Fully Overlapped)**

```
Preload_weight = L1_weight + L2_weight + L2_identity
               = 4,608 + 512 + 4,096
               = 9,216 cycles

Pipeline_critical_path = max(L1_compute_16PE, L2_compute_4PE)
                       = max(294,912, 147,456)
                       = 294,912 cycles

Total_pipeline = Preload_weight + Pipeline_critical_path
               = 9,216 + 294,912
               = 304,128 cycles

Speedup vs non-pipelined:
  Non-pipelined = 9,216 + 294,912 + 147,456 = 451,584
  Pipelined     = 304,128
  Speedup       = 451,584 / 304,128 = 1.486×
```

---

## 4. MUX/Interlock Logic for Both L1 & L2

### **Core Concept**

```
Two banks (PING + PONG):
┌─ PING BANK ────────────┐  ┌─ PONG BANK ────────────┐
│ Image rows 0-2         │  │ Image rows 3-5         │
│ (reading for compute)  │  │ (loading new data)     │
├────────────────────────┤  ├────────────────────────┤
│ Active during compute  │  │ Inactive, loading      │
│ MUX selects PING       │  │ MUX reads PING if      │
│                        │  │ compute needs it       │
└────────────────────────┘  └────────────────────────┘
        ↓ Every 3 rows ↓
   SWAP BANKS (Interlock)
```

### **Timeline**

```
t=0:      Pre-load PING (rows 0-2)      → 384 cycles
t=384:    Pre-load L2 weights            → 512 cycles
t=896:    Pre-load L2 identity           → 4,096 cycles
t=4,992:  Start compute PING + Load PONG parallel

Main loop (for each bank transition):
  - Compute PING (using rows 0-2, 8 rows max for 3×3 kernel)
  - Parallel load PONG (rows 3-5) in background
  - Compute determines if it needs PING or PONG (MUX logic)
  - Interlock blocks PONG write until compute done with PING
  - When compute starts PONG rows: swap + load PING next

Key: Compute >> Load, so load ALWAYS finishes before swap needed
     9,216 cycles/row compute > 384 cycles load ✓
```

### **MUX Logic (from Compute's Perspective)**

```c
// When accessing pixel (iw, ih):
int target_bank = -1;
int target_slot = -1;

// 1. Check active bank FIRST
if (ih >= bank_start_ih[active_bank] && 
    ih < bank_start_ih[active_bank] + BANK_SIZE) {
    target_bank = active_bank;
    target_slot = ih - bank_start_ih[active_bank];
}

// 2. If not found, check inactive bank (MUX fallback)
if (target_bank == -1) {
    if (ih >= bank_start_ih[inactive_bank] && 
        ih < bank_start_ih[inactive_bank] + BANK_SIZE) {
        target_bank = inactive_bank;
        target_slot = ih - bank_start_ih[inactive_bank];
    }
}

// 3. Calculate BRAM address
if (target_bank != -1) {
    int bram_row = target_bank * 384 + target_slot * 128 + (iw * 64) / 16;
    data = ifm_bram[bram_row];
}
```

### **Interlock Logic (Guards Load)**

```c
volatile bool compute_finished_using_old_bank = false;

#pragma omp parallel sections
{
    #pragma omp section
    {
        // COMPUTE: Process data from PING bank
        for (int oh = 0; oh < END_OH; oh++) {
            // ... compute using active_bank
        }
        compute_finished_using_old_bank = true;  // Signal done
    }

    #pragma omp section
    {
        // LOAD: Blocked until compute signals done
        while (!compute_finished_using_old_bank) {
            // Busy-wait - cannot write to inactive_bank yet
        }
        
        // Now safe to load new data into inactive_bank
        load_bank(inactive_bank, next_start_ih, ifm_bram, dram);
        
        // Swap for next iteration
        active_bank = 1 - active_bank;
    }
}
```

---

## 5. Memory Layout Comparison

### **Sequential (Old)**

```
L1_IFM_BRAM:  4,096 rows = 64 KB   (full input stored)
L2_IFM_BRAM:  4,096 rows = 64 KB   (full L1 output stored)
Total:        8,192 rows = 128 KB
```

### **Ping-Pong Optimized (New)**

```
L1_IFM_BRAM:  768 rows = 12 KB    (6 image rows: 3 PING + 3 PONG)
L2_IFM_BRAM:  768 rows = 12 KB    (6 image rows: 3 PING + 3 PONG)
Total:        1,536 rows = 24 KB
Savings:      128 KB - 24 KB = 104 KB (81.25% reduction!)
```

---

## 6. Updated Code Logic

### **Layer 1 Structure**

```c
// Removed: 
//   - l1_compute_iterations_seq_pe counting
// Kept:
//   - l1_compute_iterations_parallel_pe (actual 16-PE parallel iterations)

// Ping-Pong + MUX/Interlock:
while (oh < OUTPUT_H) {
    // MUX: Read from active_bank or inactive_bank (if compute needs it)
    if (pixel_in_active_bank) {
        data = active_bank[...]
    } else if (pixel_in_inactive_bank) {
        data = inactive_bank[...]  // Fallback read
    }
    
    // Compute with parallel reduction (NO critical section)
    #pragma omp parallel for num_threads(16)
    for (int pe = 0; pe < 16; pe++) {
        temp_accum[pe] += compute_pe(pe);  // Each thread local
    }
    
    // Interlock: Load PONG while compute uses PING
    if (compute_finished_using_ping) {
        load_bank(PONG, next_block, ifm_bram, dram);
    }
}
```

### **Layer 2 Structure**

```c
// Same Ping-Pong + MUX/Interlock as L1
// Removed:
//   - l2_compute_iterations_seq_pe counting
// Kept:
//   - l2_compute_iterations_parallel_pe (actual 4-PE parallel iterations)

for (int oh = 0; oh < OUTPUT_H; oh++) {
    for (int ow = 0; ow < OUTPUT_W; ow++) {
        // MUX: Read L1 OFM from PING/PONG bank
        if_pixel_in_active_bank:
            ifm_data = l1_ofm_bram_active[...]
        else:
            ifm_data = l1_ofm_bram_inactive[...]
        
        // Compute with 4 PE parallel reduction
        #pragma omp parallel for num_threads(4)
        for (int pe = 0; pe < 4; pe++) {
            temp_accum[pe] += compute_pe(pe);
        }
        
        // Residual add (part of compute count)
        output[pixel] = temp_accum[pe] + identity_data;
    }
}
```

---

## 7. Comparison: Old vs New Formula

| Metric | Old (Seq-PE) | New (Parallel Only) | Reduction |
|--------|-------------|-------------------|-----------|
| **L1 computation** | 4,718,592 | 294,912 | 16× |
| **L2 computation** | 524,288 | 131,072 | 4× |
| **Memory (L1)** | 64 KB | 12 KB | 81.25% |
| **Memory (L2)** | 64 KB | 12 KB | 81.25% |
| **Total memory** | 128 KB | 24 KB | 81.25% |
| **Effective cycles** | 451,584 | 304,128 | 1.49× speedup |

---

## 8. Key Benefits

✅ **No Sequential PE Counting:**
- Reflects ACTUAL hardware parallelism
- No spurious "4.7M cycles" from counting PE loop sequentially

✅ **Ping-Pong for Both L1 & L2:**
- 104 KB total memory freed
- Can now fit more weights, larger kernels, or additional processing

✅ **MUX/Interlock Strategy:**
- Load completely hidden by compute (9,216 >> 384)
- No stalls, no bottlenecks
- Fully pipelined execution

✅ **Cleaner Formula:**
- Load: One-time preload + overlap ignored (covered by compute time)
- Compute: Only parallel PE counts
- Total: Preload + max(L1, L2) compute (pipeline critical path)

---

## 9. Implementation Status

**Files to Modify:**
1. ✅ Create: `OPTIMIZED_PIPELINE_FORMULA.md` (this file)
2. 🔄 Modify: `pinelinel1withl2.c`
   - Remove seq-PE counting
   - Add L2 ping-pong + MUX/Interlock
   - Keep only parallel PE counts
   - Add interlock synchronization

**Result:**
- True parallel execution formula
- Optimal memory usage (24 KB vs 128 KB)
- Full pipeline overlap (1.49× speedup)
- Clean, understandable cycle calculations
