#pragma once

#include <iostream>
#define __gmp_const const
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "hls_math.h"

constexpr unsigned floorlog2(unsigned x)
{
#pragma HLS INLINE
    return x == 1 ? 0 : 1+floorlog2(x >> 1);
}

constexpr unsigned ceillog2(unsigned x)
{
#pragma HLS INLINE
    return x == 1 ? 0 : floorlog2(x - 1) + 1;
}

constexpr unsigned bitsNeeded(unsigned n) {
#pragma HLS INLINE
  return n <= 1 ? 0 : 1 + bitsNeeded((n + 1) / 2);
}

template <typename T>
static constexpr T ceildiv(T dividend, T divisor)
{
#pragma HLS INLINE
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
static constexpr T roundup(T dividend, T divisor)
{
#pragma HLS INLINE
    return ceildiv(dividend, divisor) * divisor;
}

template <typename T>
static constexpr T ap_fixed_epsilon()
{
#pragma HLS INLINE
    return T(1.0 / (1 << (T::width - T::iwidth)));
}

// Common Functionality Definition
#define LOG_1(n) (((n) >= 2) ? 1 : 0)
#define LOG_2(n) (((n) >= 1<<2) ? (2 + LOG_1((n)>>2)) : LOG_1(n))
#define LOG_4(n) (((n) >= 1<<4) ? (4 + LOG_2((n)>>4)) : LOG_2(n))
#define LOG_8(n) (((n) >= 1<<8) ? (8 + LOG_4((n)>>8)) : LOG_4(n))
#define LOG(n)   (((n) >= 1<<16) ? (16 + LOG_8((n)>>16)) : LOG_8(n))


typedef ap_fixed<32,16,AP_RND_CONV,AP_SAT> data_t;
//Off-chip Interface Parameters
constexpr size_t NUM_HP_IFC = 6;
constexpr size_t HP_IFC_BANDWIDTH = 128; 
constexpr size_t MAX_OFF_CHIP_BW = NUM_HP_IFC*HP_IFC_BANDWIDTH;

//Data Width Parameters. (Same as the data_t for now)
constexpr int IACTS_DATAWIDTH                = 32;   
constexpr int BIAS_DATAWIDTH                    = 32;                                                                                  
constexpr int WEIGHTS_DATAWIDTH           = 32;                                             
constexpr int OACTS_DATAWIDTH                    = 32;      

//On-chip buffer size
constexpr int IACT_BRAM_WIDTH = 32; //Stores 1 iact values at one row
constexpr int MAX_IACT_BRAM_ROW = MAX_IACTS_NUM / (IACT_BRAM_WIDTH / IACTS_DATAWIDTH); //32 rows as the maximum -> 4 brams in need
constexpr int PARALLEL_K = 32; // Given 2048 as k-dim size, process 32 at one time
constexpr int PARALLEL_N = 50; //20 //20 to get higher parallelism
constexpr int PARALLEL_M = 1;
constexpr int READ_PARALLEL_WEIGHT = PARALLEL_K * PARALLEL_N /(WEIGHT_URAM_WIDTH / WEIGHTS_DATAWIDTH); //32*50/9, weight num per block
constexpr int WEIGHT_URAM_WIDTH = 4 * 72;
constexpr int WEIGHT_NUM_PER_ROW_URAM = WEIGHT_URAM_WIDTH / WEIGHTS_DATAWIDTH; //9
constexpr int BLOCK_WEIGHT_TOTAL_IFC = PARALLEL_K * PARALLEL_N / (HP_IFC_BANDWIDTH / WEIGHTS_DATAWIDTH); //32 * 50 / 4 = 400 ifc blocks in total
constexpr int BLOCK_WEIGHT_URAM_ROW = ceildiv(PARALLEL_K * PARALLEL_N, WEIGHT_NUM_PER_ROW_URAM) ;//URAM rows needed for one block
constexpr int MAX_WEIGHT_NUM = 2048000; //2048 * 1000
constexpr int MAX_IACTS_NUM = 2048; //1*2048
constexpr int WEIGHT_PER_CYCLE = HP_IFC_BANDWIDTH / WEIGHTS_DATAWIDTH * NUM_HP_IFC; //24 weights per cycle
constexpr int IACT_PER_CYCLE = HP_IFC_BANDWIDTH / IACTS_DATAWIDTH * NUM_HP_IFC; //24 weights per cycle
constexpr int WEIGHT_PER_THREE_CYCLE = 3 * WEIGHT_PER_CYCLE; //72 weights
constexpr int BLOCK_WEIGHT_PER_THREE_CYCLE = 3 * WEIGHT_PER_CYCLE / WEIGHT_NUM_PER_ROW_URAM; //8 rows


constexpr int PARALLEL_IACTS_BANDWIDTH           = Parallel_IACT*IACTS_DATAWIDTH;
constexpr int PARALLEL_WEIGHT_BANDWIDTH           = Parallel_WEIGHT_PE*IACTS_DATAWIDTH;


//Systolic array parameters
constexpr int SYSTOLIC_DIM_X = Parallel_IACT;
constexpr int SYSTOLIC_DIM_Y = Parallel_WEIGHT_PE;
//Systolic array
constexpr int Parallel_IACT = 1;
constexpr int Parallel_WEIGHT_PE = 4;
constexpr int Total_PE = Parallel_IACT * Parallel_WEIGHT_PE;
constexpr int WEIGHT_ENTRY_PE = MAX_WEIGHT_ENTRY / Parallel_WEIGHT_PE;
constexpr int IACT_ENTRY_PE = MAX_IACTS_ENTRY / Parallel_IACT;

void LINEAR(
    ap_uint<HP_IFC_BANDWIDTH>               ifc1[MAX_IFC_ENTRY],
    ap_uint<HP_IFC_BANDWIDTH>               ifc2[MAX_IFC_ENTRY],
    ap_uint<HP_IFC_BANDWIDTH>               ifc3[MAX_IFC_ENTRY],
    ap_uint<HP_IFC_BANDWIDTH>               ifc4[MAX_IFC_ENTRY],
    ap_uint<HP_IFC_BANDWIDTH>               ifc5[MAX_IFC_ENTRY],
    ap_uint<HP_IFC_BANDWIDTH>               ifc6[MAX_IFC_ENTRY],
    int X, //Input Shape X
    int Y, //Input Shape Y
    int Wt_X, //Weight Shape X
    int Wt_Y,  //Weight Shape Y
    int bias
);