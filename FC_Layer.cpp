#include "FC_Layer.hpp"

void ReadFromMem(
    ap_uint<HP_IFC_BANDWIDTH> *ifc1,
    ap_uint<HP_IFC_BANDWIDTH> *ifc2,
    ap_uint<HP_IFC_BANDWIDTH> *ifc3,
    ap_uint<HP_IFC_BANDWIDTH> *ifc4,
    ap_uint<HP_IFC_BANDWIDTH> *ifc5,
    ap_uint<HP_IFC_BANDWIDTH> *ifc6,
    ap_uint<WEIGHT_URAM_WIDTH> weight_buffer[MAX_WEIGHT_URAM_ROW],
    hls::stream<ap_uint<WEIGHTS_DATAWIDTH>> weights_stream[Parallel_K],
    ap_uint<IACT_BRAM_WIDTH> iact_buffer[MAX_IACT_BRAM_ROW],
    hls::stream<ap_uint<IACTS_DATAWIDTH>> iacts_stream,
    int X,
    int Y,
    int Wt_X,
    int Wt_Y)
{
    // load weight from memory (2048 * 1000 in total) 24 nums per cycle -> cycle count = 2048*1000/24 cycles
    int weight_trip_count = Wt_X * Wt_Y / NUM_HP_IFC;                                     // 2048000/24/3
    int inner_trip_count = 3 * NUM_HP_IFC * HP_IFC_BANDWIDTH / sizeof(WEIGHTS_DATAWIDTH); // 24 -> load 3 cycles -> 72 nums in total
    int element_count = 0;
    int loop_count = inner_trip_count / WEIGHT_NUM_PER_ROW_URAM;
    for (int iter = 0; iter < weight_trip_count; iter = iter + 3)
    {
#pragma HLS loop_tripcount min = weight_trip_count max = weight_trip_count avg = weight_trip_count
#pragma HLS PIPELINE II = 4 // 3 for buffer writing
        if (iter > weight_trip_count)
            break;
        // cycle 1
        ap_uint<MAX_OFF_CHIP_BW> payload1 = 0;
        int addr_offset_1 = iter * NUM_HP_IFC;

        payload1.range(6 * HP_IFC_BANDWIDTH - 1, 5 * HP_IFC_BANDWIDTH) = ifc1[addr_offset_1 + 5];
        payload1.range(5 * HP_IFC_BANDWIDTH - 1, 4 * HP_IFC_BANDWIDTH) = ifc2[addr_offset_1 + 4];
        payload1.range(4 * HP_IFC_BANDWIDTH - 1, 3 * HP_IFC_BANDWIDTH) = ifc3[addr_offset_1 + 3];
        payload1.range(3 * HP_IFC_BANDWIDTH - 1, 2 * HP_IFC_BANDWIDTH) = ifc4[addr_offset_1 + 2];
        payload1.range(2 * HP_IFC_BANDWIDTH - 1, 1 * HP_IFC_BANDWIDTH) = ifc5[addr_offset_1 + 1];
        payload1.range(1 * HP_IFC_BANDWIDTH - 1, 0) = ifc6[addr_offset_1];

        // int addr_offset_4 = iter * inner_trip_count;
        // for (int i = 0; i < loop_count; i=i+WEIGHT_NUM_PER_ROW_URAM) {
        //     ap_uint<WEIGHT_URAM_WIDTH> weight_load = 0;
        //     int offset = i * WEIGHT_NUM_PER_ROW_URAM;
        //     for (int j = 0; j < WEIGHT_NUM_PER_ROW_URAM; ++j) {
        //         weight_load.range((j+1)*WEIGHTS_DATAWIDTH - 1, j*WEIGHTS_DATAWIDTH)
        //                     = payload1.range((offset+j+1)*WEIGHTS_DATAWIDTH-1, (offset+j)*WEIGHTS_DATAWIDTH);
        //     }
        // }

        // cycle 2
        ap_uint<MAX_OFF_CHIP_BW> payload2 = 0;
        int addr_offset_2 = (iter + 1) * NUM_HP_IFC;

        payload2.range(6 * HP_IFC_BANDWIDTH - 1, 5 * HP_IFC_BANDWIDTH) = ifc1[addr_offset_2 + 5];
        payload2.range(5 * HP_IFC_BANDWIDTH - 1, 4 * HP_IFC_BANDWIDTH) = ifc2[addr_offset_2 + 4];
        payload2.range(4 * HP_IFC_BANDWIDTH - 1, 3 * HP_IFC_BANDWIDTH) = ifc3[addr_offset_2 + 3];
        payload2.range(3 * HP_IFC_BANDWIDTH - 1, 2 * HP_IFC_BANDWIDTH) = ifc4[addr_offset_2 + 2];
        payload2.range(2 * HP_IFC_BANDWIDTH - 1, 1 * HP_IFC_BANDWIDTH) = ifc5[addr_offset_2 + 1];
        payload2.range(1 * HP_IFC_BANDWIDTH - 1, 0) = ifc6[addr_offset_2];

        // cycle 3
        ap_uint<MAX_OFF_CHIP_BW> payload3 = 0;
        int addr_offset_3 = (iter + 2) * NUM_HP_IFC;

        payload3.range(6 * HP_IFC_BANDWIDTH - 1, 5 * HP_IFC_BANDWIDTH) = ifc1[addr_offset_3 + 5];
        payload3.range(5 * HP_IFC_BANDWIDTH - 1, 4 * HP_IFC_BANDWIDTH) = ifc2[addr_offset_3 + 4];
        payload3.range(4 * HP_IFC_BANDWIDTH - 1, 3 * HP_IFC_BANDWIDTH) = ifc3[addr_offset_3 + 3];
        payload3.range(3 * HP_IFC_BANDWIDTH - 1, 2 * HP_IFC_BANDWIDTH) = ifc4[addr_offset_3 + 2];
        payload3.range(2 * HP_IFC_BANDWIDTH - 1, 1 * HP_IFC_BANDWIDTH) = ifc5[addr_offset_3 + 1];
        payload3.range(1 * HP_IFC_BANDWIDTH - 1, 0) = ifc6[addr_offset_3];

        // cycle4
        int offset = 0;
        int addr_offset_4 = iter * loop_count;
        ap_uint<3 *MAX_OFF_CHIP_BW> combine_payload = 0;
        combine_payload.range(MAX_OFF_CHIP_BW - 1, 0) = payload1;
        combine_payload.range(2 * MAX_OFF_CHIP_BW - 1, MAX_OFF_CHIP_BW) = payload2;
        combine_payload.range(3 * MAX_OFF_CHIP_BW - 1, 2 * MAX_OFF_CHIP_BW) = payload3;
        while (idx < loop_count)
        {
            ap_uint<WEIGHT_URAM_WIDTH> weight_load = 0;
            int offset = idx + WEIGHT_NUM_PER_ROW_URAM;
            for (int i = 0; i < WEIGHT_NUM_PER_ROW_URAM; ++i)
            {
                weight_load.range((i + 1) * WEIGHTS_DATAWIDTH - 1, i * WEIGHTS_DATAWIDTH) = combine_payload.range((offset + i + 1) * WEIGHTS_DATAWIDTH - 1, (offset + i) * WEIGHTS_DATAWIDTH);
            }
            weight_buffer[idx + addr_offset_4] = weight_load.range(WEIGHT_URAM_WIDTH - 1, 0);
            idx++;
        }
        // for (int i = 0; i < inner_trip_count; ++i) {
        //     weight_buffer[i+addr_offset_4] = payload.range((i+1)*WEIGHTS_DATAWIDTH-1, i*WEIGHTS_DATAWIDTH);
        // }
    }

    // read rest weight values
    int offset = (Wt_X * Wt_Y) / inner_trip_count / 4;
    int num_HPC_needed = (Wt_X * Wt_Y) % inner_trip_count / 4;
    ap_uint<HP_IFC_BANDWIDTH *num_HPC_needed> temp = 0;
    for (int i = 0; i < num_HPC_needed; ++i)
    { // one hpc loads 4 nums at once-> compute how many hpcs we need
        ap_uint<HP_IFC_BANDWIDTH> payload = ifc[offset + i];
        for (int j = 0; j < 4; ++j)
        {
            temp.range((i * 4 + j + 1) * WEIGHTS_DATAWIDTH - 1, (i * 4 + j) * WEIGHTS_DATAWIDTH) = payload.range((j + 1) * WEIGHTS_DATAWIDTH - 1, j * WEIGHTS_DATAWIDTH);
        }
    }

    // load input activation value
    int iact_trip_count = X * Y / NUM_HP_IFC;
    int inner_trip_count = NUM_HP_IFC * HP_IFC_BANDWIDTH / sizeof(IACTS_DATAWIDTH);
    for (int iter = 0; iter < iact_trip_count; iter = iter++)
    {
#pragma HLS loop_tripcount min = iact_trip_count max = iact_trip_count avg = iact_trip_count
#pragma HLS PIPELINE II = 1
        ap_uint<MAX_OFF_CHIP_BW> payload = 0;
        int addr_offset_1 = iter * NUM_HP_IFC;
        int addr_offset_2 = iter * inner_trip_count;

        payload1.range(6 * HP_IFC_BANDWIDTH - 1, 5 * HP_IFC_BANDWIDTH) = ifc1[addr_offset_1 + 5];
        payload1.range(5 * HP_IFC_BANDWIDTH - 1, 4 * HP_IFC_BANDWIDTH) = ifc2[addr_offset_1 + 4];
        payload1.range(4 * HP_IFC_BANDWIDTH - 1, 3 * HP_IFC_BANDWIDTH) = ifc3[addr_offset_1 + 3];
        payload1.range(3 * HP_IFC_BANDWIDTH - 1, 2 * HP_IFC_BANDWIDTH) = ifc4[addr_offset_1 + 2];
        payload1.range(2 * HP_IFC_BANDWIDTH - 1, 1 * HP_IFC_BANDWIDTH) = ifc5[addr_offset_1 + 1];
        payload1.range(1 * HP_IFC_BANDWIDTH - 1, 0) = ifc6[addr_offset_1];

        for (int i = 0; i < inner_trip_count; ++i)
        {
            iact_buffer[i + addr_offset_2] = payload.range((i + 1) * IACTS_DATAWIDTH - 1, i * IACTS_DATAWIDTH);
        }
    }

    // stream the iact and weight values
    for (int i = 0; i < X; ++i)
    {
        int trip_count = Y % Parallel_IACT == 0 ? (Y / Parallel_IACT) : (Y / Parallel_IACT + 1);
        for (int j = 0; j < trip_count; ++j)
        {
#pragma HLS loop_tripcount min = trip_count max = trip_count avg = trip_count
#pragma HLS PIPELINE II = 1
            int idx = i * Y + j;
            // ap_uint<PARALLEL_IACTS_BANDWIDTH> payload; Can reformat the buffer later to read in several number
            for (int k = 0; k < Parallel_IACT; ++k)
            {
                if (idx + k > MAX_IACTS_ENTRY)
                    break;
                iacts_stream[k] = iact_buffer[idx + k];
            }
        }
    }

    for (int i = 0; i < Wt_X; ++i)
    {
        int trip_count = Wt_Y % Parallel_WEIGHT_PE == 0 ? (Wt_Y / Parallel_WEIGHT_PE) : (Wt_Y / Parallel_WEIGHT_PE + 1);
        for (int j = 0; j < trip_count; ++j)
        {
#pragma HLS loop_tripcount min = trip_count max = trip_count avg = trip_count
#pragma HLS PIPELINE II = 1
            int idx = i * Y + j;
            // ap_uint<PARALLEL_IACTS_BANDWIDTH> payload; Can reformat the buffer later to read in several number
            for (int k = 0; k < Parallel_WEIGHT_PE; ++k)
            {
                if (idx + k > MAX_WEIGHT_ENTRY)
                    break;
                weights_stream[k] = weight_buffer[idx + k];
            }
        }
    }

    // read 640 numbers to weights_stream
    int block_num_x = Wt_X / PARALLEL_K;
    int block_num_y = Wt_Y / PARALLEL_N;
    for (int block_x = 0; block_x < block_num_x; ++block_x)
    {
        for (int block_y = 0; block_y < block_num_y; ++block_y)
        {
            for (int i = 0; i < Parallel_K; ++i)
            {
                for (int j = 0; j < PARALLEL_N; ++j)
                {
                    int idx_x = (block_x * PARALLEL_K * PARALLEL_N + block_y * PARALLEL_N + i * PARALLEL_N + j) / WEIGHT_NUM_PER_ROW_URAM;
                    int idx_y = (block_x * PARALLEL_K * PARALLEL_N + block_y * PARALLEL_N + i * PARALLEL_N + j) % WEIGHT_NUM_PER_ROW_URAM;
                    weights_stream[k].write(weight_buffer[idx_x].range((idx_y + 1) * WEIGHTS_DATAWIDTH - 1, idx_y * WEIGHTS_DATAWIDTH));
                }
            }
        }
    }
}

//Use double buffer for this function
void CreateBitMask(hls::stream<ap_uint<WEIGHTS_DATAWIDTH>> weights_stream[Parallel_K], ap_uint<PARALLEL_N * WEIGHTS_DATAWIDTH> processing_buffer[PARALLEL_K],
            ap_uint<PARALLEL_N> bit_buffer_weights[PARALLEL_K])
{
    if (weights_stream.empty()) {
        return;
    }
    //change loop order to smooth the pipeline
    for (int i = 0; i < PARALLEL_K; ++i)
    { 
        ap_uint<PARALLEL_N*WEIGHTS_DATAWIDTH> payload = 0;
        ap_uint<PARALLEL_N> bitmask = 0;
        for (int j = 0; j < PARALLEL_N) {
#pragma HLS PIPELINE II = 1
            ap_uint<WEIGHTS_DATAWIDTH> data = weights_stream[i].read();
            payload.range((j+1)*WEIGHTS_DATAWIDTH-1, j*WEIGHTS_DATAWIDTH) = data;
            bitmask.range(j+1, j) = data == 0? 0 : 1;
        }
        processing_buffer[i] = payload;
        bit_buffer_weights[i] = bitmask;
    }
}

void DPEUnit(ap_uint<IACTS_DATAWIDTH> iact_value, int iact_idx, ap_uint<PARALLEL_N * WEIGHTS_DATAWIDTH> processing_buffer[PARALLEL_K],
                    ap_uint<PARALLEL_N> bit_buffer_weights[PARALLEL_K], ap_uint<OACTS_DATAWIDTH> output_buf[PARALLEL_K][PARALLEL_N], int k_idx) {
    //Compute output indices
    if (iact_idx == -1) return;
    ap_uint<PARALLEL_N> bitmask = bit_buffer_weights[iact_idx];
    ap_uint<PARALLEL_N * WEIGHTS_DATAWIDTH> weight_row = processing_buffer[iact_idx];
    for (int i = 0; i < PARALLEL_N; ++i) {
#pragma HLS UNROLL
        ap_uint<1> bit = bitmask.range(i+1, i);
        if (bit == 1) {
            output_buf[iact_idx][i] += weight_row.range(((i+1)*WEIGHTS_DATAWIDTH-1, i*WEIGHTS_DATAWIDTH) * iact_value;
        }

        if (k_idx != 0) {
            output_buf[k_idx][i] += output_buf[k_idx][i-1];
        }
    }
}

inline void load_iact_data(ap_uint<IACTS_DATAWIDTH*PARALLEL_K> data_buffer, hls::stream<ap_uint<IACTS_DATAWIDTH>> iacts_stream) {
    for (int i = 0; i < PARALLEL_K; ++i) {
#pragma HLS PIPELINE
        data_buffer.range((i+1)*IACTS_DATAWIDTH-1, i*IACTS_DATAWIDTH) = iacts_stream.read();
    }
}

//batch_num here is for recording how many groups of PARALLEL_K we have processed
void DPEComputation(hls::stream<ap_uint<IACTS_DATAWIDTH>> iacts_stream, int batch_num, ap_uint<PARALLEL_N * WEIGHTS_DATAWIDTH> processing_buffer[PARALLEL_K],
                    ap_uint<PARALLEL_N> bit_buffer_weights[PARALLEL_K],  ap_uint<OACTS_DATAWIDTH> buffer_out[PARALLEL_N]) {
    //broadcast nonzero iact values
    ap_uint<IACTS_DATAWIDTH> iact_value = iacts_stream.read();
    int base_idx = batch_num * PARALLEL_K;
    int broadcast_idx = 0;
    ap_uint<OACTS_DATAWIDTH> local_output_buf[PARALLEL_K][PARALLEL_N]; //should be a global value
    // ap_uint<IACTS_DATAWIDTH*PARALLEL_K> first_data_buffer = 0;
    // ap_uint<IACTS_DATAWIDTH*PARALLEL_K> second_data_buffer = 0;
    // if (cycle_i % 2 == 0) {
    //     load_iact_data(first_data_buffer, iacts_stream);
    // } else {
    //     load_iact_data(second_data_buffer, iacts_stream);
    // }
    for (int i = 0; i < PARALLEL_K; ++i) {
#pragma HLS PIPELINE
        broadcast_idx = iact_value == 0 ? -1:(base_idx+i)%PARALLEL_K;
        DPEUnit(data, broadcast_idx, processing_buffer, bit_buffer_weights, local_output_buf);
    }
    for (int i = 0; i < PARALLEL_N; ++i) {
        buffer_out[i] = local_output_buf[PARALLEL_K-1][i];
    }
}

void compute_systolic(hls::stream<ap_int<IACTS_DATAWIDTH>> iacts_stream[Parallel_IACT], hls::stream<ap_int<IACTS_DATAWIDTH>> weights_stream[Parallel_WEIGHT_PE],
                      data_t bias, hls::stream<ap_int<OACTS_DATAWIDTH>> output_stream[Parallel_IACT], int X, int Y, int Wt_X, int Wt_Y)
{
    assert(Y == Wt_X);
#pragma HLS BIND_STORAGE variable = oact_buffer type = ram_t2p impl = bram latency = 1
#pragma HLS array_partition variable = oact_buffer type = cyclic factor = Parallel_IACT dim = 0
#pragma HLS array_partition variable = oact_buffer type = cyclic factor = Parallel_WEIGHT_PE dim = 1
    size_t col_size = Wt_X;
    size_t max_iter = X + Wt_Y - 2;
    size_t max_entry = X * Wt_Y;
    data_t local_out[X][Wt_Y];
    for (int k = 0; k < col_size + max_iter; ++k)
    {
        for (int i0 = X / SYSTOLIC_DIM_X; i0 > 0; --i0)
        {
            for (int j0 = Wt_Y / SYSTOLIC_DIM_Y; j0 > 0; --j0)
            {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
                for (int i = i0 * SYSTOLIC_DIM_X - 1; i >= (i0 - 1) * SYSTOLIC_DIM_X; --i)
                {
#pragma HLS UNROLL factor = SYSTOLIC_DIM_X
                    for (int j = j0 * SYSTOLIC_DIM_Y - 1; j >= (j0 - 1) * SYSTOLIC_DIM_Y; --j)
                    {
#pragma HLS UNROLL factor = SYSTOLIC_DIM_Y
                        int max = X > Wt_Y ? std::max(X, Y) : std::max(Wt_Y, Y);
                        if (k - i - j >= 0 && k - i - j <= max - 1)
                        {
                            data_t a_val = input[i][k - i - j];
                            data_t b_val = weight[k - i - j][j];
                            std::cout << "i: " << a_val << " j: " << b_val << std::endl;
                            data_t last = (k - i - j == 0) ? (data_t)bias : local_out[i][j];
                            data_t val;
#pragma HLS BIND_OP variable = val op = mul impl = dsp latency = -1
                            val = last + a_val * b_val;
                            local_out[i][j] = val;
                        }
                    }
                }
            }
        }
    }
    for (int j = Wt_Y - 1; j >= Wt_Y - 1 - idx; --j)
    {
        int count = X % Parallel_IACT == 0 ? (X / Parallel_IACT - 1) : (X / Parallel_IACT);
        for (int i = count; i >= 0; --i)
        {
            int idx = i * Parallel_IACT;
            for (int k = 0; k < Parallel_IACT; ++k)
            {
                if (idx + k > max_entry)
                    break;
                if (j == Wt_Y - 1)
                    output_stream[k] = local_out[idx + k][j];
                local_out[idx + k][j] = local_out[idx + k][j - 1];
            }
        }
    }
    //     for (int idx = Wt_X-1; idx >= 0; --idx) {
    //         for (int j = Wt_Y - 1; j >= Wt_Y-1-idx; --j) {
    // #pragma HLS PIPELINE II=1
    //             for (int i = X - 1; i >= 0; --i) {
    //                 if (j == Wt_Y-1) output[i][idx] = local_out[i][j];
    //                 local_out[i][j] = local_out[i][j-1];
    //             }
    //         }
    //     }
}

void OutputBuffer(
    ap_uint<HP_IFC_BANDWIDTH> *oacts_ifc,
    hls::stream<ap_int<OACTS_DATAWIDTH>> output_stream[Parallel_IACT],
    int X,
    int Wt_Y)
{
    ap_uint<HP_IFC_BANDWIDTH> results;
    int overall_addr = 0;
    int loop_count = X * Wt_Y;
    int counter = 4;
    assert(loop_count % Parallel_IACT == 0) for (int i = 0; i < loop_count / Parallel_IACT; ++i)
    {
        for (int idx = 0; idx < Parallel_IACT; ++idx)
        {
            results.range(32 * (counter - 1), 32 * counter - 1) = output_stream[idx];
            if (--counter == 0)
            {
                oacts_ifc[overall_addr++] = results;
                counter = 4;
            }
        }
    }
}

void LINEAR(
    ap_uint<HP_IFC_BANDWIDTH> ifc1[MAX_IFC_ENTRY],
    ap_uint<HP_IFC_BANDWIDTH> ifc2[MAX_IFC_ENTRY],
    ap_uint<HP_IFC_BANDWIDTH> ifc3[MAX_IFC_ENTRY],
    ap_uint<HP_IFC_BANDWIDTH> ifc4[MAX_IFC_ENTRY],
    ap_uint<HP_IFC_BANDWIDTH> ifc5[MAX_IFC_ENTRY],
    ap_uint<HP_IFC_BANDWIDTH> ifc6[MAX_IFC_ENTRY],
    ap_uint<HP_IFC_BANDWIDTH> ifc7[MAX_IFC_ENTRY],
    int X, // Input Shape X
    int Y,                                               // Input Shape Y
    int Wt_X,                                            // Weight Shape X
    int Wt_Y,                                            // Weight Shape Y
    int bias)
{
#define FINAL_DIM0 X
#define FINAL_DIM1 Y
#pragma HLS INTERFACE m_axi depth = MAX_IFC_ENTRY offset = slave port = ifc1 bundle = ifc1
#pragma HLS INTERFACE m_axi depth = MAX_IFC_ENTRY offset = slave port = ifc1 bundle = ifc1
#pragma HLS INTERFACE m_axi depth = MAX_IFC_ENTRY offset = slave port = ifc2 bundle = ifc2
#pragma HLS INTERFACE m_axi depth = MAX_IFC_ENTRY offset = slave port = ifc3 bundle = ifc3
#pragma HLS INTERFACE m_axi depth = MAX_IFC_ENTRY offset = slave port = ifc4 bundle = ifc4
#pragma HLS INTERFACE m_axi depth = MAX_IFC_ENTRY offset = slave port = ifc5 bundle = ifc5
#pragma HLS INTERFACE m_axi depth = MAX_IFC_ENTRY offset = slave port = ifc6 bundle = ifc6
#pragma HLS INTERFACE m_axi depth = MAX_IFC_ENTRY offset = slave port = ifc7 bundle = ifc1

#pragma HLS INTERFACE s_axilite port = X
#pragma HLS INTERFACE s_axilite port = Y
#pragma HLS INTERFACE s_axilite port = Wt_X
#pragma HLS INTERFACE s_axilite port = Wt_Y
#pragma HLS INTERFACE s_axilite port = bias

    // assign 48 urams to weight buffer and apply loop tiling
    // uram size: 72 * 4096 bits
    // weight_datawidth: 32 bits
    // uram dimension: 12 * 4
    // 9 weights per row, 4096 rows in total
    ap_uint<WEIGHT_URAM_WIDTH> weight_buffer[MAX_WEIGHT_URAM_ROW]; // need 240 urams in total (layout 60 * 4)  Each row stores 9 weight numbers.
#pragma HLS BIND_STORAGE variable = weight_buffer type = ram_t2p impl = uram latency = 1
#pragma HLS array_partition variable = weight_buffer type = cyclic factor = READ_PARALLEL_WEIGHT dim = 0

    ap_uint<PARALLEL_N * WEIGHTS_DATAWIDTH> first_processing_buffer[PARALLEL_K];
    ap_uint<PARALLEL_N * WEIGHTS_DATAWIDTH> second_processing_buffer[PARALLEL_K];

    ap_uint<PARALLEL_N> first_bit_buffer_weights[PARALLEL_K];
    ap_uint<PARALLEL_N> second_bit_buffer_weights[PARALLEL_K];
#pragma HLS BIND_STORAGE variable = first_processing_buffer type = ram_t2p impl = bram latency = 1  // 4 brams -> 128 * 128 * 1 + 32 * 512 * 4
#pragma HLS BIND_STORAGE variable = second_processing_buffer type = ram_t2p impl = bram latency = 1 // 160 * 20 as one blocks

    ap_uint<IACT_BRAM_WIDTH> iact_buffer[MAX_IACT_BRAM_ROW]; // need 4 brams (8 * 2k) in total
#pragma HLS BIND_STORAGE variable = iact_buffer type = ram_t2p impl = bram latency = 1
#pragma HLS array_partition variable = iact_buffer type = cyclic factor = Parallel_K dim = 1 // read 32 elements at one time

    hls::stream<ap_uint<IACTS_DATAWIDTH>> iacts_stream;
#pragma HLS STREAM variable = iacts_stream depth = PARALLEL_K type = fifo

    hls::stream<ap_uint<WEIGHTS_DATAWIDTH>> weights_stream[Parallel_K];
#pragma HLS STREAM variable = weights_stream depth = PARALLEL_N type = fifo

    hls::stream<ap_uint<OACTS_DATAWIDTH>> output_stream[Parallel_IACT];
#pragma HLS STREAM variable = output_stream depth = 100 type = fifo

    ap_uint<OACTS_DATAWIDTH> output_buf[Y/PARALLEL_N][PARALLEL_N];
#pragma HLS BIND_STORAGE variable = iact_buffer type = ram_t2p impl = bram latency = 1
#pragma HLS array_partition variable = iact_buffer type = complete dim = 1 // read 32 elements at one time

    int total_batch = X * Y / PARALLEL_K / PARALLEL_N;

#pragma HLS DATAFLOW
    ReadFromMem(ifc1, ifc2, ifc3, ifc4, ifc5, ifc6, ifc7, weight_buffer, weights_stream, iact_buffer, iacts_stream, X, Y, Wt_X, Wt_Y);
    //compute_systolic(iacts_stream, weights_stream, bias, output_stream, X, Y, Wt_X, Wt_Y);
    CreateBitMask(weight_stream, first_processing_buffer, first_bit_buffer_weights);
    for (int i = 0; i < total_batch; ++i) { 
        if (i != total_batch - 1) {
            if (i % 2== 0) {
                CreateBitMask(weight_stream, second_processing_buffer, second_bit_buffer_weights);
                DPEComputation(iacts_stream, i, first_processing_buffer, first_bit_buffer_weights, output_buf[i]);
            } else {
                CreateBitMask(weight_stream, first_processing_buffer, first_bit_buffer_weights);
                DPEComputation(iacts_stream, i, second_processing_buffer, second_bit_buffer_weights, output_buf[i]);
            }
        }
        else {
            DPEComputation(iacts_stream, i, second_processing_buffer, second_bit_buffer_weights, output_buf[i]); //depends on batch_num
        }
    }
    OutputBuffer(ifc7, output_stream, X, Wt_Y, output_buf);
}


void DPEClean(ap_uint<OACTS_DATAWIDTH> local_output_buf[PARALLEL_K][PARALLEL_N], ap_uint<OACTS_DATAWIDTH> buffer_out[PARALLEL_N]) {
    for (int i = 0; i < PARALLEL_N; ++i) {
#pragma HLS PIPELINE II=PARALLEL_K+1
        for (int j = 1; j < PARALLEL_K; ++j) {
            local_output_buf[j][i] += local_output_buf[j-1][i];
        }
        buffer_out[i] = local_output_buf[PARALLEL_K-1][i];
    }
}