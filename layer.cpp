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
    int block_count = Wt_X * Wt_Y / PARALLEL_K / PARALLEL_N; //compute how many blocks the program needs to load
    int inner_loop_count = ceildiv(PARALLEL_K*PARALLEL_N, 3*MAX_OFF_CHIP_BW) - 1; //how many cycles needed to read one block weight value
    int max_uram_row = ceildiv(PARALLEL_K*PARALLEL_N, MAX_WEIGHT_URAM_ROW); // compute for each block, how many lines of data it needs from URAM

    int weight_offset = 0;
    //compute what is the residual after burst read
    int residual = PARALLEL_K * PARALLEL_N - inner_loop_count*72; //72 weights read in 3 cycles
    for (size_t i = 0; i < block_count; ++i) { //iterate all blocks
#pragma HLS loop_tripcount min = block_count max = block_count avg = block_count    

        for (size_t j = 0; j < inner_loop_count; ++j) { //iterate cycles needed for one block transfer
#pragma HLS loop_tripcount min = inner_loop_count max = inner_loop_count avg = inner_loop_count 
#pragma HLS Pipeline II=4

            //cycle1
            ap_uint<MAX_OFF_CHIP_BW> payload1 = 0;
            
            int addr_offset = i * BLOCK_WEIGHT_TOTAL_IFC;
            int addr_offset_1 = addr_offset + j * NUM_HP_IFC;
            payload1.range(6 * HP_IFC_BANDWIDTH - 1, 5 * HP_IFC_BANDWIDTH) = ifc1[addr_offset_1 + 5];
            payload1.range(5 * HP_IFC_BANDWIDTH - 1, 4 * HP_IFC_BANDWIDTH) = ifc2[addr_offset_1 + 4];
            payload1.range(4 * HP_IFC_BANDWIDTH - 1, 3 * HP_IFC_BANDWIDTH) = ifc3[addr_offset_1 + 3];
            payload1.range(3 * HP_IFC_BANDWIDTH - 1, 2 * HP_IFC_BANDWIDTH) = ifc4[addr_offset_1 + 2];
            payload1.range(2 * HP_IFC_BANDWIDTH - 1, 1 * HP_IFC_BANDWIDTH) = ifc5[addr_offset_1 + 1];
            payload1.range(1 * HP_IFC_BANDWIDTH - 1, 0) = ifc6[addr_offset_1];

            //cycle2

            ap_uint<MAX_OFF_CHIP_BW> payload2 = 0;
            int addr_offset_2 = addr_offset + (j + 1) * NUM_HP_IFC;

            payload2.range(6 * HP_IFC_BANDWIDTH - 1, 5 * HP_IFC_BANDWIDTH) = ifc1[addr_offset_2 + 5];
            payload2.range(5 * HP_IFC_BANDWIDTH - 1, 4 * HP_IFC_BANDWIDTH) = ifc2[addr_offset_2 + 4];
            payload2.range(4 * HP_IFC_BANDWIDTH - 1, 3 * HP_IFC_BANDWIDTH) = ifc3[addr_offset_2 + 3];
            payload2.range(3 * HP_IFC_BANDWIDTH - 1, 2 * HP_IFC_BANDWIDTH) = ifc4[addr_offset_2 + 2];
            payload2.range(2 * HP_IFC_BANDWIDTH - 1, 1 * HP_IFC_BANDWIDTH) = ifc5[addr_offset_2 + 1];
            payload2.range(1 * HP_IFC_BANDWIDTH - 1, 0) = ifc6[addr_offset_2];

            // cycle 3
            ap_uint<MAX_OFF_CHIP_BW> payload3 = 0;
            int addr_offset_3 = addr_offset+(j + 2) * NUM_HP_IFC;

            payload3.range(6 * HP_IFC_BANDWIDTH - 1, 5 * HP_IFC_BANDWIDTH) = ifc1[addr_offset_3 + 5];
            payload3.range(5 * HP_IFC_BANDWIDTH - 1, 4 * HP_IFC_BANDWIDTH) = ifc2[addr_offset_3 + 4];
            payload3.range(4 * HP_IFC_BANDWIDTH - 1, 3 * HP_IFC_BANDWIDTH) = ifc3[addr_offset_3 + 3];
            payload3.range(3 * HP_IFC_BANDWIDTH - 1, 2 * HP_IFC_BANDWIDTH) = ifc4[addr_offset_3 + 2];
            payload3.range(2 * HP_IFC_BANDWIDTH - 1, 1 * HP_IFC_BANDWIDTH) = ifc5[addr_offset_3 + 1];
            payload3.range(1 * HP_IFC_BANDWIDTH - 1, 0) = ifc6[addr_offset_3];

            //WEIGHT_BUFFER load data
            int addr_offset_4 = BLOCK_WEIGHT_URAM_ROW * i + j * BLOCK_WEIGHT_PER_THREE_CYCLE;
            ap_uint<3 *MAX_OFF_CHIP_BW> combine_payload = 0;
            combine_payload.range(MAX_OFF_CHIP_BW - 1, 0) = payload1;
            combine_payload.range(2 * MAX_OFF_CHIP_BW - 1, MAX_OFF_CHIP_BW) = payload2;
            combine_payload.range(3 * MAX_OFF_CHIP_BW - 1, 2 * MAX_OFF_CHIP_BW) = payload3;
            for (int idx_count = 0; idx_count < WEIGHT_PER_THREE_CYCLE; idx_count+=9) {
                ap_uint<WEIGHT_URAM_WIDTH> weight_load = 0;
                int offset = idx_count + WEIGHT_NUM_PER_ROW_URAM;
                for (int i = 0; i < WEIGHT_NUM_PER_ROW_URAM; ++i)
                {
                    weight_load.range((i + 1) * WEIGHTS_DATAWIDTH - 1, i * WEIGHTS_DATAWIDTH) = combine_payload.range((idx_count + i + 1) * WEIGHTS_DATAWIDTH - 1, (idx_count + i) * WEIGHTS_DATAWIDTH);
                }
                weight_buffer[addr_offset_4++] = weight_load.range(WEIGHT_URAM_WIDTH - 1, 0);
            }
        }

        //process residual data
        //compute how many cycles needed
        int count = 0;
        int full_cycles = ceildiv(residual, WEIGHT_PER_CYCLE);
        ap_uint<3*MAX_OFF_CHIP_BW> payload = 0;
        for (size_t i = 0; i < full_cycles; ++i) {
            ap_uint<MAX_OFF_CHIP_BW> temp = 0;
            int addr_offset_1 = BLOCK_WEIGHT_URAM_ROW * i + WEIGHT_PER_THREE_CYCLE * inner_loop_count + i * WEIGHT_NUM_PER_CYCLE;
            for (int j = 0; j < 6; ++j) {
                if (count >= residual) break;
                temp.range((j+1) * HP_IFC_BANDWIDTH - 1, j * HP_IFC_BANDWIDTH) = ifc1[addr_offset_1 + j];
                count++
            }
            payload.range((i+1)*MAX_OFF_CHIP_BW-1, i*MAX_OFF_CHIP_BW) = temp;
        }
        int weight_rows = ceildiv(residual, WEIGHT_NUM_PER_ROW_URAM);
        int offset = WEIGHTS_DATAWIDTH * WEIGHT_NUM_PER_ROW_URAM;
        int addr_offset = BLOCK_WEIGHT_URAM_ROW * i + inner_loop_count * BLOCK_WEIGHT_PER_THREE_CYCLE;
        for (int i = 0; i < weight_rows; ++i) {
        #pragma HLS UNROLL
            ap_uint<WEIGHT_URAM_WIDTH> weight_load = 0;
            if (i != weight_rows - 1) {
                weight_load = payload.range((i+1)*offset-1, i*offset); 
            }
            else {
                int w_offset = residual - i * WEIGHT_NUM_PER_ROW_URAM;
                weight_load.range(w_offset*WEIGHTS_DATAWIDTH-1, 0) = payload.range(i*offset+w_offset*WEIGHTS_DATAWIDTH-1, i*offset);
            }
            weight_buffer[addr_offset++] = weight_load.range(WEIGHT_URAM_WIDTH - 1, 0);
        }
    }

    int iact_count = ceildiv(X*Y, IACT_PER_CYCLE) - 1;
    int residual = X*Y - iact_count * IACT_PER_CYCLE;
    for (int i = 0; i < iact_count; ++i) {
        ap_uint<MAX_OFF_CHIP_BW> payload = 0;
        int addr_offset = BLOCK_WEIGHT_TOTAL_IFC + i * NUM_HP_IFC;
        for (int j = 0; j < NUM_HP_IFC; ++j) {
            payload.range((j+1) * HP_IFC_BANDWIDTH - 1, j*HP_IFC_BANDWIDTH) = ifc6[addr_offset_+j];
        }
        int buffer_offset = i * IACT_PER_CYCLE;
        for (int j = 0; j < IACT_PER_CYCLE; ++j) {
            iact_buffer[buffer_offset+j] = payload.range((j+1) * IACTS_DATAWIDT-1, j*IACTS_DATAWIDTH);
        }
    } 
    //process remaining data -- less than one cycle
    for (int i = 0; i < 1; ++i) {
        ap_uint<MAX_OFF_CHIP_BW> payload = 0;
        int addr_offset = BLOCK_WEIGHT_TOTAL_IFC + iact_count * NUM_HP_IFC;
        for (int j = 0; j < NUM_HP_IFC; ++j) {
            if (j * 4 > residual) break;
            payload.range((j+1) * HP_IFC_BANDWIDTH - 1, j*HP_IFC_BANDWIDTH) = ifc6[addr_offset_+j];
        }
        for (int j = 0; j < residual; ++j) {
            iact_buffer[buffer_offset+j] = payload.range((j+1) * IACTS_DATAWIDT-1, j*IACTS_DATAWIDTH);
        }
    }

    // read 640 numbers to weights_stream
    int block_num_x = Wt_X / PARALLEL_K;
    int block_num_y = Wt_Y / PARALLEL_N;
    for (int block_x = 0; block_x < block_num_x; ++block_x)
    {
        for (int block_y = 0; block_y < block_num_y; ++block_y)
        {
            int base_offset = (block_x * block_num_y + block_y) * BLOCK_WEIGHT_URAM_ROW;
            for (int i = 0; i < Parallel_K; ++i)
            {
                for (int j = 0; j < PARALLEL_N; ++j)
                {
                #pragma HLS UNROLL
                    // int idx_x = (block_x * PARALLEL_K * PARALLEL_N + block_y * PARALLEL_N + i * PARALLEL_N + j) / WEIGHT_NUM_PER_ROW_URAM;
                    // int idx_y = (block_x * PARALLEL_K * PARALLEL_N + block_y * PARALLEL_N + i * PARALLEL_N + j) % WEIGHT_NUM_PER_ROW_URAM;
                    int idx_x = (i * PARALLEL_N + j) / WEIGHT_NUM_PER_ROW_URAM;
                    int idx_y = (i * PARALLEL_N + j) % WEIGHT_NUM_PER_ROW_URAM;
                    weights_stream[i].write(weight_buffer[base_offset+idx_x].range((base_offset+idx_y + 1) * WEIGHTS_DATAWIDTH - 1, (base_offset+idx_y) * WEIGHTS_DATAWIDTH));
                }
            }
        }
    }

    //stream the iact values
    for (int i = 0; i < X*Y; ++i) {
        iacts_stream.write(iact_buffer[i].range(IACT_BRAM_WIDTH-1, 0)); //same size for now
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

//batch_num here is for recording how many groups of PARALLEL_K we have processed
void DPEComputation(ap_uint<IACTS_DATAWIDTH> IACT_TEMP_BUFFER[PARALLEL_K], int block_idx_x, int block_idx_y,  ap_uint<PARALLEL_N * WEIGHTS_DATAWIDTH> processing_buffer[PARALLEL_K],
                    ap_uint<PARALLEL_N> bit_buffer_weights[PARALLEL_K],  ap_uint<OACTS_DATAWIDTH> buffer_out[Wt_Y/PARALLEL_N][PARALLEL_N], int Wt_X, int Wt_Y {
    //broadcast nonzero iact values
    int broadcast_idx = 0;
    ap_uint<OACTS_DATAWIDTH> local_output_buf[PARALLEL_K][PARALLEL_N]; //should be a global value
    // propagate the first row of local_out_buffer
    if (block_idx_x == 0) {
        for (int i = 0; i < PARALLEL_N; ++i) {
            local_output_buf[0][i] = 0;
        }
    } else {
        for (int i = 0; i < PARALLEL_N; ++i) {
            local_output_buf[0][i] = buffer_out[block_idx_y][i];
        }
    }
    for (int i = 0; i < PARALLEL_K; ++i) {
#pragma HLS PIPELINE
        broadcast_idx = iact_value[i] == 0 ? -1:i;
        DPEUnit(iact_value[i], broadcast_idx, processing_buffer, bit_buffer_weights, local_output_buf);
    }
    for (int i = 0; i < PARALLEL_N; ++i) {
        buffer_out[batch_num][i] = local_output_buf[PARALLEL_K-1][i];
    }
}

void OutputBuffer(
    ap_uint<HP_IFC_BANDWIDTH> *oacts_ifc,
    hls::stream<ap_int<OACTS_DATAWIDTH>> output_stream,
    int X,
    int Wt_Y,
    int address,
    ap_uint<OACTS_DATAWIDTH> output_buf[Wt_Y/PARALLEL_N][PARALLEL_N])
{
    int overall_addr = address;
    int loop_count = X * Wt_Y;
    for (int i = 0; i < loop_count / Parallel_N; ++i)
    {
        for (int idx = 0; idx < PARALLEL_N; ++idx)
        {
            output_stream.write(output_buf[i][idx]);
        }
    }

    for (int i = 0; i < Wt_Y * X / (HP_IFC_BANDWIDTH / OACTS_DATAWIDTH); ++i) {
        ap_uint<HP_IFC_BANDWIDTH> payload = 0;
        for (int j = 0; j < HP_IFC_BANDWIDTH / OACTS_DATAWIDTH; ++j) {
            payload.range((j+1) * OACTS_DATAWIDTH-1, j*OACTS_DATAWIDTH) = outputstream.read();
        }
        oacts_ifc[overall_address++] = payload;
    }
}

inline void ReadIactBuff(hls::stream<ap_uint<IACTS_DATAWIDTH>> iacts_stream, ap_uint<IACTS_DATAWIDTH> IACT_TEMP_BUFFER[PARALLEL_K]) {
    for (int i = 0; i < PARALLEL_K; ++i) {
        IACT_TEMP_BUFFER[i] = iacts_stream.read();
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
#pragma HLS BIND_STORAGE variable = first_processing_buffer type = ram_t2p impl = bram latency = 1  // 4 brams -> 128 * 128 * 1 + 32 * 512 * 4
#pragma HLS BIND_STORAGE variable = second_processing_buffer type = ram_t2p impl = bram latency = 1 // 160 * 20 as one blocks

    ap_uint<PARALLEL_N> first_bit_buffer_weights[PARALLEL_K];
    ap_uint<PARALLEL_N> second_bit_buffer_weights[PARALLEL_K];

    ap_uint<IACT_BRAM_WIDTH> iact_buffer[MAX_IACT_BRAM_ROW]; // need 4 brams (8 * 2k) in total
#pragma HLS BIND_STORAGE variable = iact_buffer type = ram_t2p impl = bram latency = 1
#pragma HLS array_partition variable = iact_buffer type = cyclic factor = Parallel_K dim = 1 // read 32 elements at one time

    hls::stream<ap_uint<IACTS_DATAWIDTH>> iacts_stream;
#pragma HLS STREAM variable = iacts_stream depth = PARALLEL_K type = fifo

    hls::stream<ap_uint<WEIGHTS_DATAWIDTH>> weights_stream[Parallel_K];
#pragma HLS STREAM variable = weights_stream depth = PARALLEL_N type = fifo

    hls::stream<ap_uint<OACTS_DATAWIDTH>> output_stream;
#pragma HLS STREAM variable = output_stream depth = Parallel_N type = fifo

    ap_uint<OACTS_DATAWIDTH> output_buf[Wt_Y/PARALLEL_N][PARALLEL_N];
#pragma HLS BIND_STORAGE variable = iact_buffer type = ram_t2p impl = bram latency = 1
#pragma HLS array_partition variable = iact_buffer type = complete dim = 1 // read 32 elements at one time

    int block_num_x = Wt_X / PARALLEL_K;
    int block_num_y = Wt_Y / PARALLEL_N;
    ap_uint<IACTS_DATAWIDTH> IACT_TEMP_BUFFER[PARALLEL_K];

    int address_ifc = block_num_x * block_num_y * BLOCK_WEIGHT_TOTAL_IFC;
    #pragma HLS DATAFLOW
    ReadFromMem(ifc1, ifc2, ifc3, ifc4, ifc5, ifc6, ifc7, weight_buffer, weights_stream, iact_buffer, iacts_stream, X, Y, Wt_X, Wt_Y);
    //compute_systolic(iacts_stream, weights_stream, bias, output_stream, X, Y, Wt_X, Wt_Y);

    for (int i = 0; i < block_num_x; ++i) {
        ReadIactBuff(iacts_stream, IACT_TEMP_BUFFER);
        CreateBitMask(weight_stream, second_processing_buffer, second_bit_buffer_weights);
        for (int j = 0; j < block_num_y; ++j) {
            int batch = i * block_num_y + j;
            if (batch != block_num_x*block_num_y - 1) {
                if (batch % 2== 0) {
                    CreateBitMask(weight_stream, second_processing_buffer, second_bit_buffer_weights);
                    DPEComputation(IACT_TEMP_BUFFER, i, j, first_processing_buffer, first_bit_buffer_weights, output_buf, Wt_X, Wt_Y);
                } else {
                    CreateBitMask(weight_stream, first_processing_buffer, first_bit_buffer_weights);
                    DPEComputation(IACT_TEMP_BUFFER, i, j, second_processing_buffer, second_bit_buffer_weights, output_buf, Wt_X, Wt_Y);
                }
            }
            else {
                DPEComputation(IACT_TEMP_BUFFER, i, j, second_processing_buffer, second_bit_buffer_weights, output_buf, Wt_X, Wt_Y); //depends on batch_num
            }
        }
    }
    OutputBuffer(ifc7, output_stream, X, Wt_Y, address_ifc, output_buf);
}