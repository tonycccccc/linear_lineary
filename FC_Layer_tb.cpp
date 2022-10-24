#include "FC_Layer.hpp"
#include "csv.h"
#include "sstream"
#include <json/value.h>

using namespace std;

// only test 1*2048 + 2048*1000 for now
int MAX_X = 1;
int MAX_Y = 2048;
int MAX_Wt_X = 2048;
int MAX_Wt_Y = 1000;

float iacts_intput[MAX_X*MAX_Y];
float weights_input[MAX_Wt_X*MAX_Wt_Y];
float oacts_intput[MAX_X*MAX_Wt_Y];
float bias_input[MAX_X*MAX_Wt_Y];

data_t iacts[MAX_X*MAX_Y];
data_t weights[MAX_Wt_X*MAX_Wt_Y];
data_t oacts[MAX_X*MAX_Wt_Y];
data_t bias[MAX_X*MAX_Wt_Y];

//ap_int<IACTS_DATAWIDTH> iacts[X * Y];
//ap_int<WEIGHTS_DATAWIDTH> weights[Wt_X * Wt_Y];
//ap_int<OACTS_QUANT_WIDTH> oacts[X * Wt_Y];

void read_bin_files(int X, int Y, int Wt_X, int Wt_Y){
    std::ifstream iacts_file("/Users/zeyuchen/Desktop/Project_Fall_2022/Fast_Switch/DSE/DSE/fpga_kernel/hls/test_csv_data/linear_input_1.pth", ios::in);
    std::ifstream weights_file("/Users/zeyuchen/Desktop/Project_Fall_2022/Fast_Switch/DSE/DSE/fpga_kernel/hls/test_csv_data/linear_weights_1.pth", ios::in);
    std::ifstream reference_oacts_file("/Users/zeyuchen/Desktop/Project_Fall_2022/Fast_Switch/DSE/DSE/fpga_kernel/hls/test_csv_data/linear_oActs_1.pth", ios::in);
    std::ifstream bias_file("/Users/zeyuchen/Desktop/Project_Fall_2022/Fast_Switch/DSE/DSE/fpga_kernel/hls/test_csv_data/linear_oActs_1.pth", ios::in);
    std::cout << "iacts data: " << std::endl;
    for(int i = 0; i < X; ++i){
        for (int j = 0; j < Y; ++j){
            std::string temp_data;
            std::stringstream ss;
            std::getline(iacts_file, temp_data, ',');
            iacts_input[i*Y+j] = std::stoi(temp_data);
        }
    }
    std::cout << std::endl;
    
    std::cout << "weights data: " << std::endl;
    for(int i = 0; i < Wt_X; i++){
        for (int j = 0; j < Wt_Y; j++) {
            std::string temp_data;
            std::stringstream ss;
            std::getline(weights_file, temp_data, ',');
            weights_input[i*Wt_Y+j] = std::stoi(temp_data);
        }
    }
    std::cout << std::endl;

    std::cout << "oacts data: " << std::endl;
    for(int i = 0; i < X; i++){
        for (int j = 0; j < Wt_Y; j++) {
            std::string temp_data;
            std::stringstream ss;
            std::getline(reference_oacts_file, temp_data, ',');
            oacts_input[i*Wt_Y+j] = std::stoi(temp_data);
        }
    }

    std::cout << "bias data: " << std::endl;
    for(int i = 0; i < X; i++){
        for (int j = 0; j < Wt_Y; j++) {
            std::string temp_data;
            std::stringstream ss;
            std::getline(bias_file, temp_data, ',');
            bias_input[i*Wt_Y+j] = std::stoi(temp_data);
        }
    }

    std::cout << std::endl;
    //iacts_file.read((char*)(***iacts_input), X*Y*sizeof(float));
    //iacts_file.close();
    //weights_file.read((char*)(***weights_input), Wt_X*Wt_Y*sizeof(float));
    //weights_file.close();
    //reference_oacts_file.read((char*)(***oacts_input), X*Wt_Y*sizeof(float));
    //reference_oacts_file.close();
    //bias = 0; //set bias to 0 for now
}

void convert_data_type(int X, int Y, int Wt_X, int Wt_Y){
    std::cout << "Input Activation" << std::endl;
    for (int i = 0; i < X; ++i) {
        std::cout << std::endl;
        for (int j = 0; j < Y; ++j) {
            std::cout << iacts_input[i*Y+j] << ", ";
            iacts[i*Y+j] = (data_t)iacts_input[i*Y+j];
        }
    }

    std::cout << "Weight Activation" << std::endl;
    for (int i = 0; i < Wt_X; ++i) {
        std::cout << std::endl;
        for (int j = 0; j < Wt_Y; ++j) {
            std::cout << weights_input[i*Wt_Y+j] << ", ";
            weights[i*Wt_Y+j] = (data_t)weights_input[i*Wt_Y+j];
        }
    }

    std::cout << "Output Activation" << std::endl;
    for (int i = 0; i < X; ++i) {
        std::cout << std::endl;
        for (int j = 0; j < Wt_Y; ++j) {
            std::cout << oacts_input[i*Wt_Y+j] << ", ";
            oacts[i*Wt_Y+j] = (data_t)oacts_input[i*Wt_Y+j];
        }
    }

    std::cout << "Bias" << std::endl;
    for (int i = 0; i < X; ++i) {
        std::cout << std::endl;
        for (int j = 0; j < Wt_Y; ++j) {
            std::cout << bias_input[i*Wt_Y+j] << ", ";
            bias[i*Wt_Y+j] = (data_t)bias_input[i*Wt_Y+j];
        }
    }
    //bias = (data_t)bias_input;
}

int main()
{
    //read layer metadata from json file
    std::ifstream json("Metedata.json", std::ifstream::binary);

    Json::Value layer_info;
    json >> layer_info;

    int X = layer_info['X'];
    int Y = layer_info['Y'];
    int Wt_X = layer_info['Wtx'];
    int Wt_Y = layer_info['Wty'];

    long double mse = 0.0;

    read_bin_files(X, Y, Wt_X, Wt_Y);  //readin input files
    convert_data_type(X, Y, Wt_X, Wt_Y);  //convert data type 
    
    int overall_addr  = 0; //set overall output address

    std::cout << "Start Data Memory Layout" << std::endl;

    uint64_t weight_matrix_size_bits = WEIGHTS_DATAWIDTH * Wt_X * Wt_Y;
    uint64_t iact_matrix_size_bits = IACTS_DATAWIDTH * X * Y;
    uint64_t bias_matrix_size_bits = BIAS_DATAWIDTH * X * Wt_Y;   //QUESTION: WHY BIAS IS A MATRIX
    uint64_t total_num_ifc_entries = (weight_matrix_size_bits + iact_matrix_size_bits + bias_matrix_size_bits) / HP_IFC_BANDWIDTH;

    assert(HP_IFC_BANDWIDTH % WEIGHTS_DATAWIDTH == 0); //check if datawidth is divisible by the IFC_Bandwidth
    assert(HP_IFC_BANDWIDTH % BIAS_DATAWIDTH == 0);
    assert(HP_IFC_BANDWIDTH % IACTS_DATAWIDTH == 0);

    ap_uint<HP_IFC_BANDWIDTH> ifc[total_num_ifc_entries];
    

    //weight data should be read column by column plus tiling in column dimension (2048 elements per column needs to be read; depends on how many columns get read at a time)
    int weight_parallel_transmission = HP_IFC_BANDWIDTH / WEIGHTS_DATAWIDTH;
    int weight_loop_count_X = Wt_X / PARALLEL_K;
    int weight_loop_count_Y = Wt_Y / PARALLEL_N;
    int weight_loop_residual = Wt_X * Wt_Y % weight_parallel_transmission; 
    int weight_addr_offset = 0;
    //Read 32 * 20 / 4 = 160 block a time
    for (int i = 0; i < weight_loop_count_X; i++) {
        for (int j = 0; j < weight_loop_count_Y; j++) {
            for (int row = 0; row < PARALLEL_K; ++row) {
                for (int col = 0; col < PARALLEL_N / weight_parallel_transmission; +=row) {
                    ap_uint<HP_IFC_BANDWIDTH> temp;
                    int idx_x = i*PARALLEL_K+row;
                    int idx_y = j*PARALLEL_N+col;
                    temp.range(j*WEIGHTS_DATAWIDTH, (j+1)*WEIGHTS_DATAWIDTH-1) = weights[idx_x][idx_y];
                }
                ifc[overall_addr + i] = temp;
                weight_addr_offset++;
            }   
        }
            
        // for (int j = 0; j < weight_parallel_transmission; ++j) {  //transport column by column
        //     int idx_x = (i*weight_parallel_transmission+j) / Wt_X;
        //     int idx_y = (i*weight_parallel_transmission+j) % Wt_X;
        //     temp.range(j*WEIGHTS_DATAWIDTH, (j+1)*WEIGHTS_DATAWIDTH-1) = weights[idx_x][idx_y]; //QUESTION: BIG_ENDIAN OR SMALL_ENGIAN READ
        // }
        // ifc[overall_addr + i] = temp;
        // weight_addr_offset++;
    }

    if (weight_loop_residual != 0) {
    //address the residual elements (Should not trigger the following part)
        for (int i = 0; i < weight_parallel_transmission; ++i) {
            if (i < weight_loop_residual) {
                int idx_x = (weight_complete_loop_count*weight_parallel_transmission+j) / Wt_X;
                int idx_y = (weight_complete_loop_count*weight_parallel_transmission+j) % Wt_X;
                temp.range(i*WEIGHTS_DATAWIDTH, (i+1)*WEIGHTS_DATAWIDTH-1) = weights[idx_x][idx_y];
            }
            else
            {
                temp.range(i*WEIGHTS_DATAWIDTH, (i+1)*WEIGHTS_DATAWIDTH-1) = 0;
            }
        }
        weight_addr_offset++;
    }

    overall_addr += weight_addr_offset;

    //Layout Iact data
    int iact_parallel_transmission = HP_IFC_BANDWIDTH / IACTS_DATAWIDTH;
    int iact_complete_loop_count = X * Y / iact_parallel_transmission;
    int iact_loop_residual = X * Y % iact_parallel_transmission; 
    int iact_addr_offset = 0;
    for (int i = 0; i < iact_complete_loop_count; ++i) {
        ap_uint<HP_IFC_BANDWIDTH> temp;
        for (int j = 0; j < iact_parallel_transmission; ++j) {
            int idx_x = (i*iact_parallel_transmission+j) / X;
            int idx_y = (i*iact_parallel_transmission+j) % X;
            temp.range(j*IACTS_DATAWIDTH, (j+1)*IACTS_DATAWIDTH-1) = iacts[idx_x][idx_y]; //QUESTION: BIG_ENDIAN OR SMALL_ENGIAN READ
        }
        ifc[overall_addr + i] = temp;
        iact_addr_offset++;
    }
    if (iact_loop_residual != 0) {
        //address the residual elements
        for (int i = 0; i < iact_parallel_transmission; ++i) {
            if (i < iact_loop_residual) {
                int idx_x = (iact_complete_loop_count*iact_parallel_transmission+j) / X;
                int idx_y = (iact_complete_loop_count*iact_parallel_transmission+j) % X;
                temp.range(i*IACTS_DATAWIDTH, (i+1)*IACTS_DATAWIDTH-1) = iacts[idx_x][idx_y];
            }
            else
            {
                temp.range(i*IACTS_DATAWIDTH, (i+1)*IACTS_DATAWIDTH-1) = 0;
            }
        }
        iact_addr_offset++;
    }

    overall_addr += iact_addr_offset;

    //read bias data
    //TODO = SKIP FOR NOW -- FIGURE OUT BIAS DIMENSION

    //perform loop tiling
    data_t output[X][Wt_Y];
    LINEAR(ifc, ifc, ifc, ifc, ifc, ifc, output_ifc, X, Y, Wt_X, Wt_Y);
    for (int i = 0; i < X; ++i) {
        for (int j = 0; j < Wt_Y/4; j+=4) {
            output[i][j] = output_ifc[overall_addr].range((j+1) * OACTS_DATAWIDTH -1, j *OACTS_DATAWIDTH);
            output[i][j+1] = output_ifc[overall_addr].range((j+2) * OACTS_DATAWIDTH -1, (j+1) *OACTS_DATAWIDTH);
            output[i][j+2] = output_ifc[overall_addr].range((j+3) * OACTS_DATAWIDTH -1, (j+2) *OACTS_DATAWIDTH);
            output[i][j+3] = output_ifc[overall_addr].range((j+4) * OACTS_DATAWIDTH -1, (j+3) *OACTS_DATAWIDTH);
            overall_addr++;
        }
    }

    std::cout << "Print Out OUTPUT Data" << std::endl;
    for (int i = 0; i < X; ++i)
    {
        for (int j = 0; j < Wt_Y; ++j)
        {
            std::cout << output[i][j] << ' ,';
        }
        std::cout << std::endl;
    }
    
    //Compute MSE
    for (int i = 0; i < X; ++i)
    {
        for (int j = 0; j < Wt_Y; ++j)
        {
            mse += std::pow((output[i][j]
                    - (data_t)oacts[i][j]), 2);
        }
    }
    
    mse = mse / (X*Wt_Y);

    std::cout << "Output MSE:  " << mse << std::endl;
    return 0;  
}