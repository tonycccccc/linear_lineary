#include <vector>
#define data_t float

class EIE {
    private:
        std::vector<data_t> input_tensor;
        std::vector<data_t> csr_nz_elements;
        std::vector<data_t> csr_zero_vec;
        std::vector<data_t> col_ptr;
};