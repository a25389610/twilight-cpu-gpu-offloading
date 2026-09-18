#include <cstdint>
#include <cstring>
#include <omp.h>
extern "C" int gather_kv(const unsigned char* k, const unsigned char* v,
                         unsigned char* out_k, unsigned char* out_v,
                         const int64_t* rows, int64_t n, int64_t source_rows,
                         int64_t row_bytes, int threads) {
    int invalid = 0;
    #pragma omp parallel num_threads(threads) reduction(|:invalid)
    {
        const int worker = omp_get_thread_num(), workers = omp_get_num_threads();
        int64_t i = n * worker / workers;
        const int64_t end = n * (worker + 1) / workers;
        while (i < end) {
            const int64_t r = rows[i];
            if (r < 0 || r >= source_rows) { invalid = 1; ++i; continue; }
            int64_t run = 1;
            while (i + run < end && run < source_rows - r && rows[i + run] == r + run) ++run;
            if (run >= 16) {
                std::memcpy(out_k + i * row_bytes, k + r * row_bytes, run * row_bytes);
                std::memcpy(out_v + i * row_bytes, v + r * row_bytes, run * row_bytes);
            } else {
                for (int64_t j = 0; j < run; ++j) {
                    std::memcpy(out_k + (i+j) * row_bytes, k + (r+j) * row_bytes, row_bytes);
                    std::memcpy(out_v + (i+j) * row_bytes, v + (r+j) * row_bytes, row_bytes);
                }
            }
            i += run;
        }
    }
    return invalid;
}
