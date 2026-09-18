#include <cstdint>
#include <cstring>
extern "C" int gather_kv(const unsigned char* k, const unsigned char* v,
                         unsigned char* out_k, unsigned char* out_v,
                         const int64_t* rows, int64_t n, int64_t source_rows,
                         int64_t row_bytes, int threads) {
    int invalid = 0;
    #pragma omp parallel for schedule(static) num_threads(threads) reduction(|:invalid)
    for (int64_t i = 0; i < n; ++i) {
        const int64_t r = rows[i];
        if (r < 0 || r >= source_rows) { invalid = 1; continue; }
        std::memcpy(out_k + i * row_bytes, k + r * row_bytes, row_bytes);
        std::memcpy(out_v + i * row_bytes, v + r * row_bytes, row_bytes);
    }
    return invalid;
}
