struct AttentionDefaultFwd {

    constexpr static bool is_softmax = true;

    template<typename Params, typename SharedAllocator>
    __device__ static inline void initialize(Params& params, SharedAllocator& al) {
        // do nothing
    }

    template<typename Params>
    __device__ static inline void load_data(Params& params, int workerid, int tic, int batch, int head, int idx_q, int idx_k) {
        // do nothing
    }

    template<typename Params>
    __device__ static inline void initialize_step(Params& params, int workerid, int tic, int batch, int head, int idx_q, int idx_k) {
        // do nothing
    }

    template<typename Params, typename Q, typename K, typename V>
    __device__ static inline void qkv_transform(Params& params, Q& q, K& k, V& v) {
        // do nothing
    }

    template<typename Params, typename T>
    __device__ static inline void logits_transform(Params& params, T& logits) {
        // do nothing
    }

    template<typename Params>
    __device__ static inline void finalize_step(Params& params) {
        // do nothing
    }

    template<typename Params>
    __device__ static inline void finalize(Params& params, int workerid, int batch, int head, int idx_k) {
        // do nothing
    }

    __device__ static inline size_t get_smem_size() {
        return 0;
    }
};
