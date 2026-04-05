# compression kernels
# fused kernel does K+V in one launch. separate kernels for backward compat.

try:
    import cuda.tile as ct
except ImportError:
    import cutile as ct  # type: ignore

from .constants import BLOCK_S, HEAD_DIM


# fused KV compression

@ct.kernel
def turboquant_compress_kv_3bit(
    K, V, Pi_T, Pi, S_T,
    K_Indices, K_Signs, K_Norms, K_RNorms,
    V_Indices, V_Norms,
    # key codebook: 4 centroids, 3 boundaries
    kc0: float, kc1: float, kc2: float, kc3: float,
    kb1: float, kb2: float, kb3: float,
    # value codebook: 8 centroids, 7 boundaries
    vc0: float, vc1: float, vc2: float, vc3: float,
    vc4: float, vc5: float, vc6: float, vc7: float,
    vb1: float, vb2: float, vb3: float, vb4: float,
    vb5: float, vb6: float, vb7: float,
    seq_len: int,
):
    # 2-bit key mse + 1-bit qjl, 3-bit values
    block_id = ct.bid(0)
    zero_pad = ct.PaddingMode.ZERO

    pi_t = ct.load(Pi_T, index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))
    pi   = ct.load(Pi,   index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))
    s_t  = ct.load(S_T,  index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))

    # keys
    k_tile = ct.load(K, index=(block_id, 0), shape=(BLOCK_S, HEAD_DIM),
                     padding_mode=zero_pad)
    k_f32 = ct.astype(k_tile, ct.float32)
    k_norms = ct.sqrt(ct.sum(k_f32 * k_f32, axis=1))
    k_safe = ct.where(k_norms > 1e-8, k_norms, 1e-8)
    k_normed = k_f32 / ct.expand_dims(k_safe, axis=1)

    ky = ct.mma(ct.astype(k_normed, ct.float16), pi_t,
                ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))

    kidx = ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32)
    kidx = ct.where(ky > kb1, 1.0, kidx)
    kidx = ct.where(ky > kb2, 2.0, kidx)
    kidx = ct.where(ky > kb3, 3.0, kidx)

    ky_hat = ct.full((BLOCK_S, HEAD_DIM), kc0, dtype=ct.float32)
    ky_hat = ct.where(kidx > 0.5, kc1, ky_hat)
    ky_hat = ct.where(kidx > 1.5, kc2, ky_hat)
    ky_hat = ct.where(kidx > 2.5, kc3, ky_hat)

    k_bar = ct.mma(ct.astype(ky_hat, ct.float16), pi,
                   ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))
    k_mse = k_bar * ct.expand_dims(k_norms, axis=1)

    residual = k_f32 - k_mse
    r_norms = ct.sqrt(ct.sum(residual * residual, axis=1))
    projected = ct.mma(ct.astype(residual, ct.float16), s_t,
                       ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))
    signs = ct.where(projected >= 0.0, 1.0, -1.0)

    ct.store(K_Indices, index=(block_id, 0), tile=ct.astype(kidx, ct.uint8))
    ct.store(K_Signs,   index=(block_id, 0), tile=ct.astype(signs, ct.int8))
    ct.store(K_Norms,   index=(block_id,),   tile=ct.astype(k_norms, ct.float16))
    ct.store(K_RNorms,  index=(block_id,),   tile=ct.astype(r_norms, ct.float16))

    # values
    v_tile = ct.load(V, index=(block_id, 0), shape=(BLOCK_S, HEAD_DIM),
                     padding_mode=zero_pad)
    v_f32 = ct.astype(v_tile, ct.float32)
    v_norms = ct.sqrt(ct.sum(v_f32 * v_f32, axis=1))
    v_safe = ct.where(v_norms > 1e-8, v_norms, 1e-8)
    v_normed = v_f32 / ct.expand_dims(v_safe, axis=1)

    vy = ct.mma(ct.astype(v_normed, ct.float16), pi_t,
                ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))

    vidx = ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32)
    vidx = ct.where(vy > vb1, 1.0, vidx)
    vidx = ct.where(vy > vb2, 2.0, vidx)
    vidx = ct.where(vy > vb3, 3.0, vidx)
    vidx = ct.where(vy > vb4, 4.0, vidx)
    vidx = ct.where(vy > vb5, 5.0, vidx)
    vidx = ct.where(vy > vb6, 6.0, vidx)
    vidx = ct.where(vy > vb7, 7.0, vidx)

    ct.store(V_Indices, index=(block_id, 0), tile=ct.astype(vidx, ct.uint8))
    ct.store(V_Norms,   index=(block_id,),   tile=ct.astype(v_norms, ct.float16))


# key compression (separate)

@ct.kernel
def turboquant_compress_2bit(
    K, Pi_T, Pi, S_T,
    Indices, Signs, Norms, RNorms,
    c0: float, c1: float, c2: float, c3: float,
    b1: float, b2: float, b3: float,
    seq_k: int,
):
    # 2-bit mse (4 centroids) + 1-bit qjl
    block_id = ct.bid(0)
    zero_pad = ct.PaddingMode.ZERO

    k_tile = ct.load(K, index=(block_id, 0), shape=(BLOCK_S, HEAD_DIM),
                     padding_mode=zero_pad)
    pi_t = ct.load(Pi_T, index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))
    pi   = ct.load(Pi,   index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))
    s_t  = ct.load(S_T,  index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))

    k_f32 = ct.astype(k_tile, ct.float32)
    norms = ct.sqrt(ct.sum(k_f32 * k_f32, axis=1))
    safe_norms = ct.where(norms > 1e-8, norms, 1e-8)
    k_normed = k_f32 / ct.expand_dims(safe_norms, axis=1)

    y = ct.mma(ct.astype(k_normed, ct.float16), pi_t,
               ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))

    idx = ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32)
    idx = ct.where(y > b1, 1.0, idx)
    idx = ct.where(y > b2, 2.0, idx)
    idx = ct.where(y > b3, 3.0, idx)

    y_hat = ct.full((BLOCK_S, HEAD_DIM), c0, dtype=ct.float32)
    y_hat = ct.where(idx > 0.5, c1, y_hat)
    y_hat = ct.where(idx > 1.5, c2, y_hat)
    y_hat = ct.where(idx > 2.5, c3, y_hat)

    k_bar_hat = ct.mma(ct.astype(y_hat, ct.float16), pi,
                       ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))
    k_mse = k_bar_hat * ct.expand_dims(norms, axis=1)

    residual = k_f32 - k_mse
    r_norms = ct.sqrt(ct.sum(residual * residual, axis=1))
    projected = ct.mma(ct.astype(residual, ct.float16), s_t,
                       ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))
    signs = ct.where(projected >= 0.0, 1.0, -1.0)

    ct.store(Indices, index=(block_id, 0), tile=ct.astype(idx, ct.uint8))
    ct.store(Signs,   index=(block_id, 0), tile=ct.astype(signs, ct.int8))
    ct.store(Norms,   index=(block_id,),   tile=ct.astype(norms, ct.float16))
    ct.store(RNorms,  index=(block_id,),   tile=ct.astype(r_norms, ct.float16))


@ct.kernel
def turboquant_compress_3bit(
    K, Pi_T, Pi, S_T,
    Indices, Signs, Norms, RNorms,
    c0: float, c1: float, c2: float, c3: float,
    c4: float, c5: float, c6: float, c7: float,
    b1: float, b2: float, b3: float, b4: float,
    b5: float, b6: float, b7: float,
    seq_k: int,
):
    # 3-bit mse (8 centroids) + 1-bit qjl
    block_id = ct.bid(0)
    zero_pad = ct.PaddingMode.ZERO

    k_tile = ct.load(K, index=(block_id, 0), shape=(BLOCK_S, HEAD_DIM),
                     padding_mode=zero_pad)
    pi_t = ct.load(Pi_T, index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))
    pi   = ct.load(Pi,   index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))
    s_t  = ct.load(S_T,  index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))

    k_f32 = ct.astype(k_tile, ct.float32)
    norms = ct.sqrt(ct.sum(k_f32 * k_f32, axis=1))
    safe_norms = ct.where(norms > 1e-8, norms, 1e-8)
    k_normed = k_f32 / ct.expand_dims(safe_norms, axis=1)

    y = ct.mma(ct.astype(k_normed, ct.float16), pi_t,
               ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))

    idx = ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32)
    idx = ct.where(y > b1, 1.0, idx)
    idx = ct.where(y > b2, 2.0, idx)
    idx = ct.where(y > b3, 3.0, idx)
    idx = ct.where(y > b4, 4.0, idx)
    idx = ct.where(y > b5, 5.0, idx)
    idx = ct.where(y > b6, 6.0, idx)
    idx = ct.where(y > b7, 7.0, idx)

    y_hat = ct.full((BLOCK_S, HEAD_DIM), c0, dtype=ct.float32)
    y_hat = ct.where(idx > 0.5, c1, y_hat)
    y_hat = ct.where(idx > 1.5, c2, y_hat)
    y_hat = ct.where(idx > 2.5, c3, y_hat)
    y_hat = ct.where(idx > 3.5, c4, y_hat)
    y_hat = ct.where(idx > 4.5, c5, y_hat)
    y_hat = ct.where(idx > 5.5, c6, y_hat)
    y_hat = ct.where(idx > 6.5, c7, y_hat)

    k_bar_hat = ct.mma(ct.astype(y_hat, ct.float16), pi,
                       ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))
    k_mse = k_bar_hat * ct.expand_dims(norms, axis=1)

    residual = k_f32 - k_mse
    r_norms = ct.sqrt(ct.sum(residual * residual, axis=1))
    projected = ct.mma(ct.astype(residual, ct.float16), s_t,
                       ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))
    signs = ct.where(projected >= 0.0, 1.0, -1.0)

    ct.store(Indices, index=(block_id, 0), tile=ct.astype(idx, ct.uint8))
    ct.store(Signs,   index=(block_id, 0), tile=ct.astype(signs, ct.int8))
    ct.store(Norms,   index=(block_id,),   tile=ct.astype(norms, ct.float16))
    ct.store(RNorms,  index=(block_id,),   tile=ct.astype(r_norms, ct.float16))


# value compression (no qjl)

@ct.kernel
def turboquant_compress_values_3bit(
    V, Pi_T,
    Indices, Norms,
    c0: float, c1: float, c2: float, c3: float,
    c4: float, c5: float, c6: float, c7: float,
    b1: float, b2: float, b3: float, b4: float,
    b5: float, b6: float, b7: float,
    seq_v: int,
):
    block_id = ct.bid(0)
    zero_pad = ct.PaddingMode.ZERO

    v_tile = ct.load(V, index=(block_id, 0), shape=(BLOCK_S, HEAD_DIM),
                     padding_mode=zero_pad)
    pi_t = ct.load(Pi_T, index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))

    v_f32 = ct.astype(v_tile, ct.float32)
    norms = ct.sqrt(ct.sum(v_f32 * v_f32, axis=1))
    safe_norms = ct.where(norms > 1e-8, norms, 1e-8)
    v_normed = v_f32 / ct.expand_dims(safe_norms, axis=1)

    y = ct.mma(ct.astype(v_normed, ct.float16), pi_t,
               ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))

    idx = ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32)
    idx = ct.where(y > b1, 1.0, idx)
    idx = ct.where(y > b2, 2.0, idx)
    idx = ct.where(y > b3, 3.0, idx)
    idx = ct.where(y > b4, 4.0, idx)
    idx = ct.where(y > b5, 5.0, idx)
    idx = ct.where(y > b6, 6.0, idx)
    idx = ct.where(y > b7, 7.0, idx)

    ct.store(Indices, index=(block_id, 0), tile=ct.astype(idx, ct.uint8))
    ct.store(Norms,   index=(block_id,),   tile=ct.astype(norms, ct.float16))


@ct.kernel
def turboquant_compress_values_2bit(
    V, Pi_T,
    Indices, Norms,
    c0: float, c1: float, c2: float, c3: float,
    b1: float, b2: float, b3: float,
    seq_v: int,
):
    block_id = ct.bid(0)
    zero_pad = ct.PaddingMode.ZERO

    v_tile = ct.load(V, index=(block_id, 0), shape=(BLOCK_S, HEAD_DIM),
                     padding_mode=zero_pad)
    pi_t = ct.load(Pi_T, index=(0, 0), shape=(HEAD_DIM, HEAD_DIM))

    v_f32 = ct.astype(v_tile, ct.float32)
    norms = ct.sqrt(ct.sum(v_f32 * v_f32, axis=1))
    safe_norms = ct.where(norms > 1e-8, norms, 1e-8)
    v_normed = v_f32 / ct.expand_dims(safe_norms, axis=1)

    y = ct.mma(ct.astype(v_normed, ct.float16), pi_t,
               ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32))

    idx = ct.zeros((BLOCK_S, HEAD_DIM), dtype=ct.float32)
    idx = ct.where(y > b1, 1.0, idx)
    idx = ct.where(y > b2, 2.0, idx)
    idx = ct.where(y > b3, 3.0, idx)

    ct.store(Indices, index=(block_id, 0), tile=ct.astype(idx, ct.uint8))
    ct.store(Norms,   index=(block_id,),   tile=ct.astype(norms, ct.float16))
