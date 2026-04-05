# turboquant engine
# cutile first, pytorch fallback if unavailable

import math
import torch

from .codebook import LloydMaxCodebook
from .constants import BLOCK_S, DEFAULT_SEED, DEFAULT_TOTAL_BITS, HEAD_DIM


def _rotation_matrix(d, seed, device="cpu"):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    return (Q * diag_sign.unsqueeze(0)).to(device)


def _qjl_matrix(d, seed, device="cpu"):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 10000)
    return torch.randn(d, d, generator=gen).to(device)


class TurboQuantEngine:

    def __init__(self, head_dim=HEAD_DIM, total_bits=DEFAULT_TOTAL_BITS,
                 seed=DEFAULT_SEED, device="cpu"):
        self.head_dim   = head_dim
        self.total_bits = total_bits
        self.mse_bits   = max(total_bits - 1, 1)
        self.device     = device

        self.Pi  = _rotation_matrix(head_dim, seed, device)
        self.PiT = self.Pi.T.contiguous()
        self.S   = _qjl_matrix(head_dim, seed, device)
        self.ST  = self.S.T.contiguous()

        self.key_codebook = LloydMaxCodebook(head_dim, self.mse_bits)
        self.val_codebook = LloydMaxCodebook(head_dim, total_bits)
        self._force_pytorch = False

    @torch.no_grad()
    def compress_kv_cache(self, past_key_values):
        kv_keys, kv_vals = self._extract_kv(past_key_values)

        layers = []
        for li in range(len(kv_keys)):
            n_heads = kv_keys[li].shape[1]
            ck_list, cv_list = [], []
            for h in range(n_heads):
                K_h = kv_keys[li][0, h].half().contiguous()
                V_h = kv_vals[li][0, h].half().contiguous()
                ck, cv = self._compress_kv_fused(K_h, V_h)
                ck_list.append(ck)
                cv_list.append(cv)
            layers.append((ck_list, cv_list))

        return {"layers": layers}

    @torch.no_grad()
    def build_cache(self, compressed):
        from transformers import DynamicCache
        cache = DynamicCache()
        for li, (ck_list, cv_list) in enumerate(compressed["layers"]):
            k_heads = [ck["k_mse"] for ck in ck_list]
            k_layer = torch.stack(k_heads).unsqueeze(0)
            v_heads = [self._decompress_values(cv) for cv in cv_list]
            v_layer = torch.stack(v_heads).unsqueeze(0)
            cache.update(k_layer, v_layer, li)
        return cache

    @torch.no_grad()
    def generate(self, model, tokenizer, prompt, max_new_tokens=100,
                 repetition_penalty=1.3):
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        out = model(**inputs, use_cache=True)

        compressed = self.compress_kv_cache(out.past_key_values)
        stats = self.compression_stats(out.past_key_values)
        cache = self.build_cache(compressed)

        seq_len = inputs["input_ids"].shape[1]
        all_ids = inputs["input_ids"][0].tolist()

        next_tok = out.logits[:, -1:].argmax(dim=-1)
        all_ids.append(next_tok.item())

        for step in range(max_new_tokens - 1):
            o = model(input_ids=next_tok, past_key_values=cache,
                      position_ids=torch.tensor([[seq_len + step]],
                                                device=self.device),
                      use_cache=True)
            cache = o.past_key_values
            logits = o.logits[:, -1, :]

            if repetition_penalty != 1.0:
                for tid in set(all_ids):
                    if logits[0, tid] > 0:
                        logits[0, tid] /= repetition_penalty
                    else:
                        logits[0, tid] *= repetition_penalty

            next_tok = logits.argmax(dim=-1, keepdim=True)
            all_ids.append(next_tok.item())
            if next_tok.item() == tokenizer.eos_token_id:
                break

        n_new = len(all_ids) - seq_len
        text = tokenizer.decode(all_ids, skip_special_tokens=True)
        return {"text": text, "tokens": n_new, "stats": stats}

    @torch.no_grad()
    def auto_tune(self, seq_len=512, warmup=10, trials=50, quality_threshold=0.85):
        import time as _time
        import torch.nn.functional as F

        on_gpu = self.device != "cpu" and torch.cuda.is_available()

        if on_gpu:
            gpu_name = torch.cuda.get_device_name()
            sm = torch.cuda.get_device_capability()
            print(f"auto-tuning on {gpu_name} (sm_{sm[0]}{sm[1]})")
        else:
            print("auto-tuning on CPU")

        # probe cutile
        cutile_ok = False
        if on_gpu:
            try:
                probe = torch.randn(64, self.head_dim, device=self.device, dtype=torch.float16)
                from .compress import turboquant_compress_2bit
                import cuda.tile as ct
                idx = torch.empty(64, self.head_dim, dtype=torch.uint8, device=self.device)
                sgn = torch.empty(64, self.head_dim, dtype=torch.int8, device=self.device)
                nrm = torch.empty(64, dtype=torch.float16, device=self.device)
                rnm = torch.empty(64, dtype=torch.float16, device=self.device)
                c = self.key_codebook.centroids.tolist()
                b = self.key_codebook.boundaries.tolist()
                ct.launch(torch.cuda.current_stream(), (1, 1, 1),
                          turboquant_compress_2bit, (
                              probe, self.PiT.half(), self.Pi.half(), self.ST.half(),
                              idx, sgn, nrm, rnm, *c, *b, 64))
                torch.cuda.synchronize()
                cutile_ok = True
            except Exception:
                pass

        print(f"cutile kernels: {'available' if cutile_ok else 'not available (pytorch fallback)'}\n")

        K = torch.randn(seq_len, self.head_dim, device=self.device, dtype=torch.float16)
        V = torch.randn(seq_len, self.head_dim, device=self.device, dtype=torch.float16)

        backends = ["cutile", "pytorch"] if cutile_ok else ["pytorch"]
        results = []

        for bits in [2, 3]:
            for backend in backends:
                eng = TurboQuantEngine(
                    head_dim=self.head_dim, total_bits=bits,
                    seed=DEFAULT_SEED, device=self.device)

                if backend == "pytorch":
                    compress_k = eng._compress_keys_pt
                    compress_v = eng._compress_values_pt
                    decompress_v = eng._decompress_values_pt
                else:
                    compress_k = eng._compress_keys
                    compress_v = eng._compress_values
                    decompress_v = eng._decompress_values

                for _ in range(warmup):
                    ck = compress_k(K)
                    cv = compress_v(V)
                    dv = decompress_v(cv)
                if on_gpu:
                    torch.cuda.synchronize()

                times = []
                for _ in range(trials):
                    if on_gpu:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        ck = compress_k(K)
                        cv = compress_v(V)
                        dv = decompress_v(cv)
                        end.record()
                        torch.cuda.synchronize()
                        times.append(start.elapsed_time(end))
                    else:
                        t0 = _time.perf_counter()
                        ck = compress_k(K)
                        cv = compress_v(V)
                        dv = decompress_v(cv)
                        times.append((_time.perf_counter() - t0) * 1000)

                median_ms = sorted(times)[len(times) // 2]

                k_cos = F.cosine_similarity(
                    ck["k_mse"].float().flatten(), K.float().flatten(), dim=0).item()
                v_cos = F.cosine_similarity(
                    dv.float().flatten(), V.float().flatten(), dim=0).item()

                ok = k_cos >= quality_threshold and v_cos >= quality_threshold
                tag = "✓" if ok else "✗"

                results.append({
                    "total_bits": bits, "backend": backend,
                    "median_ms": median_ms,
                    "key_cos": k_cos, "val_cos": v_cos, "pass": ok})
                print(f"  bits={bits}  {backend:<8s}  {median_ms:6.2f}ms  "
                      f"key_cos={k_cos:.3f}  val_cos={v_cos:.3f}  {tag}")

        valid = [r for r in results if r["pass"]]
        if not valid:
            print("\nno config passed quality threshold, keeping defaults")
            return results

        best = min(valid, key=lambda r: r["median_ms"])

        if best["total_bits"] != self.total_bits:
            self.total_bits = best["total_bits"]
            self.mse_bits = max(best["total_bits"] - 1, 1)
            self.key_codebook = LloydMaxCodebook(self.head_dim, self.mse_bits)
            self.val_codebook = LloydMaxCodebook(self.head_dim, self.total_bits)

        self._force_pytorch = (best["backend"] == "pytorch")

        print(f"\n→ selected: {best['total_bits']}-bit {best['backend']} "
              f"({best['median_ms']:.2f}ms, "
              f"key={best['key_cos']:.3f}, val={best['val_cos']:.3f})")
        return results

    def compression_stats(self, past_key_values):
        kv_keys, kv_vals = self._extract_kv(past_key_values)
        n_layers = len(kv_keys)
        n_heads  = kv_keys[0].shape[1]
        seq_len  = kv_keys[0].shape[2]

        fp16_bytes = sum(k.nbytes + v.nbytes for k, v in zip(kv_keys, kv_vals))
        tq_bytes   = self._compressed_bytes(seq_len) * n_heads * n_layers

        return {
            "seq_len": seq_len, "n_layers": n_layers, "n_heads": n_heads,
            "fp16_bytes": fp16_bytes, "tq_bytes": tq_bytes,
            "ratio": fp16_bytes / tq_bytes}

    # internals

    @staticmethod
    def _extract_kv(past_key_values):
        try:
            return past_key_values.key_cache, past_key_values.value_cache
        except AttributeError:
            keys = [kv[0] for kv in past_key_values]
            vals = [kv[1] for kv in past_key_values]
            return keys, vals

    def _compress_kv_fused(self, K, V):
        """fused K+V compression, single kernel launch (3-bit only)"""
        if self._force_pytorch or self.total_bits != 3:
            return self._compress_keys(K), self._compress_values(V)
        try:
            from .compress import turboquant_compress_kv_3bit
            import cuda.tile as ct

            seq_len, d = K.shape

            k_indices = torch.empty(seq_len, d, dtype=torch.uint8, device=K.device)
            k_signs   = torch.empty(seq_len, d, dtype=torch.int8,  device=K.device)
            k_norms   = torch.empty(seq_len, dtype=torch.float16,  device=K.device)
            k_rnorms  = torch.empty(seq_len, dtype=torch.float16,  device=K.device)
            v_indices = torch.empty(seq_len, d, dtype=torch.uint8, device=K.device)
            v_norms   = torch.empty(seq_len, dtype=torch.float16,  device=K.device)

            grid = (self._cdiv(seq_len, BLOCK_S), 1, 1)
            kc = self.key_codebook.centroids.tolist()
            kb = self.key_codebook.boundaries.tolist()
            vc = self.val_codebook.centroids.tolist()
            vb = self.val_codebook.boundaries.tolist()
            stream = torch.cuda.current_stream()

            ct.launch(stream, grid, turboquant_compress_kv_3bit, (
                K, V, self.PiT.half(), self.Pi.half(), self.ST.half(),
                k_indices, k_signs, k_norms, k_rnorms,
                v_indices, v_norms,
                *kc, *kb, *vc, *vb, seq_len))

            k_mse = self._dequant_keys(k_indices, k_norms)
            ck = {"indices": k_indices, "k_mse": k_mse, "qjl_signs": k_signs,
                  "vec_norms": k_norms, "residual_norms": k_rnorms}
            cv = {"indices": v_indices, "vec_norms": v_norms}
            return ck, cv

        except (ImportError, RuntimeError):
            return self._compress_keys_pt(K), self._compress_values_pt(V)

    def _compress_keys(self, K):
        if self._force_pytorch:
            return self._compress_keys_pt(K)
        try:
            from .compress import turboquant_compress_2bit, turboquant_compress_3bit
            import cuda.tile as ct

            seq_k, d = K.shape
            indices = torch.empty(seq_k, d, dtype=torch.uint8, device=K.device)
            signs   = torch.empty(seq_k, d, dtype=torch.int8,  device=K.device)
            norms   = torch.empty(seq_k, dtype=torch.float16,  device=K.device)
            r_norms = torch.empty(seq_k, dtype=torch.float16,  device=K.device)

            grid = (self._cdiv(seq_k, BLOCK_S), 1, 1)
            c = self.key_codebook.centroids.tolist()
            b = self.key_codebook.boundaries.tolist()
            stream = torch.cuda.current_stream()

            if self.mse_bits == 2:
                ct.launch(stream, grid, turboquant_compress_2bit, (
                    K, self.PiT.half(), self.Pi.half(), self.ST.half(),
                    indices, signs, norms, r_norms, *c, *b, seq_k))
            elif self.mse_bits == 3:
                ct.launch(stream, grid, turboquant_compress_3bit, (
                    K, self.PiT.half(), self.Pi.half(), self.ST.half(),
                    indices, signs, norms, r_norms, *c, *b, seq_k))
            else:
                return self._compress_keys_pt(K)

            k_mse = self._dequant_keys(indices, norms)
            return {"indices": indices, "k_mse": k_mse, "qjl_signs": signs,
                    "vec_norms": norms, "residual_norms": r_norms}
        except (ImportError, RuntimeError):
            return self._compress_keys_pt(K)

    def _compress_values(self, V):
        if self._force_pytorch:
            return self._compress_values_pt(V)
        try:
            from .compress import (turboquant_compress_values_3bit,
                                   turboquant_compress_values_2bit)
            import cuda.tile as ct

            seq_v, d = V.shape
            indices = torch.empty(seq_v, d, dtype=torch.uint8, device=V.device)
            norms   = torch.empty(seq_v, dtype=torch.float16,  device=V.device)

            grid = (self._cdiv(seq_v, BLOCK_S), 1, 1)
            c = self.val_codebook.centroids.tolist()
            b = self.val_codebook.boundaries.tolist()
            stream = torch.cuda.current_stream()

            if self.total_bits == 3:
                ct.launch(stream, grid, turboquant_compress_values_3bit, (
                    V, self.PiT.half(), indices, norms, *c, *b, seq_v))
            elif self.total_bits == 2:
                ct.launch(stream, grid, turboquant_compress_values_2bit, (
                    V, self.PiT.half(), indices, norms, *c, *b, seq_v))
            else:
                return self._compress_values_pt(V)

            return {"indices": indices, "vec_norms": norms}
        except (ImportError, RuntimeError):
            return self._compress_values_pt(V)

    def _decompress_values(self, cv):
        if self._force_pytorch:
            return self._decompress_values_pt(cv)
        try:
            from .decompress import (turboquant_decompress_3bit,
                                     turboquant_decompress_2bit)
            import cuda.tile as ct

            indices = cv["indices"]
            norms   = cv["vec_norms"]
            seq_v   = indices.shape[0]
            output  = torch.empty(seq_v, self.head_dim, dtype=torch.float16,
                                  device=indices.device)

            grid = (self._cdiv(seq_v, BLOCK_S), 1, 1)
            c = self.val_codebook.centroids.tolist()
            stream = torch.cuda.current_stream()

            if self.total_bits == 3:
                ct.launch(stream, grid, turboquant_decompress_3bit, (
                    indices, norms, self.Pi.half(), output, *c, seq_v))
            elif self.total_bits == 2:
                ct.launch(stream, grid, turboquant_decompress_2bit, (
                    indices, norms, self.Pi.half(), output, *c, seq_v))
            else:
                return self._decompress_values_pt(cv)
            return output
        except (ImportError, RuntimeError):
            return self._decompress_values_pt(cv)

    # pytorch fallbacks

    def _compress_keys_pt(self, K):
        K_f = K.float()
        norms = torch.norm(K_f, dim=-1, keepdim=True)
        K_normed = K_f / (norms + 1e-8)
        rotated = K_normed @ self.PiT.float()

        c = self.key_codebook.centroids.to(K.device)
        indices = (rotated.unsqueeze(-1) - c).abs().argmin(dim=-1).to(torch.uint8)

        y_hat = c[indices.long()]
        k_mse = (y_hat @ self.Pi.float()) * norms
        residual = K_f - k_mse
        r_norms = torch.norm(residual, dim=-1)

        signs = torch.sign(residual @ self.ST.float()).to(torch.int8)
        signs[signs == 0] = 1

        return {"indices": indices, "k_mse": k_mse.half(), "qjl_signs": signs,
                "vec_norms": norms.squeeze(-1).half(),
                "residual_norms": r_norms.half()}

    def _compress_values_pt(self, V):
        V_f = V.float()
        norms = torch.norm(V_f, dim=-1, keepdim=True)
        V_normed = V_f / (norms + 1e-8)
        rotated = V_normed @ self.PiT.float()

        c = self.val_codebook.centroids.to(V.device)
        indices = (rotated.unsqueeze(-1) - c).abs().argmin(dim=-1).to(torch.uint8)

        return {"indices": indices, "vec_norms": norms.squeeze(-1).half()}

    def _decompress_values_pt(self, cv):
        c = self.val_codebook.centroids.to(cv["indices"].device)
        y_hat = c[cv["indices"].long()]
        norms = cv["vec_norms"].float().unsqueeze(-1)
        return ((y_hat @ self.Pi.float()) * norms).half()

    def _dequant_keys(self, indices, norms):
        c = self.key_codebook.centroids.to(indices.device)
        y_hat = c[indices.long()]
        return ((y_hat.float() @ self.Pi.float()) * norms.float().unsqueeze(-1)).half()

    def _cdiv(self, a, b):
        return (a + b - 1) // b

    def _compressed_bytes(self, seq_len):
        d = self.head_dim
        key_bytes = (seq_len * d * self.mse_bits + seq_len * d + seq_len * 32) / 8
        val_bytes = (seq_len * d * self.total_bits + seq_len * 16) / 8
        return key_bytes + val_bytes
