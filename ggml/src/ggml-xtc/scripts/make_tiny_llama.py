#!/usr/bin/env python3
"""Create a tiny random-weight F32 llama model (byte-level vocab) to test ggml-xtc.

Usage:
    PYTHONPATH=llama.cpp/gguf-py python3 make_tiny_llama.py /tmp/tiny-llama-f32.gguf

All weights are F32, so every projection is a supported F32 MUL_MAT. With `-b 1 -ub 1`
the matmuls have M=1: only a handful of distinct kernels are needed.
"""
import sys

import numpy as np
import gguf

N_EMBD, N_HEAD, N_HEAD_KV, N_FF, N_LAYER, N_CTX = 64, 4, 2, 128, 2, 128
HEAD_DIM = N_EMBD // N_HEAD


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "tiny-llama-f32.gguf"
    rng = np.random.default_rng(0)

    # SPM-style byte vocab: <unk>, <s>, </s>, then the 256 byte tokens
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)]
    types = [2, 3, 3] + [6] * 256
    n_vocab = len(tokens)

    w = gguf.GGUFWriter(out, "llama")
    w.add_name("tiny-llama-f32")
    w.add_context_length(N_CTX)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(N_LAYER)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_vocab_size(n_vocab)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * n_vocab)
    w.add_token_types(types)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    w.add_add_bos_token(True)

    def mat(rows: int, cols: int) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * 0.05).astype(np.float32)

    def norm(n: int) -> np.ndarray:
        return np.ones(n, dtype=np.float32)

    w.add_tensor("token_embd.weight", mat(n_vocab, N_EMBD))
    w.add_tensor("output_norm.weight", norm(N_EMBD))
    w.add_tensor("output.weight", mat(n_vocab, N_EMBD))
    for i in range(N_LAYER):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", norm(N_EMBD))
        w.add_tensor(p + "attn_q.weight", mat(N_HEAD * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_k.weight", mat(N_HEAD_KV * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_v.weight", mat(N_HEAD_KV * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_output.weight", mat(N_EMBD, N_HEAD * HEAD_DIM))
        w.add_tensor(p + "ffn_norm.weight", norm(N_EMBD))
        w.add_tensor(p + "ffn_gate.weight", mat(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", mat(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", mat(N_EMBD, N_FF))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
