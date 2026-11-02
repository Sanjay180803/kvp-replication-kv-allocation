"""CPU smoke test of profile_heads.py on a tiny random Qwen2 (same GQA ratio 7:1)."""
import io, csv, sys, types, torch
sys.path.insert(0, ".")
import profile_heads as ph
from transformers import Qwen2Config, Qwen2ForCausalLM

torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=500, hidden_size=112, intermediate_size=224, num_hidden_layers=4,
                  num_attention_heads=14, num_key_value_heads=2, max_position_embeddings=4096,
                  attn_implementation=ph.ATTN_NAME, tie_word_embeddings=False)
model = Qwen2ForCausalLM(cfg).float().eval()
model.generation_config.eos_token_id = 499
tok = types.SimpleNamespace(pad_token_id=0, eos_token_id=499)
prompts = [torch.randint(1, 498, (1, P)) for P in (700, 900)]

ok = ph.sanity_checks(model, tok, prompts, n_prompts=2)
buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=ph.CSV_FIELDS); w.writeheader()
rows, _ = ph.profile_prompt(model, tok, dict(prompt_id=400, task="t0"), prompts[0],
                            ["snapkv", "streaming"], ph.RATIOS, batch_size=7, writer=w)
print(len(rows), "rows;", buf.getvalue().splitlines()[1])
# batching must not change results: rerun with batch 1 for a few heads
rows1, _ = ph.profile_prompt(model, tok, dict(prompt_id=400, task="t0"), prompts[0],
                             ["snapkv"], (0.1,), batch_size=1)
a = {(r["layer"], r["kv_head"]): float(r["kl_mean"]) for r in rows if r["scorer"] == "snapkv" and r["keep_ratio"] == 0.1}
b = {(r["layer"], r["kv_head"]): float(r["kl_mean"]) for r in rows1}
print("batch-invariance max diff:", max(abs(a[k] - b[k]) for k in a))
assert ok
