cat > merge_lora_to_16bit.py << 'EOF'
#!/usr/bin/env python3
import os

os.environ.setdefault("SAFETENSORS_FAST_LOAD", "0")  # ← mmap を使わない読み込みに切替
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")   # 念のためコンパイル系も停止
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("HF_USE_FLEX_ATTENTION", "0")

import argparse, sys, time, json, platform, socket, getpass, datetime
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, __version__ as tf_ver
from accelerate.hooks import remove_hook_from_submodules

try:
    from peft import PeftModel, __version__ as peft_ver
except Exception as e:
    print("peft が必要です: pip install peft", file=sys.stderr); raise

# 分散検出とバリア用（未インストールでも単独で動くようにフォールバック）
try:
    from accelerate import PartialState, __version__ as acc_ver
    HAVE_ACCELERATE = True
except Exception:
    HAVE_ACCELERATE = False
    acc_ver = None
    class _DummyState:
        process_index = 0
        num_processes = 1
        def wait_for_everyone(self): pass
    PartialState = _DummyState  # type: ignore

def parse_max_memory(s: Optional[str]) -> Optional[Dict[str, str]]:
    if not s:
        return None
    out = {}
    for part in s.split(","):
        k, v = part.split(":")
        k = k.strip()
        v = v.strip()
        # "0","1",... -> 0,1,... にキャスト。cpu/disk/mps はそのまま。
        key = int(k) if k.isdigit() else k
        out[key] = v
    return out

def load_tokenizer(base_model_dir: str, lora_dir: str):
    # 通常はベース側のトークナイザでOK。無ければLoRA側をフォールバック。
    try:
        tok = AutoTokenizer.from_pretrained(base_model_dir, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(lora_dir, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok

def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")

def ensure_rank_offload_dir(root: Optional[str]) -> Optional[str]:
    if not root: return None
    job = os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID") or "local"
    user = os.environ.get("USER", getpass.getuser() or "user")
    # accelerateが無い場合は rank=0 扱い
    state = PartialState() if HAVE_ACCELERATE else PartialState
    rank = getattr(state, "process_index", 0)
    off = os.path.join(root, user, "offload_cache", str(job), f"rank_{rank}")
    os.makedirs(off, exist_ok=True)
    return off

def summarize_device_map(m) -> Dict[str, int]:
    mp = getattr(m, "hf_device_map", None)
    if not mp: return {}
    counts = {}
    for _, dev in mp.items():
        counts[dev] = counts.get(dev, 0) + 1
    return counts

def write_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def base_metadata(
    base_model: str, lora_dir: str, out_dir: str, dtype16: str, compute_dtype: Optional[str],
    device_map, max_memory, offload_root, offload_folder, rank: int, world: int, phase: str
):
    return {
        "phase": phase,  # "merged_fp16" or "requantized_4bit"
        "merged_at": now_iso(),
        "host": socket.gethostname(),
        "user": os.environ.get("USER", getpass.getuser() or "user"),
        "base_model": os.path.basename(base_model.rstrip("/")) if os.path.isdir(base_model) else base_model,
        "base_model_path": os.path.abspath(base_model),
        "adapter_path": os.path.abspath(lora_dir) if lora_dir else None,
        "output_dir": os.path.abspath(out_dir),
        "dtype16": dtype16,
        "four_bit": (compute_dtype is not None),
        "compute_dtype": compute_dtype,
        "device_map": device_map,
        "max_memory": max_memory,
        "offload_root": offload_root,
        "offload_folder_rank": offload_folder,
        "rank": rank,
        "world_size": world,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": tf_ver,
            "peft": peft_ver,
            "accelerate": acc_ver,
            "platform": platform.platform(),
        },
    }

def merge_lora_to_16bit(
    base_model: str,
    lora_dir: str,
    out_dir_fp16: str,
    dtype16: str = "bf16",
    trust_remote_code: bool = False,
    device_map: Optional[str] = "auto",
    max_memory: Optional[Dict[str, str]] = None,
    offload_root: Optional[str] = None,
):
    os.makedirs(out_dir_fp16, exist_ok=True)
    dtype = torch.bfloat16 if dtype16.lower() == "bf16" else torch.float16

    # 分散状態
    state = PartialState() if HAVE_ACCELERATE else PartialState
    rank = getattr(state, "process_index", 0)
    world = getattr(state, "num_processes", 1)
    is_rank0 = (rank == 0)

    # ランク専用 offload ディレクトリ
    offload_folder = ensure_rank_offload_dir(offload_root)

    # 0) メトリクス（時間）
    metrics = {}

    # 1) ベースモデルを bf16/fp16 でロード（シャーディング＋オフロード）
    print(f"[INFO][rank {rank}/{world}] Loading base model: {base_model}")
    t0 = time.monotonic()

    # base = AutoModelForCausalLM.from_pretrained(
    #     base_model,
    #     trust_remote_code=trust_remote_code,
    #     torch_dtype=dtype,
    #     device_map=device_map,
    #     max_memory=max_memory,          # ← "disk" は入れない
    #     offload_folder=offload_folder,  # ← NVMeパス
    #     #offload_state_dict=True,        # ★ ロード中の一時RAMをNVMeへ
    #     low_cpu_mem_usage=True,
    #     use_safetensors=True,
    # )

    # base = AutoModelForCausalLM.from_pretrained(
    #     base_model,
    #     trust_remote_code=True,
    #     torch_dtype=dtype,
    #     device_map="auto",
    #     # max_memory は渡さない
    #     offload_state_dict=True,      # 一時RAM→NVMe退避（offload_root は撤去可／任意）
    #     low_cpu_mem_usage=True,
    #     use_safetensors=True,
    # )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        # max_memory は使わない（レシピB）
        offload_state_dict=True,
        low_cpu_mem_usage=False,   # ★ mmapを避ける
        use_safetensors=True,
    )

    try:
        remove_hook_from_submodules(base)
        print("[INFO] accelerate hooks removed -> merging on current devices (CPU/offload).")
    except Exception as e:
        print("[WARN] could not remove accelerate hooks:", e)
    
    #base_device_map = getattr(base, "hf_device_map", "auto")
    base_device_map = getattr(base, "hf_device_map", None)

    assert offload_folder is not None, "offload_folder must not be None"

    metrics["t_load_base_sec"] = round(time.monotonic() - t0, 3)
    dev_summary = summarize_device_map(base)
    if dev_summary:
        print(f"[INFO][rank {rank}] device_map summary: {dev_summary}")

    tok = load_tokenizer(base_model, lora_dir)

    # 2) Rank0 だけが LoRA をアタッチ→merge→保存
    if is_rank0:
        print(f"[INFO][rank 0] Attaching LoRA adapter from: {lora_dir}")
        t_attach = time.monotonic()
        model = PeftModel.from_pretrained(base,
                                         lora_dir, 
                                         is_trainable=False,
                                         device_map="auto",
                                         offload_folder=offload_folder,
                                         offload_state_dict=True,
                                         torch_dtype=dtype, 
                                         )
        metrics["t_attach_lora_sec"] = round(time.monotonic() - t_attach, 3)

        print("[INFO][rank 0] Merging LoRA -> base (merge_and_unload)")
        t_merge = time.monotonic()
        model = model.merge_and_unload()
        metrics["t_merge_sec"] = round(time.monotonic() - t_merge, 3)

        print(f"[INFO][rank 0] Saving merged 16-bit model to: {out_dir_fp16}")
        t_save = time.monotonic()
        model.save_pretrained(out_dir_fp16, safe_serialization=True, max_shard_size="5GB")
        tok.save_pretrained(out_dir_fp16)
        metrics["t_save_fp16_sec"] = round(time.monotonic() - t_save, 3)

        # メタデータ（fp16側）
        md = base_metadata(
            base_model, lora_dir, out_dir_fp16, dtype16,
            compute_dtype=None, device_map=device_map, max_memory=max_memory,
            offload_root=offload_root, offload_folder=offload_folder,
            rank=rank, world=world, phase="merged_fp16"
        )
        md["timings_sec"] = metrics
        md["device_map_summary"] = dev_summary
        write_json(os.path.join(out_dir_fp16, "metadata.json"), md)

    # 3) バリア：他ランクは保存完了を待つ
    if HAVE_ACCELERATE:
        state.wait_for_everyone()

    # 4) 全ランクで検証ロード（純粋CausalLMとして）
    print(f"[INFO][rank {rank}] Reload quick check from merged dir...")
    t_reload = time.monotonic()
    _ = AutoModelForCausalLM.from_pretrained(
        out_dir_fp16,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=offload_folder,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    metrics["t_reload_check_sec"] = round(time.monotonic() - t_reload, 3)
    print(f"[OK][rank {rank}] 16-bit merged model ready. (reload {metrics['t_reload_check_sec']}s)")
    return out_dir_fp16, metrics, dev_summary, offload_folder, rank, world

def requantize_to_4bit(
    merged_fp16_dir: str,
    out_dir_4bit: str,
    trust_remote_code: bool = False,
    compute_dtype: str = "bf16",
    device_map: Optional[str] = "auto",
    max_memory: Optional[Dict[str, str]] = None,
    offload_root: Optional[str] = None,
    base_metadata_stub: Optional[Dict] = None,
):
    os.makedirs(out_dir_4bit, exist_ok=True)

    # ランク専用 offload
    offload_folder = ensure_rank_offload_dir(offload_root)

    print("[INFO] Quantizing merged model to 4bit (nf4 + double quant)")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if compute_dtype.lower()=="bf16" else torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(merged_fp16_dir, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    metrics = {}
    t_qload = time.monotonic()
    model4 = AutoModelForCausalLM.from_pretrained(
        merged_fp16_dir,
        trust_remote_code=trust_remote_code,
        quantization_config=bnb,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=offload_folder,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    metrics["t_load_4bit_sec"] = round(time.monotonic() - t_qload, 3)

    t_qsave = time.monotonic()
    model4.save_pretrained(out_dir_4bit, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(out_dir_4bit)
    metrics["t_save_4bit_sec"] = round(time.monotonic() - t_qsave, 3)

    print("[OK] 4bit model saved to:", out_dir_4bit)

    # メタデータ（4bit側）
    if base_metadata_stub:
        md4 = base_metadata_stub.copy()
        md4.update({
            "phase": "requantized_4bit",
            "merged_at": now_iso(),
            "output_dir": os.path.abspath(out_dir_4bit),
            "four_bit": True,
            "compute_dtype": compute_dtype,
            "timings_sec_4bit": metrics,
        })
        write_json(os.path.join(out_dir_4bit, "metadata.json"), md4)

    return out_dir_4bit, metrics

def main():
    ap = argparse.ArgumentParser(description="Merge LoRA into base model and save (16-bit), optionally re-quantize to 4bit.")
    ap.add_argument("--base-model", required=True, help="HF repo id or local path to base (bf16/fp16) model")
    ap.add_argument("--lora-dir", required=True, help="Dir containing adapter_config.json & adapter_model.*")
    ap.add_argument("--out-fp16", required=True, help="Output dir for merged 16-bit model")
    ap.add_argument("--out-4bit", help="(Optional) Output dir for re-quantized 4bit model")
    ap.add_argument("--dtype16", choices=["bf16","fp16"], default="bf16")
    ap.add_argument("--compute-dtype", choices=["bf16","fp16"], default="bf16")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--device-map", default="auto", help='e.g., "auto" or None')
    ap.add_argument("--max-memory", help='e.g., "0:78GiB,1:78GiB,cpu:256GiB"')
    ap.add_argument("--offload-root", help="Root dir for per-rank offload (e.g., /nvme12/P10U029)")
    args = ap.parse_args()

    max_mem = parse_max_memory(args.max_memory)

    # === Merge to 16-bit ===
    out_fp16, metrics_fp16, dev_summary, off_rank_dir, rank, world = merge_lora_to_16bit(
        base_model=args.base_model,
        lora_dir=args.lora_dir,
        out_dir_fp16=args.out_fp16,
        dtype16=args.dtype16,
        trust_remote_code=args.trust_remote_code,
        device_map=(None if args.device_map.lower()=="none" else args.device_map),
        max_memory=max_mem,
        offload_root=args.offload_root,
    )

    # base metadata stub reused for 4bit metadata
    md_stub = base_metadata(
        base_model=args.base_model,
        lora_dir=args.lora_dir,
        out_dir=out_fp16,
        dtype16=args.dtype16,
        compute_dtype=None,
        device_map=args.device_map,
        max_memory=max_mem,
        offload_root=args.offload_root,
        offload_folder=off_rank_dir,
        rank=rank, world=world, phase="merged_fp16"
    )
    md_stub["device_map_summary"] = dev_summary
    md_stub["timings_sec"] = metrics_fp16

    # === Optional: Re-quantize to 4bit ===
    if args.out_4bit:
        try:
            _out4, metrics4 = requantize_to_4bit(
                merged_fp16_dir=out_fp16,
                out_dir_4bit=args.out_4bit,
                trust_remote_code=args.trust_remote_code,
                compute_dtype=args.compute_dtype,
                device_map=(None if args.device_map.lower()=="none" else args.device_map),
                max_memory=max_mem,
                offload_root=args.offload_root,
                base_metadata_stub=md_stub,
            )
        except Exception as e:
            print("[WARN] 4bit再量子化に失敗しました。16bit統合モデルは正常に保存されています。詳細:", e, file=sys.stderr)

if __name__ == "__main__":
    main()
EOF