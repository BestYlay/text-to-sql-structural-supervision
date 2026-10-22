import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/runtime"
MODEL=Path(os.environ["TEXT2SQL_MODEL"]) if "TEXT2SQL_MODEL" in os.environ else ROOT/"models/Qwen3-4B-Instruct-2507"
TARGETS=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","max_split_size_mb:128")

def read(path):
    return json.loads(path.read_text("utf-8"))

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda v: sorted(v) if isinstance(v, set) else str(v)), encoding="utf-8")
    temp.replace(path)

def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

def load(train=False, adapter=None, *, use_dora=False):
    import torch
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    set_seed(3407)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model, loading = AutoModelForCausalLM.from_pretrained(
        MODEL, config=cfg, dtype=torch.float16, quantization_config=quant,
        device_map={"": 0}, attn_implementation="sdpa", local_files_only=True, output_loading_info=True)
    save(OUT / ("loading-train.json" if train else "loading-inference.json"), loading)
    assert not loading.get("missing_keys") and not loading.get("mismatched_keys"), loading
    # Identical non-quantized dtypes for before-save and fresh reload inference.
    # No global autocast: native DeltaNet's explicit FP32 operations stay FP32.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    assert all(m.compute_dtype == torch.float16 for m in model.modules() if isinstance(m, bnb.nn.Linear4bit))
    if train:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0,
                    bias="none", task_type="CAUSAL_LM", target_modules=TARGETS, use_dora=use_dora))
    elif adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
        if use_dora:
            assert model.peft_config["default"].use_dora, "Expected a DoRA adapter"
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert "lora_" in name and p.dtype == torch.float32, (name, p.dtype)
    model.config.use_cache = not train
    targets = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    modules = {
        "class": type(model).__name__, "linear_attention_layers": 0, "full_attention_layers": 36,
        "lora_modules": targets, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "quantized_linear_count": sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules()),
        "nonquantized_dtypes": sorted({str(p.dtype) for p in model.parameters() if not isinstance(p, bnb.nn.Params4bit)}),
        "precision_note": "NF4 matmuls FP16; unquantized parameters FP32 via PEFT preparation; no global autocast",
        "fast_kernel_packages": False,
        "dora_modules": sum(bool(getattr(m, "use_dora", {}).get("default", False)) for m in model.modules()),
    }
    save(OUT / ("modules-train.json" if train else "modules-inference.json"), modules)
    print("MODEL_LOADED", json.dumps({k:v for k,v in modules.items() if k != "lora_modules"}), flush=True)
    return model, tokenizer()
