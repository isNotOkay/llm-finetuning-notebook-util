import os
import subprocess
import sys
import torch
from typing import Optional


def install_train_deps_if_colab():
    if "COLAB_" not in "".join(os.environ.keys()):
        print("👉 Not running in Colab — no packages installed.")
        return
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    subprocess.check_call(pip + ["--no-deps",
                                 "bitsandbytes", "accelerate", "xformers==0.0.29.post3",
                                 "peft", "trl", "triton", "cut_cross_entropy", "unsloth_zoo"
                                 ])
    subprocess.check_call(pip + [
        "sentencepiece", "protobuf",
        "datasets>=3.4.1,<4.0.0", "huggingface_hub>=0.34.0", "hf_transfer"
    ])
    subprocess.check_call(pip + ["--no-deps", "unsloth"])
    print("✅ Training dependencies installed (Colab).")


def get_hf_token(key: str = "HF_TOKEN") -> str:
    try:
        from google.colab import userdata  # type: ignore
        token = userdata.get(key)
    except Exception:
        token = os.environ.get(key)
    assert token, f"{key} not found in Colab secrets or environment."
    print("✅ HF token loaded.")
    return token

def train(trainer):
    """
    Run training and print concise GPU/runtime stats.
    """

    # --- private helpers ---
    def _show_gpu_stats():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
        return start_gpu_memory, max_memory

    def _show_final_stats(trainer_stats, start_gpu_memory: float, max_memory: float):
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

        runtime = trainer_stats.metrics.get("train_runtime")
        if runtime is not None:
            print(f"{runtime} seconds used for training.")
            print(f"{round(runtime/60, 2)} minutes used for training.")

        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # --- run ---
    start_gpu_memory, max_memory = _show_gpu_stats()
    trainer_stats = trainer.train()
    _show_final_stats(trainer_stats, start_gpu_memory, max_memory)



def push_to_hub(model, tokenizer, repo_name: str, hf_token: str):
    """
    Push model + tokenizer to Hugging Face Hub.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        repo_name: HuggingFace repo name (e.g. "user/repo")
        hf_token: Hugging Face API token (string)
    """
    assert hf_token, "hf_token is required!"
    model.push_to_hub(repo_name, use_auth_token=hf_token)
    tokenizer.push_to_hub(repo_name, use_auth_token=hf_token)
    print(f"✅ Model + tokenizer pushed to Hub: {repo_name}")


# ---- Dependency-injected helpers (no heavy imports here) ----
def resolve_base_from_peft(repo_id: str, revision: Optional[str], token: Optional[str], *, PeftConfig):
    cfg = PeftConfig.from_pretrained(repo_id, revision=revision, token=token)
    base = cfg.base_model_name_or_path
    print(f"Base model from PEFT config: {base}")
    return base


def prepare_model_for_inference(base_model, repo_id: str, *, revision: str, token: str, PeftModel, FastLanguageModel):
    model = PeftModel.from_pretrained(base_model, repo_id, revision=revision, token=token)
    model = FastLanguageModel.for_inference(model)
    model.eval()
    print("✅ Loaded private PEFT repo + tokenizer. Ready for inference.")
    return model


def run_batched_inference(
        prompts: list[str],
        *,
        model,
        tokenizer,
        torch,  # injected
        GenerationConfig,  # injected
        batch_size: int = 32,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        print_results: bool = True,
):
    """
    Batched chat inference for any HF chat model (Unsloth-compatible).
    - Builds chat prompts via tokenizer.apply_chat_template (supports string OR block content)
    - Tokenizes with padding, runs .generate, strips prompt via attention_mask
    - Decodes with skip_special_tokens=True
    Returns: (questions, answers)
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    device = next(model.parameters()).device
    import contextlib
    amp = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()

    # --- renderer that supports both styles
    def render_chat_text(prompt: str) -> str:
        # Try simple string content first (Llama 3.x etc.)
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            # Fallback: content blocks (Gemma-style)
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                # Last-resort plain prompt if no chat template
                return f"User: {prompt}\nAssistant:"

    all_questions, all_answers = [], []

    with torch.inference_mode(), amp:
        for s in range(0, len(prompts), batch_size):
            batch_prompts = prompts[s:s + batch_size]

            chat_texts = [render_chat_text(p) for p in batch_prompts]

            inputs = tokenizer(chat_texts, return_tensors="pt", padding=True).to(device)
            outputs = model.generate(**inputs, generation_config=gen_cfg, use_cache=True)

            input_lens = inputs["attention_mask"].sum(dim=1)
            for i, out in enumerate(outputs):
                gen_only = out[input_lens[i]:]
                text = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
                all_questions.append(batch_prompts[i])
                all_answers.append(text)

    if print_results:
        for idx, (q, a) in enumerate(zip(all_questions, all_answers), 1):
            print(f"Q{idx}: {q}\nA{idx}: {a}\n{'-' * 50}")

    return all_questions, all_answers


def run_openrouter_eval(
        eval_model: str,
        questions: list[str],
        answers: list[str],
        *,
        url: str = "https://openrouter.ai/api/v1/chat/completions",
):
    """
    Run evaluation of (question, answer) pairs via OpenRouter.
    Prints detailed results + summary stats.
    """
    import os, json, time, requests, re
    from typing import Literal

    EvalLabel = Literal["correct", "incorrect", "not_relevant"]

    # --- API key
    try:
        from google.colab import userdata  # type: ignore
        api_key = userdata.get("OPENROUTER_API_KEY")
    except Exception:
        api_key = os.environ.get("OPENROUTER_API_KEY")
    assert api_key, "OPENROUTER_API_KEY not found in Colab secrets or environment!"
    print("✅ OpenRouter API key loaded.")

    # --- Helpers
    def build_eval_messages(question: str, answer: str):
        system = (
            "You are evaluating answers about a SQLite User Defined Function (UDF) called OD_Feiertag().\n"
            "Facts:\n"
            "- OD_Feiertag() is a custom SQLite UDF, not built-in.\n"
            "- It takes a date as input.\n"
            "- It returns the name of a German holiday if the date is a holiday.\n"
            "- If the date is not a holiday, it returns an empty string.\n"
            "- It never returns numbers, booleans, or NULL.\n\n"
            "Rules:\n"
            "1) If the answer matches these facts → label=correct.\n"
            "2) If the answer mentions OD_Feiertag but is wrong/misleading → label=incorrect and give a short reason (≤30 words).\n"
            "3) If the answer is unrelated to OD_Feiertag → label=not_relevant.\n"
            "4) Return ONLY a single JSON object exactly like: "
            '{"label":"correct|incorrect|not_relevant","reason":""}'
        )
        user = (
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}\n\n"
            "Return exactly one JSON object with keys: label, reason."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def strip_fences(s: str) -> str:
        s = s.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        return s.strip()

    def post_openrouter(payload: dict, retries: int = 5, backoff: float = 1.5) -> dict:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "unsloth-odfeiertag-evaluator",
        }
        last_text, last_code = "", 0
        for attempt in range(retries):
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            last_code, last_text = resp.status_code, resp.text[:400]
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff ** attempt)
                continue
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:400]}")
        raise RuntimeError(f"OpenRouter retry limit exceeded. Last: {last_code} {last_text}")

    # --- Colors
    RESET = "\033[0m";
    GREEN = "\033[92m";
    RED = "\033[91m";
    GRAY = "\033[90m"

    def paint(label: str, s: str) -> str:
        return (GREEN if label == "correct" else RED if label == "incorrect" else GRAY) + s + RESET

    # --- Evaluation loop
    labels: list[str] = []
    num_empty = 0

    for idx, (q, a) in enumerate(zip(questions, answers), 1):
        ans = "" if a is None else str(a)
        if ans.strip() == "":
            label, reason = "incorrect", "Answer is empty"
        else:
            payload = {"model": eval_model, "messages": build_eval_messages(q, ans), "temperature": 0,
                       "max_tokens": 120}
            try:
                data = post_openrouter(payload)
                content = data["choices"][0]["message"]["content"]
                obj = json.loads(strip_fences(content))
                label = str(obj.get("label", "")).lower()
                reason = str(obj.get("reason", "")).strip()
            except Exception:
                label, reason = "not_relevant", ""

            if label not in ("correct", "incorrect", "not_relevant"):
                label, reason = "not_relevant", ""
            elif label != "incorrect":
                reason = ""
            else:
                reason = reason[:400]

        if label == "incorrect" and reason == "Answer is empty":
            num_empty += 1
        labels.append(label)
        verdict = paint(label, f"[{label.upper()}]")
        print(f"Q{idx}: {q}\nA{idx} {verdict}: {a}")
        if label == "incorrect" and reason:
            print(paint("incorrect", f"Reason: {reason}"))
        print("-" * 50)

    # --- Summary
    total = len(labels)
    num_correct = sum(1 for x in labels if x == "correct")
    num_incorrect = sum(1 for x in labels if x == "incorrect")
    num_not_rel = sum(1 for x in labels if x == "not_relevant")
    print("\n=== Evaluation summary ===")
    print(paint("correct", f"correct       : {num_correct}/{total}"))
    print(paint("incorrect", f"incorrect     : {num_incorrect}/{total}"))
    print(paint("not_relevant", f"not_relevant  : {num_not_rel}/{total}"))
    print(paint("incorrect", f"empty answers : {num_empty}"))
    relevant = num_correct + num_incorrect
    if relevant:
        print(f"Accuracy on relevant items: {num_correct / relevant:.1%}  (out of {relevant} relevant items)")
