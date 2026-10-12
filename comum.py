"""
comum.py - Funcoes compartilhadas por todos os scripts do projeto
(avaliacao, treino e frontend usam exatamente o mesmo prompt e o mesmo
jeito de carregar o modelo, para que a comparacao antes x depois seja justa).
"""

import json
from pathlib import Path

import torch

MODELO_PADRAO = "HuggingFaceTB/SmolLM2-360M-Instruct"
ROTULOS = ("golpe", "seguro")

SISTEMA = (
    "Você é um detector de golpes em mensagens. Leia a mensagem e responda "
    "com apenas uma palavra: golpe ou seguro."
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float32


def resumo_ambiente() -> dict:
    """Imprime e devolve as informacoes da GPU (item 5 do enunciado)."""
    info = dict(pytorch=torch.__version__, device=DEVICE, dtype=str(DTYPE).replace("torch.", ""))
    if DEVICE == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    else:
        print("ATENCAO: CUDA indisponivel, rodando em CPU. O trabalho exige GPU.")
    print(" | ".join(f"{k}: {v}" for k, v in info.items()))
    return info


def pico_vram_mb() -> int:
    return round(torch.cuda.max_memory_allocated() / 1024**2) if DEVICE == "cuda" else 0


def montar_mensagens(mensagem: str, rotulo: str | None = None) -> list[dict]:
    """Conversa no formato de chat. Com `rotulo`, inclui a resposta (usado no treino)."""
    msgs = [
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": f"Mensagem: {mensagem}"},
    ]
    if rotulo is not None:
        msgs.append({"role": "assistant", "content": rotulo})
    return msgs


def carregar(modelo: str = MODELO_PADRAO, adaptador: str | None = None, dtype=None):
    """Carrega tokenizer e modelo na GPU. `adaptador` e a pasta de um LoRA treinado."""
    dtype = dtype or DTYPE
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(modelo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:                                    # transformers 5
        llm = AutoModelForCausalLM.from_pretrained(modelo, dtype=dtype)
    except TypeError:                       # transformers 4
        llm = AutoModelForCausalLM.from_pretrained(modelo, torch_dtype=dtype)
    if adaptador:
        from peft import PeftModel
        llm = PeftModel.from_pretrained(llm, adaptador)
    llm.to(device=DEVICE, dtype=dtype).eval()   # base e adaptador no mesmo dtype
    return tok, llm


def extrair_rotulo(texto: str) -> str:
    """Transforma a resposta livre do modelo em 'golpe', 'seguro' ou 'invalido'."""
    t = texto.lower()
    posicoes = {r: t.find(r) for r in ROTULOS if r in t}
    if not posicoes:
        return "invalido"
    return min(posicoes, key=posicoes.get)   # se citar os dois, vale o primeiro


@torch.no_grad()
def classificar(tok, llm, mensagens: list[str], lote: int = 16, max_new_tokens: int = 6,
                temperature: float = 0.0) -> list[tuple[str, str]]:
    """Devolve [(rotulo, resposta_bruta)] para cada mensagem. temperature=0 -> deterministico."""
    tok.padding_side = "left"               # obrigatorio para gerar em lote
    saidas = []
    for i in range(0, len(mensagens), lote):
        prompts = [
            tok.apply_chat_template(montar_mensagens(m), tokenize=False, add_generation_prompt=True)
            for m in mensagens[i:i + lote]
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
        amostragem = dict(do_sample=True, temperature=temperature) if temperature > 0 else dict(do_sample=False)
        out = llm.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **amostragem)
        for seq in out[:, enc["input_ids"].shape[1]:]:
            bruto = tok.decode(seq, skip_special_tokens=True).strip()
            saidas.append((extrair_rotulo(bruto), bruto))
    return saidas


def ler_jsonl(caminho: str | Path) -> list[dict]:
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(linha) for linha in f if linha.strip()]