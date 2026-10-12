"""
03_treinar.py - Fine-tuning do detector de golpes com LoRA, na GPU.

O laco e o mesmo da aula: forward -> loss -> backward -> step.
A loss so cobra a RESPOSTA ("golpe" / "seguro"); o enunciado e ignorado (-100).

Exemplos (cada execucao e uma linha da tabela de experimentos):
  python 03_treinar.py --nome lora_padrao
  python 03_treinar.py --nome lora_lr5e-5 --lr 5e-5
  python 03_treinar.py --nome lora_r4 --r 4
  python 03_treinar.py --nome lora_1ep --epocas 1
  python 03_treinar.py --nome lora_60amostras --amostras 60
  python 03_treinar.py --nome lora_135m --modelo HuggingFaceTB/SmolLM2-135M-Instruct

Saidas em runs/<nome>/: adaptador/ (pesos LoRA), treino.json e loss.png
Depois:  python 02_avaliar.py --nome <nome> --adaptador runs/<nome>/adaptador [--modelo ...]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

from comum import DEVICE, DTYPE, MODELO_PADRAO, carregar, ler_jsonl, montar_mensagens, pico_vram_mb, resumo_ambiente

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODULOS_LORA = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def preparar(tok, item: dict) -> tuple[list[int], list[int]]:
    """Devolve (input_ids, labels). Labels do enunciado = -100 -> nao entram na loss."""
    prompt = tok.apply_chat_template(montar_mensagens(item["mensagem"]), tokenize=False, add_generation_prompt=True)
    ids_prompt = tok(prompt, add_special_tokens=False)["input_ids"]
    ids_resposta = tok(item["rotulo"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    return ids_prompt + ids_resposta, [-100] * len(ids_prompt) + ids_resposta


def montar_lote(exemplos: list[tuple[list[int], list[int]]], pad_id: int) -> dict:
    maior = max(len(ids) for ids, _ in exemplos)
    input_ids, mascara, labels = [], [], []
    for ids, lab in exemplos:
        falta = maior - len(ids)
        input_ids.append(ids + [pad_id] * falta)
        mascara.append([1] * len(ids) + [0] * falta)
        labels.append(lab + [-100] * falta)
    return dict(input_ids=torch.tensor(input_ids, device=DEVICE),
                attention_mask=torch.tensor(mascara, device=DEVICE),
                labels=torch.tensor(labels, device=DEVICE))


def forward(llm, lote: dict) -> torch.Tensor:
    # pesos em float32 (atualizacoes precisas) e contas em bf16 na GPU (rapido e leve)
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=DTYPE == torch.bfloat16):
        return llm(**lote).loss


@torch.no_grad()
def loss_media(llm, exemplos, tamanho_lote, pad_id) -> float:
    llm.eval()
    soma = 0.0
    for i in range(0, len(exemplos), tamanho_lote):
        parte = exemplos[i:i + tamanho_lote]
        soma += forward(llm, montar_lote(parte, pad_id)).item() * len(parte)
    llm.train()
    return soma / len(exemplos)


def main() -> None:
    p = argparse.ArgumentParser(description="Treina um adaptador LoRA para o detector de golpes.")
    p.add_argument("--nome", required=True, help="nome do experimento (vira runs/<nome>/)")
    p.add_argument("--modelo", default=MODELO_PADRAO)
    p.add_argument("--treino", default="dados/treino.jsonl")
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    p.add_argument("--r", type=int, default=16, help="rank do LoRA")
    p.add_argument("--alpha", type=int, default=None, help="lora_alpha (padrao: 2 x r)")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lote", type=int, default=16, help="tamanho do lote (batch size)")
    p.add_argument("--validacao", type=float, default=0.1, help="fracao do treino separada para validacao")
    p.add_argument("--amostras", type=int, default=None, help="usa so N exemplos do treino (para testar com menos dados)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.alpha = args.alpha or 2 * args.r

    from peft import LoraConfig, get_peft_model
    from transformers import get_linear_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    ambiente = resumo_ambiente()

    tok, llm = carregar(args.modelo, dtype=torch.float32)
    cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                     target_modules=MODULOS_LORA, task_type="CAUSAL_LM")
    llm = get_peft_model(llm, cfg)
    treinaveis = sum(x.numel() for x in llm.parameters() if x.requires_grad)
    total = sum(x.numel() for x in llm.parameters())
    print(f"Parametros treinaveis: {treinaveis:,} de {total:,} ({100 * treinaveis / total:.2f}%)")

    # A validacao sai do TREINO. O teste.jsonl nao e tocado aqui.
    exemplos = [preparar(tok, item) for item in ler_jsonl(args.treino)]
    random.shuffle(exemplos)
    if args.amostras:
        exemplos = exemplos[:args.amostras]
    n_val = int(len(exemplos) * args.validacao)
    validacao, treino = exemplos[:n_val], exemplos[n_val:]
    passos_total = args.epocas * -(-len(treino) // args.lote)
    print(f"Treino: {len(treino)} | validacao: {len(validacao)} | passos: {passos_total}")

    otimizador = torch.optim.AdamW([x for x in llm.parameters() if x.requires_grad], lr=args.lr)
    agenda = get_linear_schedule_with_warmup(otimizador, int(0.05 * passos_total), passos_total)

    llm.train()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    curva, por_epoca, passo, inicio = [], [], 0, time.time()
    for epoca in range(1, args.epocas + 1):
        random.shuffle(treino)
        for i in range(0, len(treino), args.lote):
            loss = forward(llm, montar_lote(treino[i:i + args.lote], tok.pad_token_id))   # forward -> loss
            loss.backward()                                                              # backward
            otimizador.step()                                                            # step
            agenda.step()
            otimizador.zero_grad()
            passo += 1
            curva.append(dict(passo=passo, loss=round(loss.item(), 4)))
            if passo % 10 == 0 or passo == passos_total:
                print(f"  epoca {epoca} | passo {passo:>4}/{passos_total} | loss {loss.item():.4f}")
        registro = dict(epoca=epoca, passo=passo)
        if validacao:
            registro["loss_validacao"] = round(loss_media(llm, validacao, args.lote, tok.pad_token_id), 4)
            print(f"== fim da epoca {epoca}: loss de validacao {registro['loss_validacao']:.4f}")
        por_epoca.append(registro)
    segundos = round(time.time() - inicio, 1)
    print(f"Treino concluido em {segundos}s | pico de VRAM: {pico_vram_mb()} MB")

    pasta = Path("runs") / args.nome
    pasta.mkdir(parents=True, exist_ok=True)
    llm.save_pretrained(pasta / "adaptador")
    with (pasta / "treino.json").open("w", encoding="utf-8") as f:
        json.dump(dict(parametros=vars(args), ambiente=ambiente, parametros_treinaveis=treinaveis,
                       parametros_total=total, segundos=segundos, pico_vram_mb=pico_vram_mb(),
                       curva=curva, epocas=por_epoca), f, ensure_ascii=False, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([c["passo"] for c in curva], [c["loss"] for c in curva], label="treino (por passo)")
        if validacao:
            ax.plot([e["passo"] for e in por_epoca], [e["loss_validacao"] for e in por_epoca],
                    marker="o", label="validação (por época)")
        ax.set_xlabel("passo"); ax.set_ylabel("loss"); ax.set_title(f"Curva de loss - {args.nome}")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(pasta / "loss.png", dpi=150)
    except ImportError:
        print("matplotlib nao instalado: grafico de loss nao foi gerado.")

    extra = "" if args.modelo == MODELO_PADRAO else f" --modelo {args.modelo}"
    print(f"\nSalvo em {pasta}\nAgora avalie com a mesma prova:")
    print(f"  python 02_avaliar.py --nome {args.nome} --adaptador {(pasta / 'adaptador').as_posix()}{extra}")


if __name__ == "__main__":
    main()