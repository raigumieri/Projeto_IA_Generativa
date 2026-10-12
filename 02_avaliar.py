"""
02_avaliar.py - Aplica a prova (dados/teste.jsonl) a um modelo e calcula a nota.

A mesma prova serve para o "antes" e para o "depois":
  python 02_avaliar.py --nome base_360m
  python 02_avaliar.py --nome lora_r16 --adaptador runs/lora_r16/adaptador
  python 02_avaliar.py --nome base_135m --modelo HuggingFaceTB/SmolLM2-135M-Instruct

Saida: tabela no console + resultados/<nome>.json (usado depois na comparacao).
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from comum import MODELO_PADRAO, carregar, classificar, ler_jsonl, pico_vram_mb, resumo_ambiente

if hasattr(sys.stdout, "reconfigure"):      # acentos corretos no console do Windows
    sys.stdout.reconfigure(encoding="utf-8")


def pct(acertos: int, total: int) -> float:
    return round(100 * acertos / total, 1) if total else 0.0


def resumir(itens: list[dict]) -> dict:
    """itens: dicts com 'rotulo' (gabarito), 'previsto', 'tipo' e 'familia'."""
    def acuracia(sub):
        return dict(acertos=sum(i["previsto"] == i["rotulo"] for i in sub), total=len(sub),
                    pct=pct(sum(i["previsto"] == i["rotulo"] for i in sub), len(sub)))

    por_tipo, por_familia = defaultdict(list), defaultdict(list)
    for i in itens:
        por_tipo[i["tipo"]].append(i)
        por_familia[i["familia"]].append(i)

    # "golpe" e a classe positiva: o erro mais grave e o golpe que passa como seguro
    vp = sum(i["rotulo"] == "golpe" and i["previsto"] == "golpe" for i in itens)
    fn = sum(i["rotulo"] == "golpe" and i["previsto"] != "golpe" for i in itens)
    fp = sum(i["rotulo"] == "seguro" and i["previsto"] == "golpe" for i in itens)
    precisao = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0

    return dict(
        geral=acuracia(itens),
        por_tipo={t: acuracia(s) for t, s in sorted(por_tipo.items())},
        por_familia={f: acuracia(s) for f, s in sorted(por_familia.items())},
        golpe=dict(precisao=round(100 * precisao, 1), recall=round(100 * recall, 1), f1=round(100 * f1, 1)),
        confusao=dict(Counter(f"{i['rotulo']} -> {i['previsto']}" for i in itens)),
        invalidas=sum(i["previsto"] == "invalido" for i in itens),
    )


def imprimir(nome: str, r: dict) -> None:
    g = r["geral"]
    print(f"\n===== {nome} =====")
    print(f"Acuracia geral : {g['acertos']}/{g['total']} = {g['pct']}%")
    for tipo, a in r["por_tipo"].items():
        print(f"  {tipo:<9}: {a['acertos']}/{a['total']} = {a['pct']}%")
    print(f"Classe golpe   : precisao {r['golpe']['precisao']}% | recall {r['golpe']['recall']}% | F1 {r['golpe']['f1']}%")
    print(f"Respostas fora do formato (nem golpe nem seguro): {r['invalidas']}")
    print("Confusao (gabarito -> previsto):")
    for chave, n in sorted(r["confusao"].items()):
        print(f"  {chave:<22}{n:>4}")
    print("Por familia:")
    for familia, a in r["por_familia"].items():
        print(f"  {familia:<28}{a['acertos']:>3}/{a['total']:<3}{a['pct']:>7}%")


def main() -> None:
    p = argparse.ArgumentParser(description="Avalia um modelo no conjunto de teste.")
    p.add_argument("--nome", required=True, help="nome do experimento (vira resultados/<nome>.json)")
    p.add_argument("--modelo", default=MODELO_PADRAO)
    p.add_argument("--adaptador", default=None, help="pasta do adaptador LoRA treinado")
    p.add_argument("--teste", default="dados/teste.jsonl")
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    ambiente = resumo_ambiente()
    tok, llm = carregar(args.modelo, args.adaptador)
    itens = ler_jsonl(args.teste)

    inicio = time.time()
    saidas = classificar(tok, llm, [i["mensagem"] for i in itens], temperature=args.temperature)
    segundos = round(time.time() - inicio, 1)
    for item, (previsto, bruto) in zip(itens, saidas):
        item["previsto"], item["resposta"] = previsto, bruto

    resumo = resumir(itens)
    imprimir(args.nome, resumo)
    print(f"\nTempo: {segundos}s para {len(itens)} mensagens | pico de VRAM: {pico_vram_mb()} MB")

    erros = [i for i in itens if i["previsto"] != i["rotulo"]]
    if erros:
        print("\nExemplos de erro:")
        for i in erros[:5]:
            print(f"  [{i['rotulo']} -> {i['previsto']}] resposta={i['resposta']!r}\n    {i['mensagem']}")

    Path("resultados").mkdir(exist_ok=True)
    destino = Path("resultados") / f"{args.nome}.json"
    with destino.open("w", encoding="utf-8") as f:
        json.dump(dict(nome=args.nome, modelo=args.modelo, adaptador=args.adaptador,
                       temperature=args.temperature, ambiente=ambiente, segundos=segundos,
                       pico_vram_mb=pico_vram_mb(), resumo=resumo, itens=itens),
                  f, ensure_ascii=False, indent=2)
    print(f"\nSalvo em {destino}")


if __name__ == "__main__":
    main()
