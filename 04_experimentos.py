"""
04_experimentos.py - Roda a bateria de experimentos (item 3: testar diferentes parametros).

Cada experimento muda UM parametro em relacao ao "lora_padrao"
(3 epocas, lr 2e-4, r 16, 510 exemplos, SmolLM2-360M, temperature 0).
Para cada um: treina (se for o caso) e aplica as duas provas:
  dados/teste.jsonl   -> resultados/<nome>.json
  dados/desafio.jsonl -> resultados/<nome>__desafio.json

O que ja foi feito e pulado; use --refazer para rodar tudo de novo.
Depois:  python 05_comparar.py
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

M135 = "HuggingFaceTB/SmolLM2-135M-Instruct"

EXPERIMENTOS = [
    # sem treino (o "antes")
    dict(nome="base_135m", modelo=M135),
    dict(nome="base_360m"),
    # referencia
    dict(nome="lora_padrao", treino={}),
    # epocas
    dict(nome="lora_ep1", treino={"epocas": 1}),
    dict(nome="lora_ep6", treino={"epocas": 6}),
    # learning rate
    dict(nome="lora_lr1e-5", treino={"lr": 1e-5}),
    dict(nome="lora_lr5e-5", treino={"lr": 5e-5}),
    dict(nome="lora_lr1e-3", treino={"lr": 1e-3}),
    # rank do LoRA
    dict(nome="lora_r2", treino={"r": 2}),
    dict(nome="lora_r64", treino={"r": 64}),
    # quantidade de dados
    dict(nome="lora_60amostras", treino={"amostras": 60}),
    # tamanho do modelo
    dict(nome="lora_135m", modelo=M135, treino={}),
    # parametro de geracao: mesmo adaptador do lora_padrao, com sorteio na resposta
    dict(nome="lora_padrao_temp0.7", adaptador_de="lora_padrao", temperature=0.7),
    dict(nome="lora_padrao_temp1.5", adaptador_de="lora_padrao", temperature=1.5),
]

PROVAS = [("dados/teste.jsonl", ""), ("dados/desafio.jsonl", "__desafio")]


def rodar(comando: list[str]) -> None:
    print("\n>>>", " ".join(comando[1:]), flush=True)
    subprocess.run(comando, check=True)


def mensagens(caminho: str) -> set[str]:
    with open(caminho, encoding="utf-8") as f:
        return {json.loads(linha)["mensagem"] for linha in f if linha.strip()}


def resultado_valido(destino: Path, prova: str) -> bool:
    """So reaproveita um resultado se ele foi medido na versao atual da prova."""
    if not destino.exists():
        return False
    with destino.open(encoding="utf-8") as f:
        return {i["mensagem"] for i in json.load(f)["itens"]} == mensagens(prova)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--refazer", action="store_true", help="ignora o que ja existe e roda tudo de novo")
    p.add_argument("--so", nargs="*", help="roda apenas os experimentos com estes nomes")
    args = p.parse_args()

    for prova, _ in PROVAS:
        if not Path(prova).exists():
            sys.exit(f"Falta {prova}. Rode antes: python 01_dados.py e python 01b_desafio.py")

    for exp in EXPERIMENTOS:
        nome = exp["nome"]
        if args.so and nome not in args.so:
            continue
        modelo = ["--modelo", exp["modelo"]] if "modelo" in exp else []
        adaptador = None

        if "treino" in exp:
            adaptador = Path("runs") / nome / "adaptador"
            if args.refazer or not (adaptador / "adapter_config.json").exists():
                extras = [x for k, v in exp["treino"].items() for x in (f"--{k}", str(v))]
                rodar([sys.executable, "03_treinar.py", "--nome", nome, *modelo, *extras])
        elif "adaptador_de" in exp:
            adaptador = Path("runs") / exp["adaptador_de"] / "adaptador"

        for prova, sufixo in PROVAS:
            destino = Path("resultados") / f"{nome}{sufixo}.json"
            if not args.refazer and "treino" not in exp and resultado_valido(destino, prova):
                print(f"\n--- {destino} ja existe, pulando")
                continue
            if not args.refazer and "treino" in exp and resultado_valido(destino, prova) \
                    and destino.stat().st_mtime >= (adaptador / "adapter_config.json").stat().st_mtime:
                print(f"\n--- {destino} ja existe, pulando")
                continue
            comando = [sys.executable, "02_avaliar.py", "--nome", f"{nome}{sufixo}", "--teste", prova, *modelo]
            if adaptador:
                comando += ["--adaptador", adaptador.as_posix()]
            if "temperature" in exp:
                comando += ["--temperature", str(exp["temperature"])]
            rodar(comando)

    print("\nBateria concluida. Gere a tabela e o grafico com:  python 05_comparar.py")


if __name__ == "__main__":
    main()
