"""
05_comparar.py - Placar de todos os experimentos (antes x depois, parametro por parametro).

Le resultados/*.json e runs/*/treino.json e gera:
  resultados/tabela.md   -> tabela pronta para colar no README
  resultados/grafico.png -> acuracia de cada experimento nas duas provas

Uso:  python 05_comparar.py
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PASTA = Path("resultados")
PADRAO = dict(epocas=3, lr=2e-4, r=16)


def ler(caminho: Path) -> dict | None:
    if not caminho.exists():
        return None
    with caminho.open(encoding="utf-8") as f:
        return json.load(f)


def fmt(valor) -> str:
    return "-" if valor is None else (f"{valor:g}" if isinstance(valor, float) else str(valor))


def main() -> None:
    linhas = []
    for arquivo in sorted(PASTA.glob("*.json"), key=lambda a: a.stat().st_mtime):
        if arquivo.stem.endswith("__desafio"):
            continue
        teste = ler(arquivo)
        if not teste or "resumo" not in teste:
            continue
        nome = teste["nome"]
        desafio = ler(PASTA / f"{nome}__desafio.json")
        # pasta do treino: a do proprio experimento ou a do adaptador reutilizado
        pasta_run = Path(teste["adaptador"]).parent if teste.get("adaptador") else None
        treino = ler(pasta_run / "treino.json") if pasta_run else None
        par = treino["parametros"] if treino else {}
        r = teste["resumo"]
        linhas.append(dict(
            nome=nome,
            modelo=teste["modelo"].split("SmolLM2-")[-1].replace("-Instruct", ""),
            epocas=par.get("epocas"), lr=par.get("lr"), r=par.get("r"),
            amostras=(par.get("amostras") or "todas") if treino else None,
            temp=teste.get("temperature", 0.0),
            teste=r["geral"]["pct"],
            variacao=r["por_tipo"].get("variacao", {}).get("pct"),
            inedita=r["por_tipo"].get("inedita", {}).get("pct"),
            desafio=desafio["resumo"]["geral"]["pct"] if desafio else None,
            recall=desafio["resumo"]["golpe"]["recall"] if desafio else None,
            precisao=desafio["resumo"]["golpe"]["precisao"] if desafio else None,
            invalidas=r["invalidas"] + (desafio["resumo"]["invalidas"] if desafio else 0),
            tempo=treino["segundos"] if treino and pasta_run.name == nome else None,
            vram=treino["pico_vram_mb"] if treino and pasta_run.name == nome else None,
        ))
    if not linhas:
        sys.exit("Nenhum resultado em resultados/. Rode antes: python 04_experimentos.py")

    colunas = [("Experimento", "nome"), ("Modelo", "modelo"), ("Épocas", "epocas"), ("LR", "lr"), ("Rank", "r"),
               ("Exemplos", "amostras"), ("Temp.", "temp"), ("Teste %", "teste"), ("Variação %", "variacao"),
               ("Inéditas %", "inedita"), ("Desafio %", "desafio"), ("Recall golpe (desafio) %", "recall"),
               ("Precisão golpe (desafio) %", "precisao"), ("Fora do formato", "invalidas"),
               ("Treino (s)", "tempo"), ("VRAM treino (MB)", "vram")]
    tabela = ["| " + " | ".join(t for t, _ in colunas) + " |", "|" + "---|" * len(colunas)]
    tabela += ["| " + " | ".join(fmt(l[c]) for _, c in colunas) + " |" for l in linhas]
    texto = "\n".join(tabela) + "\n"
    (PASTA / "tabela.md").write_text(texto, encoding="utf-8")
    print(texto)

    print(f"{'experimento':<24}{'teste':>8}{'ineditas':>10}{'desafio':>9}")
    for l in linhas:
        print(f"{l['nome']:<24}{fmt(l['teste']):>8}{fmt(l['inedita']):>10}{fmt(l['desafio']):>9}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib nao instalado: grafico nao gerado.")

    nomes = [l["nome"] for l in linhas][::-1]
    y = range(len(nomes))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(nomes) + 1.6))
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ax.barh([i + 0.2 for i in y], [l["teste"] for l in linhas][::-1], height=0.36,
            color="#2a78d6", label="Teste (gerado, 125 mensagens)")
    ax.barh([i - 0.2 for i in y], [l["desafio"] or 0 for l in linhas][::-1], height=0.36,
            color="#eb6834", label="Desafio (escrito à mão, 40 mensagens)")
    ax.axvline(50, color="#52514e", linewidth=1, linestyle=":")
    ax.text(50, len(nomes) - 0.35, " chute (50%)", color="#52514e", fontsize=8, va="bottom")
    ax.set_yticks(list(y)); ax.set_yticklabels(nomes)
    ax.set_xlim(0, 100); ax.set_xlabel("Acurácia (%)")
    ax.set_title("Detector de golpes: acurácia por experimento", loc="left", fontsize=12)
    ax.grid(axis="x", color="#52514e", alpha=0.15); ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.3 / len(nomes)), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(PASTA / "grafico.png", dpi=150, facecolor=fig.get_facecolor())
    print(f"\nSalvos: {PASTA / 'tabela.md'} e {PASTA / 'grafico.png'}")


if __name__ == "__main__":
    main()
