"""
06_app.py - Frontend do detector de golpes (item 6 do enunciado).

O usuario escreve a mensagem (o prompt) e ve, lado a lado, a resposta do
modelo SEM treino e do modelo TREINADO, com o tempo de resposta na GPU.

Uso:  python 06_app.py            (abre em http://127.0.0.1:7860)
      python 06_app.py --padrao lora_r64
"""

import argparse
import json
import time
from pathlib import Path

import gradio as gr
import torch

from comum import DEVICE, MODELO_PADRAO, carregar, classificar, resumo_ambiente

EXEMPLOS = [
    "Seu pacote está retido na alfândega. Pague a taxa de R$ 18,90 em http://entrega-liberar.top para receber.",
    "oi pai, troquei de número, salva esse. consegue me mandar 900 no pix agora? te devolvo amanhã",
    "Seu código de verificação é 482913. Não compartilhe este código com ninguém.",
    "Lembrete: sua consulta com a Dra. Helena é amanhã às 14h. Responda SIM para confirmar.",
    "Olá, sou do suporte do banco. Me informe o código que chegou por SMS para cancelar a compra indevida.",
    "Galera, a pizza ficou R$ 32 pra cada um. Me passem o Pix quando puderem.",
]

_cache: dict = {}


def adaptadores() -> dict[str, dict]:
    """Experimentos treinados disponiveis em runs/ -> {nome: {modelo, adaptador}}."""
    achados = {}
    for cfg in sorted(Path("runs").glob("*/adaptador/adapter_config.json")):
        pasta = cfg.parent.parent
        modelo = MODELO_PADRAO
        info = pasta / "treino.json"
        if info.exists():
            modelo = json.loads(info.read_text(encoding="utf-8"))["parametros"]["modelo"]
        achados[pasta.name] = dict(modelo=modelo, adaptador=cfg.parent.as_posix())
    return achados


def obter(modelo: str, adaptador: str | None):
    chave = (modelo, adaptador)
    if chave not in _cache:
        _cache[chave] = carregar(modelo, adaptador)
    return _cache[chave]


def responder(tok, llm, mensagem: str, temperature: float) -> str:
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    inicio = time.time()
    rotulo, bruto = classificar(tok, llm, [mensagem], temperature=temperature)[0]
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - inicio) * 1000
    veredito = {"golpe": "## 🚨 GOLPE", "seguro": "## ✅ SEGURO"}.get(rotulo, "## ❓ Resposta fora do formato")
    return f"{veredito}\n\nResposta do modelo: `{bruto or '(vazia)'}`\n\nTempo: {ms:.0f} ms em {DEVICE.upper()}"


def analisar(mensagem: str, experimento: str, temperature: float):
    mensagem = (mensagem or "").strip()
    if not mensagem:
        aviso = "Digite ou cole uma mensagem para analisar."
        return aviso, aviso
    exp = adaptadores()[experimento]
    antes = responder(*obter(exp["modelo"], None), mensagem, temperature)
    depois = responder(*obter(exp["modelo"], exp["adaptador"]), mensagem, temperature)
    return antes, depois


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--padrao", default=None, help="experimento selecionado ao abrir")
    p.add_argument("--porta", type=int, default=7860)
    args = p.parse_args()

    disponiveis = adaptadores()
    if not disponiveis:
        raise SystemExit("Nenhum modelo treinado em runs/. Rode antes: python 03_treinar.py --nome lora_padrao")
    inicial = args.padrao if args.padrao in disponiveis else (
        "lora_r64" if "lora_r64" in disponiveis else next(iter(disponiveis)))

    amb = resumo_ambiente()
    gpu = f"{amb.get('gpu', 'CPU')} · {amb.get('vram_total_gb', '-')} GB · PyTorch {amb['pytorch']} · {amb['dtype']}"
    obter(disponiveis[inicial]["modelo"], None)                       # carrega antes de abrir a pagina
    obter(disponiveis[inicial]["modelo"], disponiveis[inicial]["adaptador"])

    with gr.Blocks(title="Detector de golpes") as app:
        gr.Markdown("# Detector de golpes em mensagens\n"
                    "SmolLM2 rodando localmente, antes e depois do fine-tuning com LoRA.\n\n"
                    f"**GPU:** {gpu}")
        with gr.Row():
            with gr.Column(scale=3):
                mensagem = gr.Textbox(label="Mensagem recebida (SMS, WhatsApp, e-mail)", lines=4,
                                      placeholder="Cole aqui a mensagem que você quer verificar")
                botao = gr.Button("Analisar", variant="primary")
            with gr.Column(scale=1):
                experimento = gr.Dropdown(list(disponiveis), value=inicial, label="Modelo treinado")
                temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.1, label="Temperature",
                                        info="0 = sempre a mesma resposta")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Antes: modelo sem treino")
                antes = gr.Markdown("Aguardando mensagem.")
            with gr.Column():
                gr.Markdown("### Depois: modelo treinado")
                depois = gr.Markdown("Aguardando mensagem.")
        gr.Examples(EXEMPLOS, inputs=mensagem, label="Exemplos (clique para preencher)")
        gr.Markdown("Projeto acadêmico. O modelo foi treinado com mensagens sintéticas e erra: "
                    "não use como única forma de verificar uma mensagem real.")
        botao.click(analisar, [mensagem, experimento, temperature], [antes, depois])
        mensagem.submit(analisar, [mensagem, experimento, temperature], [antes, depois])

    app.launch(server_port=args.porta, inbrowser=True)


if __name__ == "__main__":
    main()
