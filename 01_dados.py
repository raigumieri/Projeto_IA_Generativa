"""
01_dados.py - Gera o dataset do detector de golpes em mensagens.

Ideia (a mesma da aula 7): cada FAMILIA e um tipo de mensagem. Modelos de frase
combinados com listas de valores viram dezenas de mensagens diferentes com o
mesmo rotulo ("golpe" ou "seguro").

Saidas:
  dados/treino.jsonl -> usado no treino
  dados/teste.jsonl  -> NUNCA entra no treino. Dois tipos de item:
      "variacao": familia conhecida, combinacao de palavras nao vista
      "inedita" : familia que nao existe no treino (mede generalizacao)

Todas as empresas, links e chaves sao ficticios.

Uso:  python 01_dados.py
"""

import itertools
import json
import random
import string
from collections import Counter
from pathlib import Path

SEED = 42
N_TREINO = 30          # mensagens de cada familia que vao para o treino
N_TESTE = 5            # mensagens de cada familia que vao para o teste (variacao)
N_INEDITA = 10         # mensagens de cada familia inedita (so teste)
PASTA = Path("dados")

# Valores reaproveitados por varias familias
BANCOS = ["Banco Aurora", "NovaPay", "Banco Litoral", "Banco Horizonte"]
LOJAS = ["Loja Mundial", "MercadoBom", "Casa & Cia", "TecnoShop"]
TRANSPORTADORAS = ["Expresso Sul", "LogBrasil", "Rota Entregas", "Envia Já"]
LINKS = [
    "http://aurora-seguro.top/login",
    "bit.ly/regularizar-agora",
    "http://atendimento-cliente.xyz",
    "http://validar-acesso.click/app",
]
CHAVES_PIX = ["pagamentos2026@mail.top", "11 90000-0000", "quitacao.rapida@mail.xyz"]
NOMES = ["Ana", "Bruno", "Carla", "Diego"]
HORAS = ["9h", "14h", "16h30", "19h"]

# ---------------------------------------------------------------------------
# Familias que aparecem no TREINO (e no teste como "variacao")
# ---------------------------------------------------------------------------
FAMILIAS = {
    # ------------------------------ GOLPE ------------------------------
    "falso_banco": dict(
        rotulo="golpe",
        modelos=[
            "{banco}: sua conta foi bloqueada por {motivo}. Regularize agora em {link} para evitar o cancelamento.",
            "Alerta {banco}! Identificamos {motivo}. Confirme seus dados em {link} em até {prazo} ou sua conta será encerrada.",
            "{banco} informa: {motivo} em sua conta. Acesse {link} e atualize sua senha imediatamente.",
        ],
        campos=dict(
            banco=BANCOS,
            motivo=["atividade suspeita", "tentativa de acesso não autorizado",
                    "cadastro desatualizado", "movimentação irregular"],
            link=LINKS,
            prazo=["24 horas", "2 horas", "30 minutos"],
        ),
    ),
    "taxa_entrega": dict(
        rotulo="golpe",
        modelos=[
            "{transp}: seu pacote está retido por falta de pagamento da taxa de {valor}. Pague em {link} para liberar a entrega.",
            "Sua encomenda não pôde ser entregue. Pague a taxa alfandegária de {valor} em {link} ou o pacote será devolvido.",
            "{transp} avisa: tentativa de entrega sem sucesso. Reagende pagando {valor} em {link}.",
        ],
        campos=dict(transp=TRANSPORTADORAS, valor=["R$ 12,90", "R$ 27,50", "R$ 9,99", "R$ 34,00"], link=LINKS),
    ),
    "premio_falso": dict(
        rotulo="golpe",
        modelos=[
            "Parabéns! Você foi sorteado e ganhou {premio}. Para resgatar, informe seu CPF e os dados do cartão em {link}.",
            "Você é o ganhador de {premio} da promoção {loja}! Resgate em até {prazo} pagando apenas o frete em {link}.",
            "ÚLTIMO AVISO: seu prêmio de {premio} expira hoje. Clique em {link} e confirme seus dados bancários.",
        ],
        campos=dict(
            premio=["um celular novo", "R$ 5.000", "um vale-compras de R$ 1.000", "uma TV de 55 polegadas"],
            loja=LOJAS, link=LINKS, prazo=["24 horas", "1 hora", "hoje"],
        ),
    ),
    "falso_parente": dict(
        rotulo="golpe",
        modelos=[
            "Oi {parente}, troquei de número, salva esse aqui. Preciso que você faça um Pix de {valor} pra mim agora, te devolvo {quando}.",
            "{parente}, meu celular quebrou e estou usando esse número. Consegue pagar um boleto de {valor} pra mim? É urgente, depois te explico.",
            "Oi {parente}, sou eu. Estou com número novo. Me empresta {valor} por Pix? Não posso atender ligação agora.",
        ],
        campos=dict(
            parente=["mãe", "pai", "vó", "tia"],
            valor=["R$ 850", "R$ 1.200", "R$ 2.500", "R$ 600"],
            quando=["amanhã", "na sexta", "ainda hoje"],
        ),
    ),
    "vaga_falsa": dict(
        rotulo="golpe",
        modelos=[
            "Vaga home office: ganhe {valor} por dia apenas curtindo vídeos. Sem experiência. Chame no {contato} e pague a taxa de cadastro para começar.",
            "Você foi selecionado para trabalho de meio período com renda de {valor} por dia. Envie seus dados bancários para {contato}.",
            "Oportunidade única! Renda extra de {valor} diários trabalhando pelo celular. Vagas limitadas, cadastre-se em {link}.",
        ],
        campos=dict(
            valor=["R$ 300", "R$ 500", "R$ 800", "R$ 1.000"],
            contato=["11 90000-0000", "@rh.vagas.online", "21 90000-0000", "@renda.extra.ja"],
            link=LINKS,
        ),
    ),
    "pedido_de_codigo": dict(
        rotulo="golpe",
        modelos=[
            "Olá, sou do suporte da {empresa}. Enviamos um código de 6 dígitos para o seu celular por engano. Pode me informar o código para {motivo}?",
            "Central {empresa}: para {motivo}, responda esta mensagem com o código que você acabou de receber por SMS.",
            "Aqui é da equipe de segurança da {empresa}. Para {motivo}, preciso que me diga o código que chegou no seu celular.",
        ],
        campos=dict(
            empresa=BANCOS + LOJAS,
            motivo=["cancelar a compra indevida", "confirmar sua identidade",
                    "evitar o bloqueio da sua conta", "liberar seu cadastro"],
        ),
    ),
    "divida_falsa": dict(
        rotulo="golpe",
        modelos=[
            "Consta um débito de {valor} em seu CPF. Quite hoje com {desc} de desconto pelo Pix, chave {chave}. Após {prazo} seu nome será negativado.",
            "Aviso de protesto: boleto de {valor} em aberto. Evite o cartório pagando agora em {link}.",
            "Seu CPF será bloqueado em {prazo} por dívida de {valor}. Regularize com {desc} de desconto em {link}.",
        ],
        campos=dict(
            valor=["R$ 389,90", "R$ 1.240,00", "R$ 742,15", "R$ 96,30"],
            desc=["70%", "80%", "90%"], chave=CHAVES_PIX,
            prazo=["24 horas", "48 horas", "hoje"], link=LINKS,
        ),
    ),
    "investimento_milagroso": dict(
        rotulo="golpe",
        modelos=[
            "Transforme {valor} em {retorno} em apenas {prazo}! Método garantido, sem risco. Envie o Pix para {chave} e receba o lucro no mesmo dia.",
            "Grupo VIP de investimentos: lucro garantido de {pct} ao dia. Últimas vagas, deposite {valor} para entrar: {link}.",
            "Robô de apostas com {pct} de acerto. Deposite {valor} hoje e saque {retorno} em {prazo}. Garanta sua vaga em {link}.",
        ],
        campos=dict(
            valor=["R$ 100", "R$ 250", "R$ 500"], retorno=["R$ 2.000", "R$ 5.000", "R$ 10.000"],
            prazo=["24 horas", "3 dias", "uma semana"], pct=["15%", "30%", "98%"],
            chave=CHAVES_PIX, link=LINKS,
        ),
    ),

    # ------------------------------ SEGURO -----------------------------
    # Repare que varias mensagens seguras citam banco, Pix, valores e codigos.
    # Isso e proposital: o modelo nao pode aprender o atalho "falou de banco = golpe".
    "codigo_legitimo": dict(
        rotulo="seguro",
        modelos=[
            "Seu código de verificação {empresa} é {codigo}. Não compartilhe este código com ninguém.",
            "{empresa}: use o código {codigo} para confirmar seu acesso. Ele expira em {minutos} minutos.",
            "{codigo} é o seu código de segurança {empresa}. Nunca pedimos esse código por telefone ou mensagem.",
        ],
        campos=dict(empresa=BANCOS + LOJAS, codigo=["482913", "170356", "905521", "338047"], minutos=["5", "10", "15"]),
    ),
    "entrega_legitima": dict(
        rotulo="seguro",
        modelos=[
            "{transp}: seu pedido {pedido} saiu para entrega e deve chegar hoje até as {hora}.",
            "Seu pedido {pedido} foi entregue. Obrigado por comprar com a {loja}!",
            "{transp} informa: o pedido {pedido} está a caminho. Acompanhe pelo aplicativo oficial.",
        ],
        campos=dict(transp=TRANSPORTADORAS, pedido=["#48213", "#90377", "#15564", "#72009"], hora=HORAS, loja=LOJAS),
    ),
    "lembrete_consulta": dict(
        rotulo="seguro",
        modelos=[
            "Lembrete: sua consulta com {prof} está marcada para {dia} às {hora}. Responda SIM para confirmar.",
            "Olá! Confirmamos seu horário com {prof} para {dia}, às {hora}. Em caso de imprevisto, avise com antecedência.",
        ],
        campos=dict(
            prof=["a Dra. Helena", "o Dr. Marcos", "a dentista", "o fisioterapeuta"],
            dia=["amanhã", "segunda-feira", "quinta-feira", "o dia 15"], hora=HORAS,
        ),
    ),
    "conversa_pessoal": dict(
        rotulo="seguro",
        modelos=[
            "Oi {nome}, tudo bem? Vamos {atividade} {quando}?",
            "{nome}, não esquece de {tarefa} {quando}.",
        ],
        campos=dict(
            nome=NOMES,
            atividade=["almoçar", "ao cinema", "jogar bola", "tomar um café"],
            tarefa=["comprar pão", "levar o guarda-chuva", "ligar para a vó", "buscar a encomenda na portaria"],
            quando=["amanhã", "no sábado", "hoje à noite", "no domingo"],
        ),
    ),
    "pix_entre_conhecidos": dict(
        rotulo="seguro",
        modelos=[
            "{nome}, te mandei o Pix de {valor} {motivo}. Confere aí quando puder.",
            "Recebi seu Pix de {valor} {motivo}, obrigado! Já está tudo certo.",
            "{nome}, a sua parte {motivo} ficou {valor}. Sem pressa, me paga quando a gente se encontrar.",
        ],
        campos=dict(
            nome=NOMES, valor=["R$ 25", "R$ 40", "R$ 62,50", "R$ 120"],
            motivo=["da pizza", "do churrasco", "do presente", "da vaquinha"],
        ),
    ),
    "aviso_de_banco_legitimo": dict(
        rotulo="seguro",
        modelos=[
            "{banco}: compra aprovada de {valor} em {loja}. Se não reconhece, fale conosco pelo aplicativo oficial.",
            "{banco}: sua fatura de {mes} fechou em {valor}. Consulte os detalhes no app.",
            "{banco} informa: Pix recebido de {valor}. O saldo já está disponível no aplicativo.",
        ],
        campos=dict(
            banco=BANCOS, valor=["R$ 59,90", "R$ 230,00", "R$ 18,75", "R$ 1.045,20"],
            loja=LOJAS, mes=["setembro", "outubro", "novembro"],
        ),
    ),
    "promocao_legitima": dict(
        rotulo="seguro",
        modelos=[
            "{loja}: {desc} de desconto em {categoria} até {dia}. Aproveite nas lojas ou no nosso app.",
            "Semana do cliente {loja}! Frete grátis em {categoria} nas compras acima de {valor}.",
        ],
        campos=dict(
            loja=LOJAS, desc=["10%", "20%", "30%"],
            categoria=["eletrônicos", "roupas", "itens de casa", "livros"],
            dia=["domingo", "sexta-feira", "amanhã", "o fim do mês"], valor=["R$ 99", "R$ 150", "R$ 200"],
        ),
    ),
    "trabalho_e_escola": dict(
        rotulo="seguro",
        modelos=[
            "Bom dia, equipe. A reunião sobre {assunto} foi marcada para {dia} às {hora}.",
            "Pessoal, o professor avisou que a entrega de {assunto} ficou para {dia}.",
        ],
        campos=dict(
            assunto=["o projeto final", "o relatório mensal", "o trabalho em grupo", "a apresentação"],
            dia=["amanhã", "segunda-feira", "quinta-feira", "o dia 12"], hora=HORAS,
        ),
    ),
    "conta_de_servico": dict(
        rotulo="seguro",
        modelos=[
            "{servico}: sua conta de {mes} no valor de {valor} vence {dia}. Pague pelo app oficial ou por débito automático.",
            "{servico} informa: pagamento da fatura de {mes} confirmado. Obrigado!",
        ],
        campos=dict(
            servico=["Energia Sul", "Águas do Vale", "NetFibra", "TeleMais"],
            mes=["setembro", "outubro", "novembro"], valor=["R$ 89,90", "R$ 142,37", "R$ 210,05", "R$ 64,10"],
            dia=["amanhã", "dia 10", "dia 20", "na sexta-feira"],
        ),
    ),
}

# ---------------------------------------------------------------------------
# Familias que NUNCA aparecem no treino (so no teste, como "inedita")
# ---------------------------------------------------------------------------
INEDITAS = {
    "multa_de_transito_falsa": dict(
        rotulo="golpe",
        modelos=[
            "Departamento de Trânsito: consta multa de {valor} em seu veículo. Pague com desconto até hoje em {link} para evitar a suspensão da CNH.",
            "Notificação de trânsito: infração registrada em seu CPF. Consulte e pague em {link} em até {prazo}.",
        ],
        campos=dict(valor=["R$ 195,23", "R$ 293,47", "R$ 130,16"], link=LINKS, prazo=["24 horas", "48 horas", "hoje"]),
    ),
    "recadastramento_falso": dict(
        rotulo="golpe",
        modelos=[
            "Seu benefício será suspenso. Faça o recadastramento em {link} informando CPF e senha em até {prazo}.",
            "Atenção: cadastro do auxílio pendente. Atualize seus dados e a senha do aplicativo em {link} para não perder o pagamento.",
        ],
        campos=dict(link=LINKS, prazo=["24 horas", "48 horas", "hoje"]),
    ),
    "aviso_de_condominio": dict(
        rotulo="seguro",
        modelos=[
            "Condomínio informa: manutenção {item} {dia}, das {inicio} às {fim}. Contamos com a compreensão de todos.",
            "Aviso aos moradores: a assembleia será {dia} às {noite}, no salão de festas.",
        ],
        campos=dict(
            item=["dos elevadores", "da caixa d'água", "do portão da garagem"],
            dia=["amanhã", "na terça-feira", "no sábado"], inicio=["8h", "9h", "10h"], fim=["12h", "17h"],
            noite=["19h", "19h30", "20h"],
        ),
    ),
    "reserva_confirmada": dict(
        rotulo="seguro",
        modelos=[
            "Reserva confirmada no {lugar} para {n} pessoas, {dia} às {hora}. Até lá!",
            "{lugar}: sua reserva de {dia} às {hora} está confirmada. Tolerância de 15 minutos.",
        ],
        campos=dict(
            lugar=["Restaurante Sabor da Serra", "Cantina Bella Massa", "Bistrô do Porto"],
            n=["2", "4", "6"], dia=["amanhã", "sexta-feira", "sábado"], hora=["12h30", "20h", "21h"],
        ),
    ),
}


def expandir(familia: dict) -> list[str]:
    """Todas as combinacoes de cada modelo com os campos que ele realmente usa."""
    mensagens = []
    for modelo in familia["modelos"]:
        usados = [nome for _, nome, _, _ in string.Formatter().parse(modelo) if nome]
        listas = [familia["campos"][c] for c in usados]
        for combinacao in itertools.product(*listas):
            mensagens.append(modelo.format(**dict(zip(usados, combinacao))))
    return sorted(set(mensagens))


def gravar(caminho: Path, itens: list[dict]) -> None:
    with caminho.open("w", encoding="utf-8") as f:
        for item in itens:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    rng = random.Random(SEED)
    treino, teste = [], []

    print(f"{'familia':<28}{'rotulo':<9}{'geradas':>8}{'teste':>7}{'treino':>8}{'descarte':>10}")
    for nome, familia in FAMILIAS.items():
        mensagens = expandir(familia)
        rng.shuffle(mensagens)
        assert len(mensagens) >= N_TESTE + N_TREINO, f"familia {nome} gera poucas mensagens"
        para_teste = mensagens[:N_TESTE]
        para_treino = mensagens[N_TESTE:N_TESTE + N_TREINO]
        base = dict(rotulo=familia["rotulo"], familia=nome)
        teste += [dict(mensagem=m, tipo="variacao", **base) for m in para_teste]
        treino += [dict(mensagem=m, **base) for m in para_treino]
        descarte = len(mensagens) - N_TESTE - N_TREINO
        print(f"{nome:<28}{familia['rotulo']:<9}{len(mensagens):>8}{N_TESTE:>7}{N_TREINO:>8}{descarte:>10}")

    for nome, familia in INEDITAS.items():
        mensagens = expandir(familia)
        rng.shuffle(mensagens)
        assert len(mensagens) >= N_INEDITA, f"familia inedita {nome} gera poucas mensagens"
        teste += [dict(mensagem=m, tipo="inedita", rotulo=familia["rotulo"], familia=nome)
                  for m in mensagens[:N_INEDITA]]
        print(f"{nome:<28}{familia['rotulo']:<9}{len(mensagens):>8}{N_INEDITA:>7}{'-':>8}{'(inedita)':>10}")

    # Garantia da regra da disciplina: o teste nunca entra no treino
    vistos = {item["mensagem"] for item in treino}
    assert not any(item["mensagem"] in vistos for item in teste), "vazamento treino -> teste"

    rng.shuffle(treino)
    rng.shuffle(teste)
    PASTA.mkdir(exist_ok=True)
    gravar(PASTA / "treino.jsonl", treino)
    gravar(PASTA / "teste.jsonl", teste)

    print()
    print(f"dados/treino.jsonl: {len(treino)} mensagens", dict(Counter(i["rotulo"] for i in treino)))
    print(f"dados/teste.jsonl : {len(teste)} mensagens", dict(Counter((i["tipo"], i["rotulo"]) for i in teste)))
    print("\nExemplos do treino:")
    for item in treino[:4]:
        print(f"  [{item['rotulo']}] {item['mensagem']}")


if __name__ == "__main__":
    main()