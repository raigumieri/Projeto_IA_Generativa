"""
01b_desafio.py - Grava dados/desafio.jsonl: 40 mensagens escritas A MAO.

O teste.jsonl vem dos mesmos geradores do treino, entao e uma prova facil.
Este conjunto nao usa nenhum modelo de frase do 01_dados.py: tem girias,
abreviacoes, golpes sem link e mensagens seguras com "cara de golpe".
E a prova mais honesta de como o detector se sai fora do dataset.
Todas as empresas, links e contatos sao ficticios.

Uso:  python 01b_desafio.py
"""

import json
from pathlib import Path

GOLPES = [
    ("link", "Detectamos um login no seu app às 03:12 em outro aparelho. Não foi você? Cancele agora: http://seguranca-conta.top/cancelar"),
    ("link", "Seus pontos do cartão vencem HOJE. Troque 38.450 pontos por dinheiro antes da meia-noite: bit.ly/resgate-pontos"),
    ("link", "Olá, cliente. Sua assinatura de streaming não foi renovada. Atualize o cartão em http://assinatura-renovar.xyz para não perder o acesso."),
    ("link", "Comunicado: seu CPF foi usado em um cadastro suspeito. Consulte gratuitamente e bloqueie em http://cpf-protegido.click"),
    ("link", "Você tem um valor a receber de R$ 1.873,22 esquecido em bancos. Saque liberado só hoje pelo link bit.ly/valor-esquecido"),
    ("link", "Sua conta do app de mensagens será desativada em 12h por denúncias. Verifique sua identidade: http://verificar-conta.top"),
    ("conversa", "oi mãe meu cel caiu na privada kkk to com esse numero provisorio. consegue me mandar 700 no pix hj? amanha eu explico tudo"),
    ("conversa", "Bom dia! Aqui é o Rafael, da central de fraudes do seu banco. Identificamos uma compra de R$ 3.200. Para bloquear preciso que confirme a senha do cartão."),
    ("conversa", "Oi, tudo bem? Vi seu anúncio do sofá. Já fiz o pagamento, mas o site pediu que você pague a taxa de liberação de R$ 80 pra receber. Te mando o link."),
    ("conversa", "Amor, não consigo falar agora. Preciso que você pague esse boleto pra mim urgente, vence em 1 hora. Depois te explico, confia."),
    ("conversa", "Olá, falo do RH da empresa. Seu currículo foi aprovado! Para garantir a vaga, é necessário pagar o exame admissional de R$ 95 via Pix ainda hoje."),
    ("conversa", "Sou advogado e tenho uma herança de R$ 2,4 milhões em seu nome. Para iniciar a liberação, envie cópia do RG e os dados da sua conta."),
    ("conversa", "Oi, mandei um código de 6 números pro seu celular sem querer, era pra minha conta. Me passa ele por favor? É rapidinho."),
    ("conversa", "Moça, aqui é da operadora. Seu chip vai ser cancelado hoje. Para manter o número, me informe o código que acabou de chegar por SMS."),
    ("oferta", "iPhone novo por R$ 899, só hoje, estoque de leilão da Receita. Pagamento apenas por Pix antecipado. Chama no privado."),
    ("oferta", "Empréstimo aprovado de R$ 15.000 mesmo negativado, sem consulta. Para liberar, deposite o seguro de R$ 250 na conta do gerente."),
    ("oferta", "Ganhe R$ 200 agora só por avaliar restaurantes no celular. Tarefa leva 5 minutos. Faça um depósito de ativação de R$ 30 e comece."),
    ("oferta", "Consegui dobrar meu dinheiro em 2 dias com esse grupo, é real. Entra com R$ 150 e sai com R$ 300, o admin garante. Te passo o Pix dele."),
    ("oferta", "Sua CNH pode sair sem prova e sem aula por R$ 600. Processo sigiloso, documento original. Pagamento adiantado."),
    ("oferta", "Promoção relâmpago: passagens aéreas por R$ 49 para qualquer destino. Garanta informando os dados do cartão e o código de segurança neste chat."),
]

SEGUROS = [
    ("alerta_legitimo", "Você fez login em um novo aparelho. Se não foi você, abra o aplicativo e troque sua senha. Não enviamos links por SMS."),
    ("alerta_legitimo", "Compra de R$ 3.200,00 recusada por segurança. Para liberar, acesse o app do banco. Nunca pedimos senha por mensagem."),
    ("alerta_legitimo", "Seu boleto do condomínio de outubro já está disponível no aplicativo da administradora. Vencimento dia 10."),
    ("alerta_legitimo", "Aviso: sua fatura vence amanhã. Se já pagou, desconsidere esta mensagem."),
    ("alerta_legitimo", "Seu código para redefinir a senha é 771204. Se você não pediu, ignore esta mensagem e não informe o código a ninguém."),
    ("alerta_legitimo", "Débito automático da conta de luz realizado com sucesso. Valor: R$ 187,40."),
    ("conversa", "mãe, cheguei bem. o celular tava sem bateria, por isso não respondi. te ligo mais tarde"),
    ("conversa", "Oi! Te mandei 700 no Pix referente ao aluguel deste mês, tá? Confere e me avisa."),
    ("conversa", "Amor, paguei o boleto da internet agora de manhã. O comprovante tá no grupo da casa."),
    ("conversa", "Urgente: alguém viu minha chave? Preciso sair em 10 minutos e não acho em lugar nenhum."),
    ("conversa", "Oi, vi seu anúncio do sofá. Ainda está disponível? Posso passar aí no sábado para ver pessoalmente?"),
    ("conversa", "Parabéns pela aprovação!! Você merece demais. Vamos comemorar no fim de semana?"),
    ("conversa", "Galera, a vaquinha do presente da Carla fechou em R$ 30 por pessoa. Quem puder me passa até sexta."),
    ("conversa", "Bom dia, professor. Não vou conseguir ir à aula hoje, estou sem transporte. Posso entregar o trabalho amanhã?"),
    ("servico", "Seu pedido saiu para entrega. O entregador vai pedir os 4 últimos dígitos do código que aparece no app para confirmar o recebimento."),
    ("servico", "Olá! Seu currículo foi selecionado para a próxima etapa. A entrevista será online, na quinta às 15h. Não há nenhum custo no processo."),
    ("servico", "Sua consulta foi remarcada para terça, 8h30, a pedido do médico. Se não puder, ligue para a recepção."),
    ("servico", "Promoção de aniversário da loja: 15% de desconto em todo o site até domingo. Cupom: FESTA15."),
    ("servico", "Seu carro está pronto. O valor do serviço ficou R$ 480, como combinado. Pode retirar até as 18h."),
    ("servico", "Lembrete da biblioteca: o livro que você pegou emprestado deve ser devolvido até sexta-feira."),
]

itens = [dict(mensagem=m, rotulo="golpe", tipo="desafio", familia=f"golpe_{f}") for f, m in GOLPES]
itens += [dict(mensagem=m, rotulo="seguro", tipo="desafio", familia=f"seguro_{f}") for f, m in SEGUROS]

Path("dados").mkdir(exist_ok=True)
with open("dados/desafio.jsonl", "w", encoding="utf-8") as f:
    for item in itens:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"dados/desafio.jsonl: {len(itens)} mensagens ({len(GOLPES)} golpe, {len(SEGUROS)} seguro)")
