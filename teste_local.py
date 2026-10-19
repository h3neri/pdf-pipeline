#!/usr/bin/env python3
"""
teste_local.py — testa o pipeline SEM GPU e SEM os PDFs reais.

1) Gera um PDF sintético de 14 páginas com as armadilhas típicas:
       python teste_local.py gerar-pdf --saida teste.pdf
   - cabeçalho e rodapé repetidos ("Página N de 14")
   - sumário com pontilhado (.......)
   - duas colunas, tabela, hifenização no fim da linha
   - página de assinaturas com "xxxxx", "____" e CPF mascarado
   - página só-imagem (sem camada de texto, como um escaneado)
   - página com camada de texto corrompida (texto visível ≠ texto extraído)
   - página mista (texto + imagem que contém texto) e página em branco
   Também grava teste.gabarito.json com a transcrição "verdadeira" de cada página.

2) Sobe um servidor FALSO compatível com a API da OpenAI (como vLLM/llama.cpp):
       python teste_local.py servidor-mock --gabarito teste.gabarito.json --porta 8000
   Ele "transcreve" devolvendo o gabarito, mas injeta falhas para exercitar o
   pipeline: erros HTTP 500 aleatórios, laço de repetição (pág. 3, 1ª tentativa),
   cercas ```markdown (pág. 4), frase introdutória (pág. 8), omissão de linha
   (pág. 12), número trocado (pág. 13) e caracteres chineses (pág. 14, 1ª tentativa).

Dependências: pymupdf (só para gerar o PDF). O servidor usa apenas a biblioteca padrão.
"""
import argparse
import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOTAL = 14
CAB = "BANCO EXEMPLO S.A. - Condições Gerais do Produto Crédito Pessoal"
ROD2 = "Documento público - versão 2026"

P = [
    "O presente instrumento estabelece as condições gerais aplicáveis à contratação de crédito "
    "pessoal junto ao Banco Exemplo S.A., inscrito no CNPJ sob o nº 00.000.000/0001-00, doravante "
    "denominado BANCO. A contratação poderá ser realizada nas agências, nos canais digitais ou por "
    "meio de correspondentes no País, observada a legislação vigente.",
    "A taxa de juros efetiva será de 2,49% ao mês, e o Custo Efetivo Total (CET) será informado ao "
    "CLIENTE antes da contratação, em percentual anual, conforme a regulamentação aplicável.",
    "O valor mínimo de cada operação é de R$ 500,00 e o valor máximo, de R$ 150.000,00, sujeito à "
    "análise de crédito. O prazo de pagamento poderá variar de 6 a 72 parcelas mensais e sucessivas, "
    "vencendo-se a primeira em até 59 dias após a liberação dos recursos.",
    "Em caso de atraso, incidirão sobre o valor da parcela vencida juros remuneratórios, juros "
    "moratórios de 1% ao mês e multa de 2%, calculados pro rata die até a data do efetivo pagamento.",
    "O CLIENTE poderá liquidar antecipadamente o saldo devedor, total ou parcialmente, com redução "
    "proporcional dos juros, mediante solicitação em qualquer canal de atendimento.",
    "As informações pessoais do CLIENTE serão tratadas em conformidade com a Lei Geral de Proteção "
    "de Dados Pessoais (Lei nº 13.709/2018), exclusivamente para as finalidades descritas na "
    "Política de Privacidade.",
    "O seguro prestamista é opcional e, se contratado, garantirá a quitação do saldo devedor em caso "
    "de morte ou invalidez permanente total por acidente, nos limites da apólice.",
    "Reclamações podem ser registradas no SAC, pelo telefone 0800 000 0000, disponível 24 horas por "
    "dia, ou na Ouvidoria, após o primeiro atendimento.",
    "A tarifa de cadastro, quando aplicável, será cobrada uma única vez no início do relacionamento, "
    "no valor de R$ 85,00, conforme a tabela de tarifas vigente.",
    "Este documento poderá ser alterado a qualquer tempo, mediante comunicação prévia de 30 (trinta) "
    "dias, e a versão atualizada estará disponível no site do BANCO.",
]
SUMARIO = [
    ("Capítulo 1 - Disposições Gerais", 3),
    ("Capítulo 2 - Encargos e Tarifas", 5),
    ("Capítulo 3 - Pagamento Antecipado", 9),
    ("Capítulo 4 - Assinaturas", 11),
]
TABELA = [
    ["Produto", "Taxa (a.m.)", "Prazo máximo", "Tarifa"],
    ["Crédito Pessoal", "2,49%", "72 meses", "R$ 85,00"],
    ["Crédito Consignado", "1,72%", "96 meses", "R$ 0,00"],
    ["Antecipação do 13º", "1,99%", "12 meses", "R$ 0,00"],
]
HIFEN = [  # linhas manuais para forçar hifenização no fim da linha
    "A contratação do seguro não é condição para a contra-",
    "tação do crédito, sendo vedada qualquer forma de venda",
    "casada, nos termos da legislação de defesa do consu-",
    "midor e da regulamentação bancária em vigor.",
]
QUADRO = [
    "Quadro-resumo da operação",
    "Taxa de juros: 2,49% a.m.",
    "Prazo: até 72 meses",
    "Valor: até R$ 150.000,00",
]
ASSINATURA = [
    "E, por estarem justas e contratadas, as partes assinam o presente instrumento.",
    "São Paulo, ____ de ________________ de 20____.",
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "CONTRATANTE",
    "Nome: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "CPF: xxx.xxx.xxx-xx",
    "________________________________________",
    "BANCO EXEMPLO S.A.",
]


# --------------------------------------------------------------------------- PDF
def gerar_pdf(saida):
    import pymupdf

    W, H = pymupdf.paper_size("a4")
    M = 56
    corpo = pymupdf.Rect(M, 80, W - M, H - 80)
    doc = pymupdf.open()
    gab = {}

    def cab_rod(pg, n):
        pg.insert_text((M, 42), CAB, fontsize=8, fontname="helv")
        pg.insert_text((M, H - 40), f"Página {n} de {TOTAL}", fontsize=8, fontname="helv")
        pg.insert_text((W - M - 140, H - 40), ROD2, fontsize=8, fontname="helv")

    def caixa(pg, rect, paragrafos, fs=10.5):
        texto = "\n\n".join(paragrafos)
        sobra = pg.insert_textbox(rect, texto, fontsize=fs, fontname="helv",
                                  align=pymupdf.TEXT_ALIGN_JUSTIFY)
        assert sobra >= 0, "texto não coube na caixa"

    def png_de_texto(linhas_ou_par, w, h, fs=11, dpi=150, margem=40):
        tmp = pymupdf.open()
        tp = tmp.new_page(width=w, height=h)
        tp.insert_textbox(pymupdf.Rect(margem, margem, w - margem, h - margem),
                          "\n\n".join(linhas_ou_par), fontsize=fs, fontname="helv")
        return tp.get_pixmap(dpi=dpi).tobytes("png")

    def gab_pagina(n, corpo_linhas):
        return [CAB] + corpo_linhas + [f"Página {n} de {TOTAL}", ROD2]

    for n in range(1, TOTAL + 1):
        pg = doc.new_page(width=W, height=H)
        if n == 7:  # página em branco (sem cabeçalho/rodapé)
            gab[str(n)] = []
            continue
        if n == 6:  # página "escaneada": só imagem, sem camada de texto
            linhas = [CAB] + [P[8], P[9]] + [f"Página {n} de {TOTAL}", ROD2]
            pg.insert_image(pg.rect, stream=png_de_texto(linhas, W, H))
            gab[str(n)] = linhas
            continue
        if n == 10:  # camada de texto corrompida: imagem legível + texto invisível embaralhado
            linhas = [CAB] + [P[4], P[6]] + [f"Página {n} de {TOTAL}", ROD2]
            pg.insert_image(pg.rect, stream=png_de_texto(linhas, W, H))
            def cesar(s):
                return "".join(chr((ord(c) - 97 + 3) % 26 + 97) if "a" <= c <= "z" else c for c in s)
            pg.insert_textbox(corpo, cesar(P[4] + " " + P[6]), fontsize=10.5,
                              fontname="helv", render_mode=3)
            gab[str(n)] = linhas
            continue

        cab_rod(pg, n)
        if n == 1:
            pg.insert_text((M, 100), "CONDIÇÕES GERAIS - CRÉDITO PESSOAL", fontsize=14, fontname="hebo")
            caixa(pg, pymupdf.Rect(M, 120, W - M, H - 80), [P[0], P[1]])
            corpo_l = ["CONDIÇÕES GERAIS - CRÉDITO PESSOAL", P[0], P[1]]
        elif n == 2:
            y = 110
            pg.insert_text((M, y), "SUMÁRIO", fontsize=12, fontname="hebo")
            corpo_l = ["SUMÁRIO"]
            for titulo, pag in SUMARIO:
                y += 24
                pontos = "." * (95 - len(titulo))
                pg.insert_text((M, y), f"{titulo} {pontos} {pag}", fontsize=10, fontname="helv")
                corpo_l.append(f"{titulo} {pontos} {pag}")
        elif n == 4:  # duas colunas
            meio = W / 2
            caixa(pg, pymupdf.Rect(M, 80, meio - 10, H - 80), [P[4], P[5]], fs=10)
            caixa(pg, pymupdf.Rect(meio + 10, 80, W - M, H - 80), [P[6], P[7]], fs=10)
            corpo_l = [P[4], P[5], P[6], P[7]]
        elif n == 5:  # tabela
            xs = [M, M + 160, M + 270, M + 380]
            corpo_l = ["Tabela 1 - Condições por produto"]
            pg.insert_text((M, 110), corpo_l[0], fontsize=11, fontname="hebo")
            for i, linha in enumerate(TABELA):
                y = 140 + 22 * i
                for x, cel in zip(xs, linha):
                    pg.insert_text((x, y), cel, fontsize=10, fontname="hebo" if i == 0 else "helv")
                corpo_l.append(" | ".join(linha))
        elif n == 8:  # mista: texto + imagem com texto
            caixa(pg, pymupdf.Rect(M, 80, W - M, 260), [P[0]])
            png = png_de_texto(QUADRO, W - 2 * M, 380, fs=16)
            pg.insert_image(pymupdf.Rect(M, 280, W - M, 660), stream=png)
            corpo_l = [P[0]] + QUADRO
        elif n == 9:  # hifenização manual
            for i, linha in enumerate(HIFEN):
                pg.insert_text((M, 110 + 16 * i), linha, fontsize=11, fontname="helv")
            corpo_l = [" ".join(HIFEN).replace("contra- tação", "contratação")
                       .replace("consu- midor", "consumidor")]
        elif n == 11:  # assinaturas
            y = 110
            for linha in ASSINATURA:
                pg.insert_text((M, y), linha, fontsize=10.5, fontname="helv")
                y += 28
            corpo_l = list(ASSINATURA)
        else:
            pares = {3: [P[2], P[3]], 12: [P[1], P[4]], 13: [P[7], P[8], P[2]], 14: [P[9], P[2]]}[n]
            caixa(pg, corpo, pares)
            corpo_l = pares
        gab[str(n)] = gab_pagina(n, corpo_l)

    doc.save(saida, garbage=3, deflate=True)
    gpath = saida.rsplit(".", 1)[0] + ".gabarito.json"
    with open(gpath, "w", encoding="utf-8") as f:
        json.dump(gab, f, ensure_ascii=False, indent=1)
    print(f"PDF: {saida} ({TOTAL} páginas)\nGabarito: {gpath}")


# ------------------------------------------------------------- servidor falso
class Mock:
    gabarito = {}
    modelo = "mock-ocr"
    p_erro = 0.05
    lat = (0.05, 0.25)
    tentativas = {}
    trava = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silencioso
        pass

    def _json(self, code, obj):
        dados = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(dados)))
        self.end_headers()
        self.wfile.write(dados)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": Mock.modelo, "object": "model"}]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        tam = int(self.headers.get("Content-Length", 0))
        corpo = json.loads(self.rfile.read(tam) or b"{}")
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._json(404, {"error": "not found"})
        partes = corpo["messages"][-1]["content"]
        if not any(p.get("type") == "image_url" for p in partes):
            return self._json(400, {"error": "sem imagem"})
        chave = self.headers.get("X-Pagina", "?:1")
        pag = chave.rsplit(":", 1)[-1]
        if pag not in Mock.gabarito and pag.isdigit():  # PDF maior que o gabarito: repete em ciclo
            pag = str((int(pag) - 1) % len(Mock.gabarito) + 1)
        with Mock.trava:
            Mock.tentativas[chave] = Mock.tentativas.get(chave, 0) + 1
            tent = Mock.tentativas[chave]
        time.sleep(random.uniform(*Mock.lat))
        if random.random() < Mock.p_erro:
            return self._json(500, {"error": "falha simulada"})

        linhas = Mock.gabarito.get(pag)
        fim = "stop"
        if linhas is None:
            texto = ""
        elif not linhas:
            texto = "[PÁGINA SEM TEXTO]"
        else:
            linhas = list(linhas)
            if pag == "12":                      # omissão persistente
                del linhas[2]
            if pag == "13":                      # número alucinado persistente
                linhas = [l.replace("R$ 500,00", "R$ 5.000,00") for l in linhas]
            texto = "\n\n".join(linhas)
            if pag == "3" and tent == 1:         # laço de repetição -> truncado
                texto += "\n" + "\n".join(["juros moratórios de 1% ao mês"] * 300)
                fim = "length"
            if pag == "14" and tent == 1:        # troca de idioma
                texto = texto.replace("documento", "文件")
            if pag == "4":
                texto = "```markdown\n" + texto + "\n```"
            if pag == "8":
                texto = "Aqui está a transcrição da página:\n\n" + texto
        mx = corpo.get("max_tokens") or 4096
        return self._json(200, {
            "id": f"mock-{chave}-{tent}", "object": "chat.completion", "model": Mock.modelo,
            "choices": [{"index": 0, "finish_reason": fim,
                         "message": {"role": "assistant", "content": texto}}],
            "usage": {"prompt_tokens": 1500, "completion_tokens": min(len(texto) // 4, mx),
                      "total_tokens": 1500 + min(len(texto) // 4, mx)},
        })


def servidor(gabarito, porta, p_erro, modelo, latencia):
    with open(gabarito, encoding="utf-8") as f:
        Mock.gabarito = json.load(f)
    Mock.p_erro, Mock.modelo, Mock.lat = p_erro, modelo, tuple(latencia)
    srv = ThreadingHTTPServer(("127.0.0.1", porta), Handler)
    srv.daemon_threads = True
    print(f"Servidor falso em http://127.0.0.1:{porta}/v1  (modelo '{modelo}', erro {p_erro:.0%})",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("gerar-pdf")
    a.add_argument("--saida", default="teste.pdf")
    b = sub.add_parser("servidor-mock")
    b.add_argument("--gabarito", default="teste.gabarito.json")
    b.add_argument("--porta", type=int, default=8000)
    b.add_argument("--p-erro", type=float, default=0.05, help="probabilidade de HTTP 500 por requisição")
    b.add_argument("--modelo", default="mock-ocr")
    b.add_argument("--latencia", type=float, nargs=2, default=[0.05, 0.25], metavar=("MIN", "MAX"),
                   help="segundos de 'inferência' simulada por requisição")
    args = ap.parse_args()
    if args.cmd == "gerar-pdf":
        gerar_pdf(args.saida)
    else:
        servidor(args.gabarito, args.porta, args.p_erro, args.modelo, args.latencia)


if __name__ == "__main__":
    sys.exit(main())
