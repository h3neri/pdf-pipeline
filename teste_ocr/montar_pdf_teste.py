#!/usr/bin/env python3
"""
montar_pdf_teste.py — monta o PDF de teste da transcrição imagem -> txt.

Independente do pdf_pipeline.py: mede cada página dos PDFs direto com o PyMuPDF, classifica a página por
tipo (texto corrido, tabela, slide, texto dentro de imagem...), tira as páginas repetidas entre documentos
e sorteia algumas de cada tipo. O resultado é um PDF só com as páginas sorteadas, na ordem do sorteio,
mais um manifesto CSV que diz de onde veio cada página e por que ela entrou.

Uso (na pasta pipeline/):
  python teste_ocr/montar_pdf_teste.py pdf_to_image/                 # mede, sorteia e monta
  python teste_ocr/montar_pdf_teste.py pdf_to_image/ --so-medir      # só grava a tabela de medidas

Saídas (em teste_ocr/):
  medidas.csv          uma linha por página dos PDFs originais (só números, nenhum texto dos documentos)
  amostra_teste.pdf    as páginas sorteadas, com um marcador por página (tipo e origem)
  manifesto.csv        pagina_teste -> documento, página original, tipo, medidas principais

O sorteio é reprodutível: mesma semente, mesmos PDFs -> mesmas páginas.
"""
import argparse
import csv
import hashlib
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf

AQUI = Path(__file__).resolve().parent

RE_NUMERO = re.compile(r"\d+(?:[.,]\d+)*")
RE_URL = re.compile(r"https?://|www\.", re.I)
RE_PREENCH = re.compile(r"_{4,}|[xX]{4,}|\*{4,}")
RE_PONTILHADO = re.compile(r"(?:\. ?){5,}")
RE_GRUDADA = re.compile(r"[A-Za-zÀ-ÿ]{22,}")                      # 22+ letras: quase sempre palavras grudadas
RE_LIGADURA = re.compile(r"\bf[il] [a-zà-ü]{3,}")                   # "fi nanceira"
RE_ESPACADA = re.compile(r"(?:\b[A-Za-zÀ-ÿ0-9] ){5,}[A-Za-zÀ-ÿ0-9]\b")  # "I m p o r t a n t e"
# campos de formulário: "( ) Sim", "___/___/____", "XXXX", "AAAAMMDD"
RE_CAMPO = re.compile(r"\(\s{0,3}\)\s*[A-Za-zÀ-ÿ]|_{2,}\s*/\s*_{2,}|\b[xX]{4,}\b|\bAAAAMMDD\b|\bHHMMSS\b")

# Tipos de página, em ordem de prioridade (uma página que se encaixa em vários fica com o primeiro),
# e quantas páginas sortear de cada um. Total: 48.
# Não há tipo "gráfico": nas 2.322 páginas, toda página com muitos vetores é tabela ou formulário.
# Gráfico, se houver, está como imagem e cai em "texto_em_imagem" ou "design_com_fundo".
COTAS = {
    "em_branco": 1,             # o modelo devolve vazio ou inventa?
    "texto_em_imagem": 6,       # prints de sistema, telas de app: só o modelo de imagem lê
    "capa_ou_divisor": 3,       # pouco texto, muito visual
    "formulario_ou_campos": 3,  # ( ) Sim ( ) Não, ___/___/____, XXXX
    "duas_colunas": 2,          # ordem de leitura
    "tabela_densa": 2,          # centenas de números numa página (tabelas do MCR)
    "tabela": 5,
    "sumario_pontilhado": 2,
    "defeito_do_nativo": 4,     # palavras grudadas, ligadura partida, letras espaçadas
    "links_urls": 3,            # endereços web no texto
    "slide": 5,                 # páginas 16:9
    "design_com_fundo": 4,      # texto sobre imagem decorativa
    "numeros_e_valores": 4,     # R$, %, datas, artigos
    "texto_corrido": 4,
}


# Rótulos corrigidos olhando as páginas sorteadas (5/nov/2026, semente 2026). A regra automática não vê
# a diferença nestes casos; a coluna tipo_auto do manifesto guarda o rótulo original.
TIPO_CONFERIDO = {
    ("documentos assets 1.pdf", 15): "tabela",       # a grade da tabela é uma imagem (cobertura 0,6)
    ("documentos assets 1.pdf", 309): "fluxograma",  # passos com setas e ícones, não duas colunas
}


def chave_texto(t: str) -> str:
    """Impressão digital do texto da página, para achar páginas repetidas entre documentos."""
    t = unicodedata.normalize("NFKC", t.lower())
    t = re.sub(r"\b\d{1,3}\b", " ", re.sub(r"\W+", " ", t))         # ignora número de página
    return hashlib.md5(" ".join(t.split()).encode()).hexdigest()


def duas_colunas(page, texto_blocos) -> bool:
    """Há blocos de texto lado a lado (um na metade esquerda, outro na direita) na mesma altura?"""
    w = page.rect.width
    esq = [b for b in texto_blocos if b[2] < 0.55 * w and len(b[4]) > 80]
    dir_ = [b for b in texto_blocos if b[0] > 0.45 * w and len(b[4]) > 80]
    sobrepoe = sum(1 for a in esq for b in dir_ if min(a[3], b[3]) - max(a[1], b[1]) > 20)
    return sobrepoe >= 2 and sum(len(b[4]) for b in esq) > 300 and sum(len(b[4]) for b in dir_) > 300


def medir_documento(caminho: str) -> list:
    linhas = []
    doc = pymupdf.open(caminho)
    for i, page in enumerate(doc):
        r = page.rect
        texto = page.get_text()
        blocos = [b for b in page.get_text("blocks") if b[6] == 0]
        area = abs(r) or 1.0
        cob = 0.0
        for info in page.get_image_info():
            q = pymupdf.Rect(info["bbox"]) & r
            if not q.is_empty:
                cob += abs(q)
        try:
            tabs = page.find_tables().tables
            celulas = max((t.row_count * t.col_count for t in tabs), default=0)
        except Exception:
            tabs, celulas = [], 0
        try:
            n_desenhos = len(page.get_cdrawings())
        except Exception:
            n_desenhos = 0
        linhas.append({
            "documento": Path(caminho).name, "pagina": i + 1,
            "largura": round(r.width), "altura": round(r.height),
            "n_chars": len("".join(texto.split())), "n_palavras": len(texto.split()),
            "cobertura_img": round(min(1.0, cob / area), 3), "n_desenhos": n_desenhos,
            "n_tabelas": len(tabs), "max_celulas": celulas,
            "n_links": sum(1 for l in page.get_links() if l.get("uri")) + len(RE_URL.findall(texto)),
            "n_numeros": len(RE_NUMERO.findall(texto)), "n_reais": texto.count("R$"), "n_pct": texto.count("%"),
            "n_preench": len(RE_PREENCH.findall(texto)), "n_campos": len(RE_CAMPO.findall(texto)),
            "n_pontilhado": len(RE_PONTILHADO.findall(texto)),
            "n_grudadas": len(RE_GRUDADA.findall(texto)), "n_ligadura": len(RE_LIGADURA.findall(texto)),
            "n_espacadas": len(RE_ESPACADA.findall(texto)),
            "duas_colunas": int(duas_colunas(page, blocos)),
            "chave": chave_texto(texto) if len(texto.split()) >= 30 else "",
        })
    return linhas


def classificar(m: dict, img_txt: dict) -> str:
    """Primeiro tipo que se encaixa, na ordem de COTAS."""
    pal, cob = m["n_palavras"], m["cobertura_img"]
    so_img = img_txt.get((m["documento"], m["pagina"]), 0)
    paisagem = m["largura"] > 1.3 * m["altura"]
    # as brancas do Bacen têm 16 vetores invisíveis; a contracapa do glossário (logotipo em vetor) tem 31
    if pal == 0 and cob < 0.02 and m["n_desenhos"] < 20:
        return "em_branco"
    if so_img >= 15 or (pal < 10 and cob > 0.5):
        return "texto_em_imagem"
    if pal < 40:
        return "capa_ou_divisor"
    if m["n_campos"] >= 3:
        return "formulario_ou_campos"
    if m["duas_colunas"]:
        return "duas_colunas"
    if m["n_tabelas"] >= 1 and (m["max_celulas"] >= 150 or m["n_numeros"] >= 150):
        return "tabela_densa"
    if m["n_tabelas"] >= 1 and m["max_celulas"] >= 9:
        return "tabela"
    if m["n_pontilhado"] >= 5:
        return "sumario_pontilhado"
    if m["n_grudadas"] >= 3 or m["n_ligadura"] >= 2 or m["n_espacadas"] >= 1:
        return "defeito_do_nativo"
    if m["n_links"] >= 1:
        return "links_urls"
    if paisagem:
        return "slide"
    if cob >= 0.3:
        return "design_com_fundo"
    if m["n_reais"] + m["n_pct"] >= 5 or m["n_numeros"] >= 60:
        return "numeros_e_valores"
    if pal >= 250:
        return "texto_corrido"
    return ""                      # página comum, curta: não entra no sorteio


def carregar_texto_em_imagem() -> dict:
    """Contagem de palavras que só existem dentro das imagens (medida com OCR em 5/nov/2026)."""
    arq = AQUI.parent / "experimento" / "texto_em_imagem.csv"
    if not arq.exists():
        print(f"AVISO: {arq} não existe; o tipo 'texto_em_imagem' usa só a cobertura de imagem")
        return {}
    nomes = {}  # doc_id do pipeline -> nome do arquivo
    for pdf in (AQUI.parent / "pdf_to_image").glob("*.pdf"):
        s = unicodedata.normalize("NFKD", pdf.stem)
        s = "".join(c for c in s if not unicodedata.combining(c)).lower()
        nomes[re.sub(r"[^a-z0-9]+", "_", s).strip("_")] = pdf.name
    out = {}
    with open(arq, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["doc_id"] in nomes:
                out[(nomes[r["doc_id"]], int(r["pagina"]))] = int(r["palavras_so_na_imagem"])
    return out


def sortear(medidas: list, img_txt: dict, semente: int, distancia: int = 6) -> list:
    rnd = random.Random(semente)
    vistos, unicas = {}, []
    # entre páginas repetidas, fica a do documento que não é só cópia (o regulamentos.pdf repete 87%)
    for m in sorted(medidas, key=lambda m: (m["documento"] == "regulamentos.pdf", m["documento"], m["pagina"])):
        if m["chave"] and m["chave"] in vistos:
            continue
        if m["chave"]:
            vistos[m["chave"]] = m
        unicas.append(m)
    por_tipo = defaultdict(list)
    for m in unicas:
        m["tipo"] = classificar(m, img_txt)
        if m["tipo"]:
            por_tipo[m["tipo"]].append(m)
    escolhidas = []

    def perto(m) -> bool:
        """Páginas vizinhas costumam ser quase iguais (os tutoriais "Passo 3, 4, 5..."): exige distância."""
        return any(e["documento"] == m["documento"] and abs(e["pagina"] - m["pagina"]) < distancia
                   for e in escolhidas)

    # os tipos raros escolhem primeiro; senão a distância mínima pode deixá-los sem página
    for tipo in sorted(COTAS, key=lambda t: len(por_tipo.get(t, []))):
        cota = COTAS[tipo]
        grupo = por_tipo.get(tipo, [])
        por_doc = defaultdict(list)
        for m in grupo:
            por_doc[m["documento"]].append(m)
        for g in por_doc.values():
            rnd.shuffle(g)
        docs = sorted(por_doc)
        rnd.shuffle(docs)
        sel = []
        while len(sel) < cota and any(por_doc.values()):   # rodízio: espalha o tipo pelos documentos
            for d in docs:
                while por_doc[d] and len(sel) < cota:
                    m = por_doc[d].pop()
                    if not perto(m):
                        sel.append(m)
                        escolhidas.append(m)
                        break
        if len(sel) < cota:
            print(f"AVISO: tipo '{tipo}' só tem {len(sel)} página(s) disponíveis (cota {cota})")
    print("\nPáginas candidatas por tipo (já sem repetidas):")
    for tipo in COTAS:
        print(f"  {tipo:22s} {len(por_tipo.get(tipo, [])):5d}")
    print(f"  {'(fora do sorteio)':22s} {sum(1 for m in unicas if not m['tipo']):5d}")
    print(f"  repetidas descartadas: {len(medidas) - len(unicas)}")
    rnd.shuffle(escolhidas)        # ordem embaralhada: quem revisa não vê os tipos em bloco
    return escolhidas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entradas", nargs="+", help="PDFs e/ou pastas com PDFs")
    ap.add_argument("--saida", default=str(AQUI), help="pasta de saída (padrão: teste_ocr/)")
    ap.add_argument("--semente", type=int, default=2026)
    ap.add_argument("--processos", type=int, default=2)
    ap.add_argument("--distancia", type=int, default=6,
                    help="distância mínima entre páginas escolhidas do mesmo documento (padrão: %(default)s)")
    ap.add_argument("--so-medir", action="store_true", help="só grava medidas.csv")
    ap.add_argument("--remedir", action="store_true", help="mede de novo mesmo se medidas.csv existir")
    a = ap.parse_args()

    pdfs = []
    for e in a.entradas:
        p = Path(e)
        pdfs += sorted(p.glob("*.pdf")) if p.is_dir() else [p]
    if not pdfs:
        print("Nenhum PDF encontrado.")
        return 1
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    caminhos = {p.name: p for p in pdfs}

    arq_medidas = saida / "medidas.csv"
    medidas = None
    if arq_medidas.exists() and not a.remedir:   # medir leva ~2 min; reaproveita se os PDFs são os mesmos
        with open(arq_medidas, encoding="utf-8") as f:
            medidas = [{k: (v if k in ("documento", "chave") else float(v) if "." in v else int(v))
                        for k, v in r.items()} for r in csv.DictReader(f)]
        if {m["documento"] for m in medidas} != set(caminhos):
            medidas = None
        else:
            print(f"Usando as medidas já gravadas em {arq_medidas} (--remedir para medir de novo)")
    if medidas is None:
        print(f"Medindo {len(pdfs)} PDF(s)...")
        with ProcessPoolExecutor(a.processos) as ex:
            medidas = [m for lista in ex.map(medir_documento, [str(p) for p in pdfs]) for m in lista]
        with open(arq_medidas, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(medidas[0]))   # 'chave' é só um hash: não tem texto
            w.writeheader()
            w.writerows(medidas)
        print(f"{len(medidas)} páginas medidas -> {arq_medidas}")
    if a.so_medir:
        return 0

    escolhidas = sortear(medidas, carregar_texto_em_imagem(), a.semente, a.distancia)
    for m in escolhidas:           # correções da revisão visual: só o rótulo muda, o sorteio é o mesmo
        m["tipo_auto"] = m["tipo"]
        m["tipo"] = TIPO_CONFERIDO.get((m["documento"], m["pagina"]), m["tipo"])

    teste = pymupdf.open()
    abertos = {}
    toc = []
    for n, m in enumerate(escolhidas, 1):
        src = abertos.setdefault(m["documento"], pymupdf.open(caminhos[m["documento"]]))
        teste.insert_pdf(src, from_page=m["pagina"] - 1, to_page=m["pagina"] - 1, links=False, annots=False)
        toc.append([1, f"{n:02d} | {m['tipo']} | {Path(m['documento']).stem} p.{m['pagina']}", n])
    teste.set_toc(toc)             # marcadores: só navegação, não aparecem na imagem da página
    teste.set_metadata({"title": "Amostra de teste: imagem -> txt", "subject": f"semente {a.semente}",
                        "creator": "teste_ocr/montar_pdf_teste.py"})
    teste.save(saida / "amostra_teste.pdf", garbage=4, deflate=True)

    campos = ["pagina_teste", "tipo", "tipo_auto", "documento", "pagina_original", "largura", "altura",
              "n_palavras", "cobertura_img", "n_tabelas", "n_desenhos", "n_numeros", "n_links", "n_campos",
              "n_pontilhado"]
    with open(saida / "manifesto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(campos)
        for n, m in enumerate(escolhidas, 1):
            w.writerow([n] + [m["pagina"] if c == "pagina_original" else m[c] for c in campos[1:]])
    print(f"\n{len(escolhidas)} páginas -> {saida / 'amostra_teste.pdf'}")
    print("por tipo:", dict(Counter(m["tipo"] for m in escolhidas)))
    print("por documento:", dict(Counter(m["documento"] for m in escolhidas)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
