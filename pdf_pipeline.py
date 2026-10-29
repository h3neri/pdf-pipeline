#!/usr/bin/env python3
"""
pdf_pipeline.py — PDFs extensos -> texto limpo, com triagem por página, VLM local
assíncrono com checkpoint, validação automática e exportação (.txt + .jsonl).

Subcomandos (rode cada um com -h para ver as opções):
  triagem   Lê cada página com PyMuPDF, extrai a camada de texto nativa, mede a
            qualidade e classifica a página: ok | misto | vlm | vazia.
  amostra   Sorteia páginas estratificadas e gera modelos de gabarito (PNG + TXT)
            para a equipe corrigir à mão (o padrão-ouro da avaliação).
  vlm       Envia páginas a um servidor local compatível com a API da OpenAI
            (vLLM, llama.cpp, Ollama, SGLang): 1 imagem por requisição, N em
            paralelo, gravando cada página no SQLite assim que termina. Se o
            processo cair, rode o mesmo comando de novo: ele continua de onde parou.
  importar  Carrega saídas externas (gabarito humano, Docling, outro pipeline)
            como mais uma "fonte" de texto por página.
  validar   Compara uma fonte com uma referência (texto nativo ou gabarito):
            omissões, alucinações, números, CER/WER, laços de repetição, idioma.
  revisar   Gera um HTML lado a lado (imagem da página x diferença de texto).
  exportar  Escolhe a melhor fonte por página, limpa (xxxx, ____, cabeçalho e
            rodapé repetidos, hifenização), confere se nenhuma página faltou e
            grava um .txt por documento + corpus.jsonl + manifesto.json.
  status    Resumo do andamento (pode rodar em paralelo com o `vlm`).

Fluxo típico:
  python pdf_pipeline.py triagem pdfs/
  python pdf_pipeline.py vlm --url http://localhost:8000/v1
  python pdf_pipeline.py validar --hip vlm:<modelo>
  python pdf_pipeline.py exportar --saida corpus/

Dependências: veja requirements.txt (instale com ./setup.sh; opcional: nvidia-ml-py)
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import difflib
import hashlib
import html
import json
import multiprocessing as mp
import os
import random
import re
import signal
import sqlite3
import statistics
import sys
import time
import unicodedata
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import pymupdf  # PyMuPDF >= 1.24
except ImportError:  # versões antigas só expõem o nome "fitz"
    import fitz as pymupdf  # type: ignore

VERSAO = "1.0"
SENTINELA_VAZIA = "[PÁGINA SEM TEXTO]"
FLAGS_TEXTO = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP  # ligaduras expandidas

PROMPT_PADRAO = """Transcreva fielmente TODO o texto visível nesta página de documento.

Regras:
- Siga a ordem natural de leitura; em páginas com colunas, termine uma coluna antes de passar à próxima.
- Não traduza, não resuma, não corrija e não complete nada: copie exatamente o que está escrito, inclusive números, valores, datas, siglas e pontuação.
- Cada parágrafo em uma linha; separe parágrafos com uma linha em branco.
- Tabelas: uma linha por linha da tabela, com as células separadas por " | ".
- Ignore elementos puramente gráficos (logotipos, ícones, linhas decorativas).
- Se a página não tiver nenhum texto, responda apenas: [PÁGINA SEM TEXTO]
Responda somente com a transcrição, sem comentários."""

PROMPT_ANCORA = """

Para conferência, abaixo está o texto bruto extraído da camada digital do PDF (pode estar fora de ordem, incompleto ou com erros). Use-o apenas para confirmar a grafia de números e nomes; a transcrição deve seguir a imagem.
<<<
{texto}
>>>"""

# ----------------------------------------------------------------- expressões
# Preenchimento de formulário: 4+ do mesmo caractere (com espaços simples entre eles).
RE_PREENCH = re.compile(r"([xX_\-\u2010\u2011\u2012–—=*])(?:[ \t]?\1){3,}")
# Pontilhado (sumários): some, mas não conta como "linha de formulário".
RE_PONTILHADO = re.compile(r"([.·…])(?:[ \t]?\1){3,}")
# Máscaras como xxx.xxx.xxx-xx, x.xxx,xx, XX/XX/XXXX (exige 4+ "x" no total).
RE_MASCARA = re.compile(r"(?<!\w)[xX]+(?:[.\-/,][xX]+)+(?!\w)")
RE_CPF = re.compile(r"(?<!\d)\d{3}\.\d{3}\.\d{3}-\d{2}(?!\d)")
RE_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
RE_PALAVRA = re.compile(r"[^\W\d_]+")
RE_TOKEN = re.compile(r"\w+")
RE_NUMERO = re.compile(r"\d+(?:[.,]\d+)*")
RE_LISTA = re.compile(r"^\s*(?:\(?\d{1,3}[.)]|\(?[a-zA-Z][.)]|[IVXLCivxlc]+[.)]|[-–•●▪◦·*])\s+")
RE_HIFEN = re.compile(r"(?<!\w)([^\W\d_]{2,})-\n?([^\W\d_]{2,})(?!\w)")
RE_PREAMBULO = re.compile(  # "Aqui está a transcrição da página:" (só se falar de transcrição/texto)
    r"^\s*(aqui est[áa]|seguem?|claro|certo|abaixo|here is|here's|sure|certainly|below is)\b[^\n]{0,100}"
    r"\b(transcri\w*|texto|conte[úu]do|p[áa]gina|markdown|text|content|page)\b[^\n]{0,60}:\s*$", re.I)
LIGADURAS = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
INVISIVEIS = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"), None)
STOP = set("""de a o que e do da em um para é com não uma os no se na por mais as dos como mas foi ao
ele das tem à seu sua ou ser quando muito há nos já está eu também só pelo pela até isso ela entre era
depois sem mesmo aos ter seus quem nas me esse eles estão você foram essa num nem suas meu às minha
numa pelos elas qual será nós lhe deles essas esses pelas este dele esta estes estas cada sobre
the of and to in is for on that by with as are be or from at this an which our we its was not have
has were their it""".split())

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS docs(
  doc_id TEXT PRIMARY KEY, caminho TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
  n_paginas INTEGER NOT NULL, triado_em TEXT, avisos TEXT);
CREATE TABLE IF NOT EXISTS paginas(
  doc_id TEXT NOT NULL, pagina INTEGER NOT NULL, categoria TEXT NOT NULL, motivo TEXT,
  n_chars INTEGER, frac_invalidos REAL, frac_stop REAL, frac_vogais REAL, cobertura_img REAL,
  n_lixo INTEGER, n_topo INTEGER, n_base INTEGER, texto_nativo TEXT, PRIMARY KEY(doc_id, pagina));
CREATE TABLE IF NOT EXISTS saidas(
  doc_id TEXT NOT NULL, pagina INTEGER NOT NULL, fonte TEXT NOT NULL, status TEXT NOT NULL,
  texto TEXT, tentativas INTEGER DEFAULT 0, fim TEXT, erro TEXT, problemas TEXT, meta TEXT,
  atualizado_em TEXT, PRIMARY KEY(doc_id, pagina, fonte));
CREATE TABLE IF NOT EXISTS validacao(
  doc_id TEXT NOT NULL, pagina INTEGER NOT NULL, ref TEXT NOT NULL, hip TEXT NOT NULL,
  categoria TEXT, metricas TEXT, sinais TEXT, PRIMARY KEY(doc_id, pagina, ref, hip));
"""


# --------------------------------------------------------------- utilidades
def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def agora() -> str:
    return datetime.now().isoformat(timespec="seconds")


def abrir_db(caminho: str) -> sqlite3.Connection:
    con = sqlite3.connect(caminho, timeout=60)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    con.execute("PRAGMA synchronous=NORMAL")  # com WAL: rápido e sem corromper o banco
    return con


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")[:60] or "doc"


def apelido_modelo(modelo: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", modelo.rsplit("/", 1)[-1].lower()).strip("-") or "vlm"


def normalizar_unicode(t: str) -> str:
    t = unicodedata.normalize("NFC", t).translate(INVISIVEIS)
    for a, b in LIGADURAS.items():
        t = t.replace(a, b)
    return t.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")


def tokens(t: str) -> list[str]:
    return [w.lower() for w in RE_TOKEN.findall(t)]


# ============================================================== 1. TRIAGEM
def agrupar_linhas(linhas):
    """Linhas lado a lado na mesma altura (células de tabela) viram uma só: 'a | b | c'."""
    grupos = []
    for bbox, t in linhas:
        cy, h = (bbox[1] + bbox[3]) / 2, bbox[3] - bbox[1]
        if grupos:
            g = grupos[-1]
            if abs(g["cy"] - cy) < 0.5 * max(1.0, min(g["h"], h)) and bbox[0] >= g["x1"] - 1:
                g["itens"].append((bbox[0], t))
                g["x1"] = bbox[2]
                continue
        grupos.append({"cy": cy, "h": h, "x1": bbox[2], "itens": [(bbox[0], t)]})
    return [" | ".join(t for _, t in sorted(g["itens"])) for g in grupos]


def refluir(linhas: list[str]) -> list[str]:
    """Junta as linhas visuais de um bloco em parágrafos (exceto itens de lista e linhas de tabela)."""
    paras, atual = [], ""
    for l in linhas:
        if not atual:
            atual = l
        elif " | " in l or " | " in atual or RE_LISTA.match(l):
            paras.append(atual)
            atual = l
        elif atual.endswith("-") and len(atual) > 1 and atual[-2].isalpha() and l[:1].islower():
            atual += l  # "contra-" + "tação" -> "contra-tação" (a exportação decide se tira o hífen)
        else:
            atual += " " + l
    if atual:
        paras.append(atual)
    return paras


def extrair_texto(page, faixa_borda: float):
    """Texto nativo da página: um parágrafo por linha; cabeçalho primeiro e rodapé por último.
    Devolve também quantas linhas vieram da faixa de cima/de baixo (candidatas a cabeçalho/rodapé)."""
    r = page.rect
    lim_topo, lim_base = r.y0 + faixa_borda * r.height, r.y1 - faixa_borda * r.height
    topo, corpo, base = [], [], []
    n_linhas = n_tab = 0
    for b in page.get_text("dict", flags=FLAGS_TEXTO)["blocks"]:
        if b.get("type", 0) != 0:
            continue
        linhas = []
        for ln in b["lines"]:
            t = " ".join("".join(s["text"] for s in ln["spans"]).split())
            if t:
                linhas.append((ln["bbox"], t))
        if not linhas:
            continue
        rows = agrupar_linhas(linhas)
        n_linhas += len(rows)
        n_tab += sum(" | " in x for x in rows)
        paras = refluir(rows)
        y0, y1 = b["bbox"][1], b["bbox"][3]
        (topo if y1 <= lim_topo else base if y0 >= lim_base else corpo).extend(paras)
    return "\n".join(topo + corpo + base), len(topo), len(base), n_tab, n_linhas


VOGAIS = set("aeiouyáéíóúâêôãõàü")


def metricas_texto(t: str):
    """n_chars, fração inválida, fração de palavras comuns, fração de letras, fração de vogais, nº de palavras."""
    chars = [c for c in t if not c.isspace()]
    n = len(chars)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0
    invalidos = sum(1 for c in chars if c == "\ufffd" or unicodedata.category(c) in ("Cc", "Co", "Cs"))
    letras = sum(1 for c in chars if c.isalpha())
    sem_lixo = RE_MASCARA.sub(" ", RE_PREENCH.sub(" ", t.lower()))  # "xxxx" não conta como consoante
    so_letras = [c for c in sem_lixo if c.isalpha()]
    vogais = sum(1 for c in so_letras if c in VOGAIS) / max(len(so_letras), 1)
    palavras = RE_PALAVRA.findall(sem_lixo)
    stop = sum(1 for w in palavras if w in STOP)
    return n, invalidos / n, stop / max(len(palavras), 1), letras / n, vogais, len(palavras)


def cobertura_imagens(page) -> float:
    area = abs(page.rect) or 1.0
    total = 0.0
    try:
        infos = page.get_image_info()
    except Exception:
        return 0.0
    for info in infos:
        r = pymupdf.Rect(info["bbox"]) & page.rect
        if not r.is_empty:
            total += abs(r)
    return min(1.0, total / area)


def contar_lixo(t: str) -> int:
    return (len(RE_PREENCH.findall(t)) + len(RE_PONTILHADO.findall(t))
            + sum(1 for m in RE_MASCARA.finditer(t) if sum(c in "xX" for c in m.group(0)) >= 4))


def analisar_pagina(doc, i: int, a) -> dict:
    reg = {"pagina": i + 1, "categoria": "vlm", "motivo": "", "n_chars": 0, "frac_invalidos": 0.0,
           "frac_stop": 0.0, "frac_vogais": 0.0, "cobertura_img": 0.0, "n_lixo": 0, "n_topo": 0,
           "n_base": 0, "texto_nativo": ""}
    try:
        page = doc.load_page(i)
        texto, n_topo, n_base, n_tab, n_linhas = extrair_texto(page, a.faixa_borda)
        cob = cobertura_imagens(page)
    except Exception as e:  # página corrompida: deixa para o VLM (que talvez consiga renderizar)
        reg["motivo"] = f"erro_extracao: {e}"[:200]
        return reg
    n, f_inv, f_stop, f_letras, f_vog, n_pal = metricas_texto(texto)
    n_desenhos = 0
    if n < a.min_chars:  # só conta vetores quando precisa (texto convertido em curvas)
        try:
            n_desenhos = len(page.get_cdrawings())
        except Exception:
            pass
    tem_visual = cob >= 0.02 or n_desenhos >= a.min_desenhos
    if n == 0:
        cat, motivo = ("vlm", "sem_camada_de_texto") if tem_visual else ("vazia", "")
    elif f_inv > a.max_invalidos:
        cat, motivo = "vlm", "caracteres_invalidos"
    elif n_pal >= 30 and f_letras > 0.5 and (f_stop < a.min_stop or (f_stop < 0.10 and f_vog < a.min_vogais)):
        cat, motivo = "vlm", "texto_ilegivel"  # fonte com codificação quebrada: letras "embaralhadas"
    elif n < a.min_chars and tem_visual:
        cat, motivo = "vlm", "pouco_texto"
    elif cob >= a.max_img:
        cat, motivo = "misto", "imagem_grande"
    elif a.tabelas_para_vlm and n_linhas >= 3 and n_tab / n_linhas >= 0.3:
        cat, motivo = "misto", "tabela"
    else:
        cat, motivo = "ok", ""
    reg.update(categoria=cat, motivo=motivo, n_chars=n, frac_invalidos=round(f_inv, 4),
               frac_stop=round(f_stop, 4), frac_vogais=round(f_vog, 4), cobertura_img=round(cob, 4),
               n_lixo=contar_lixo(texto), n_topo=n_topo, n_base=n_base, texto_nativo=texto)
    return reg


def listar_pdfs(entradas: list[str]) -> list[Path]:
    achados = []
    for e in entradas:
        p = Path(e)
        if p.is_dir():
            achados += [x for x in p.rglob("*") if x.is_file() and x.suffix.lower() == ".pdf"]
        elif p.is_file():
            achados.append(p)
        else:
            log(f"AVISO: não encontrado: {e}")
    return sorted({x.resolve() for x in achados})


def apagar_doc(con, doc_id: str, tabelas=("validacao", "saidas", "paginas", "docs")) -> None:
    for tab in tabelas:
        con.execute(f"DELETE FROM {tab} WHERE doc_id=?", (doc_id,))
    con.commit()


def triar_documento(con, caminho: Path, a) -> None:
    sha = sha256_arquivo(caminho)
    por_sha = con.execute("SELECT * FROM docs WHERE sha256=?", (sha,)).fetchone()
    por_caminho = con.execute("SELECT * FROM docs WHERE caminho=?", (str(caminho),)).fetchone()
    if por_caminho and por_caminho["sha256"] != sha:
        if not a.refazer:
            log(f"AVISO: {caminho.name} mudou desde a triagem (sha256 diferente). "
                f"Use --refazer para descartar o que foi feito com a versão antiga e triar de novo.")
            return
        log(f"{por_caminho['doc_id']}: arquivo mudou — apagando triagem e saídas da versão antiga")
        apagar_doc(con, por_caminho["doc_id"])
    if por_sha:
        doc_id = por_sha["doc_id"]
        if por_sha["caminho"] != str(caminho):
            con.execute("UPDATE docs SET caminho=? WHERE doc_id=?", (str(caminho), doc_id))
            con.commit()
        if a.refazer:  # mesmo arquivo: refaz só a triagem; as saídas do VLM continuam válidas
            apagar_doc(con, doc_id, ("validacao", "paginas"))
        elif por_sha["triado_em"]:
            log(f"{doc_id}: já triado ({por_sha['n_paginas']} páginas) — pulando")
            return
    else:
        doc_id = slug(caminho.stem)
        if con.execute("SELECT 1 FROM docs WHERE doc_id=?", (doc_id,)).fetchone():
            doc_id = f"{doc_id}_{sha[:6]}"

    doc = pymupdf.open(caminho)
    if doc.needs_pass and not doc.authenticate(""):
        log(f"ERRO: {caminho.name} é protegido por senha — pulando")
        return
    avisos = []
    if getattr(doc, "is_repaired", False):
        avisos.append("arquivo reparado ao abrir (estrutura interna danificada): confira a contagem de páginas")
    n = doc.page_count
    con.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,NULL,?)",
                (doc_id, str(caminho), sha, n, json.dumps(avisos, ensure_ascii=False)))
    feitas = {r[0] for r in con.execute("SELECT pagina FROM paginas WHERE doc_id=?", (doc_id,))}
    if feitas:
        log(f"{doc_id}: retomando triagem ({len(feitas)}/{n} páginas já analisadas)")
    t0, lote, cats = time.time(), [], Counter()
    cols = ["pagina", "categoria", "motivo", "n_chars", "frac_invalidos", "frac_stop", "frac_vogais",
            "cobertura_img", "n_lixo", "n_topo", "n_base", "texto_nativo"]

    def gravar():
        con.executemany(f"INSERT OR REPLACE INTO paginas(doc_id,{','.join(cols)}) "
                        f"VALUES (?{',?' * len(cols)})", [[doc_id] + [r[c] for c in cols] for r in lote])
        con.commit()
        lote.clear()

    for i in range(n):
        if i + 1 in feitas:
            continue
        reg = analisar_pagina(doc, i, a)
        cats[reg["categoria"]] += 1
        lote.append(reg)
        if len(lote) >= 200:  # checkpoint da triagem
            gravar()
            log(f"{doc_id}: {i + 1}/{n} páginas")
    gravar()
    con.execute("UPDATE docs SET triado_em=? WHERE doc_id=?", (agora(), doc_id))
    con.commit()
    dt = time.time() - t0
    tot = Counter(dict(con.execute("SELECT categoria, COUNT(*) FROM paginas WHERE doc_id=? "
                                   "GROUP BY categoria", (doc_id,)).fetchall()))
    log(f"{doc_id}: {n} páginas em {dt:.1f}s ({(n - len(feitas)) / max(dt, 1e-9):.0f} pág/s) -> "
        + ", ".join(f"{k}={tot[k]}" for k in ("ok", "misto", "vlm", "vazia")))
    for av in avisos:
        log(f"  AVISO: {av}")


def cmd_triagem(a) -> int:
    con = abrir_db(a.db)
    pdfs = listar_pdfs(a.entradas)
    if not pdfs:
        log("Nenhum PDF encontrado.")
        return 1
    log(f"{len(pdfs)} PDF(s) para triar (banco: {a.db})")
    for p in pdfs:
        try:
            triar_documento(con, p, a)
        except Exception as e:
            log(f"ERRO ao triar {p.name}: {e!r}")
    return 0


# =========================================================== limpeza e contexto
@dataclass
class Contexto:
    """Informação de documento inteiro usada na limpeza de cada página."""
    boiler: set = field(default_factory=set)
    freq: Counter = field(default_factory=Counter)
    vocab: set = field(default_factory=set)


def chave_linha(l: str) -> str:
    s = unicodedata.normalize("NFKD", l.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\d+", "#", s)
    return " ".join(re.sub(r"[^a-z#]+", " ", s).split())


def chaves_da_linha(l: str) -> set:
    ch = {chave_linha(l)} | {chave_linha(c) for c in l.split(" | ")}
    return {c for c in ch if c}


def indices_candidatos(linhas: list[str], bandas, k: int) -> set:
    """Linhas que podem ser cabeçalho/rodapé: as que vieram das faixas de cima/baixo da página
    (texto nativo, sabemos a posição) ou, sem geometria (texto do VLM), as k primeiras/últimas."""
    if bandas is not None:
        n_topo, n_base = bandas
        return set(range(min(n_topo, len(linhas)))) | set(range(max(len(linhas) - n_base, 0), len(linhas)))
    idx = [i for i, l in enumerate(linhas) if l.strip()]
    return set(idx[:k] + idx[-k:])


def contexto_doc(itens, n_paginas: int, k: int = 3) -> Contexto:
    """Aprende o que é cabeçalho/rodapé no documento: linhas candidatas (ver indices_candidatos)
    que se repetem em várias páginas, comparadas sem acentos/pontuação e com dígitos -> '#'.
    `itens`: lista de (texto, bandas) com bandas=(n_topo, n_base) ou None.
    Com geometria basta repetir em ~5% das páginas (mín. 3, máx. 10); sem ela, exige ~30%
    (mín. 3, máx. 20) para não apagar conteúdo que só por acaso se repete na borda."""
    cont_banda, cont_borda, vocab = Counter(), Counter(), set()
    for t, bandas in itens:
        if not t:
            continue
        vocab.update(w.lower() for w in RE_PALAVRA.findall(t))
        linhas = t.split("\n")
        chaves = set()
        for i in indices_candidatos(linhas, bandas, k):
            if 0 < len(linhas[i]) <= 300:
                chaves |= chaves_da_linha(linhas[i])
        (cont_banda if bandas is not None else cont_borda).update(chaves)
    lim_banda = max(3, min(0.05 * n_paginas, 10))
    lim_borda = max(3, min(0.30 * n_paginas, 20))
    boiler = ({c for c, f in cont_banda.items() if f >= lim_banda}
              | {c for c, f in cont_borda.items() if f >= lim_borda})
    return Contexto(boiler, Counter({c: max(cont_banda[c], cont_borda[c]) for c in boiler}), vocab)


def eh_boilerplate(l: str, boiler: set) -> bool:
    if not boiler:
        return False
    total = chave_linha(l)
    if total and total in boiler:
        return True
    celulas = [chave_linha(c) for c in l.split(" | ")]
    celulas = [c for c in celulas if c]
    return bool(celulas) and all(c in boiler for c in celulas)


def remover_enchimento(l: str):
    removido = 0

    def preench(m):
        nonlocal removido
        removido += len(m.group(0).replace(" ", "").replace("\t", ""))
        return " "

    def mascara(m):
        nonlocal removido
        s = m.group(0)
        if sum(c in "xX" for c in s) >= 4:
            removido += len(s)
            return " "
        return s

    l2 = RE_PREENCH.sub(preench, l)
    l2 = RE_MASCARA.sub(mascara, l2)
    l2 = RE_PONTILHADO.sub(" ", l2)
    return l2, removido


def deshifenizar(t: str, vocab: set) -> str:
    """'contra-tação' -> 'contratação' só se a forma sem hífen aparece em outro lugar do documento."""
    if not vocab:
        return t

    def f(m):
        junto = m.group(1) + m.group(2)
        if junto.lower() in vocab:
            return junto
        return m.group(0).replace("-\n", "-")
    return RE_HIFEN.sub(f, t)


def limpar_pagina(texto: str, ctx: Contexto | None, mascarar_cpf: bool = False, bandas=None,
                  k: int = 3) -> str:
    """Limpeza determinística de uma página (a MESMA função é usada na validação e na exportação)."""
    if not texto or texto.strip() == SENTINELA_VAZIA:
        return ""
    linhas = normalizar_unicode(texto).split("\n")
    bordas = indices_candidatos(linhas, bandas, k)
    saida = []
    for i, l in enumerate(linhas):
        if i in bordas and ctx and eh_boilerplate(l, ctx.boiler):
            continue
        l2, removido = remover_enchimento(l)
        if l2 != l:
            resto = l2.strip()
            if not RE_TOKEN.search(resto):
                continue  # sobrou só pontuação ("//", ":", ".")
            cheio = len(re.sub(r"\s", "", l))
            if removido and removido / max(cheio, 1) >= 0.5 and len(resto.split()) <= 6:
                continue  # linha de formulário: "Nome: xxxxxxxx", "CPF: xxx.xxx.xxx-xx"
            l = l2
        if mascarar_cpf:
            l = RE_CPF.sub("[CPF]", l)
        l = re.sub(r"[ \t]+", " ", l).strip()
        l = re.sub(r" ([.,;:!?])", r"\1", l)
        saida.append(l)
    t = "\n".join(saida)
    if ctx:
        t = deshifenizar(t, ctx.vocab)
    return re.sub(r"\n{2,}", "\n", t).strip()  # um parágrafo (ou linha de tabela) por linha


# ================================================================ 2. AMOSTRA
def render_png(caminho: str, pagina: int, lado_px: int, formato: str = "png") -> bytes:
    with pymupdf.open(caminho) as doc:
        return _pixmap(doc[pagina - 1], lado_px).tobytes(formato)


def _pixmap(page, lado_px: int):
    zoom = lado_px / max(page.rect.width, page.rect.height)
    return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)


LEIA_ME_GABARITO = """COMO PREENCHER O GABARITO (padrão-ouro)

Cada página sorteada tem um .png (a imagem) e um .txt com o texto nativo extraído do PDF.
Edite o .txt até ele ficar EXATAMENTE como o texto final desejado para o corpus:

1. Conteúdo fiel à imagem: corrija palavras/números errados, complete o que faltar
   (inclusive texto que só existe dentro de imagens) e apague o que não está na página.
2. Remova o que a limpeza deve remover: cabeçalho e rodapé repetidos, número de página,
   linhas de preenchimento (xxxx, ____, ....), campos vazios de formulário.
3. Um parágrafo por linha (linha em branco entre parágrafos é opcional).
4. Tabelas: uma linha por linha da tabela, células separadas por " | ".
5. Página sem texto útil: deixe o arquivo vazio.

Combinem as regras ANTES de começar (o ideal: duas pessoas fazem 5 páginas iguais e comparam).
Depois: python pdf_pipeline.py importar --fonte gabarito --dir {pasta}
"""


def cmd_amostra(a) -> int:
    con = abrir_db(a.db)
    rows = con.execute("SELECT p.doc_id, p.pagina, p.categoria, p.n_lixo, p.texto_nativo, d.caminho "
                       "FROM paginas p JOIN docs d USING(doc_id) WHERE p.categoria != 'vazia'").fetchall()
    if not rows:
        log("Nada triado ainda.")
        return 1
    rnd = random.Random(a.seed)
    grupos = defaultdict(list)
    for r in rows:  # estratos: categoria x tem-lixo x documento
        grupos[(r["categoria"], (r["n_lixo"] or 0) > 0, r["doc_id"])].append(r)
    for g in grupos.values():
        rnd.shuffle(g)
    chaves = sorted(grupos)
    rnd.shuffle(chaves)
    escolhidas = []
    while len(escolhidas) < a.n and any(grupos.values()):
        for k in chaves:  # round-robin: cada estrato ganha 1 antes de algum ganhar 2
            if grupos[k] and len(escolhidas) < a.n:
                escolhidas.append(grupos[k].pop())
    escolhidas.sort(key=lambda r: (r["doc_id"], r["pagina"]))
    pasta = Path(a.saida)
    pasta.mkdir(parents=True, exist_ok=True)
    with open(pasta / "paginas.txt", "w", encoding="utf-8") as f:
        for r in escolhidas:
            f.write(f"{r['doc_id']}:{r['pagina']}\n")
    for r in escolhidas:
        base = pasta / f"{r['doc_id']}__p{r['pagina']:05d}"
        base.with_suffix(".png").write_bytes(render_png(r["caminho"], r["pagina"], a.lado_px))
        txt = base.with_suffix(".txt")
        if not txt.exists() or a.sobrescrever:  # nunca apaga um gabarito já corrigido
            txt.write_text(r["texto_nativo"] or "", encoding="utf-8")
    (pasta / "LEIA-ME.txt").write_text(LEIA_ME_GABARITO.format(pasta=pasta), encoding="utf-8")
    cats = Counter(r["categoria"] for r in escolhidas)
    log(f"{len(escolhidas)} páginas em {pasta}/ ({dict(cats)}). Lista: {pasta / 'paginas.txt'}")
    return 0


# ==================================================================== 3. VLM
_DOCS_ABERTOS: dict = {}


def _init_processo_render():
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C é tratado só pelo processo principal


def _render_png_cache(caminho: str, pagina: int, lado_px: int) -> bytes:
    """Roda nos processos auxiliares: mantém poucos PDFs abertos (PyMuPDF não é thread-safe)."""
    doc = _DOCS_ABERTOS.get(caminho)
    if doc is None:
        if len(_DOCS_ABERTOS) >= 4:
            _DOCS_ABERTOS.pop(next(iter(_DOCS_ABERTOS))).close()
        doc = _DOCS_ABERTOS[caminho] = pymupdf.open(caminho)
    return _pixmap(doc[pagina - 1], lado_px).tobytes("png")


def pos_processar(conteudo: str) -> str:
    t = (conteudo or "").strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()  # modelos com "raciocínio"
    m = re.match(r"^```[\w-]*[ \t]*\n(.*?)\n?```\s*$", t, flags=re.S)
    if m:
        t = m.group(1).strip()
    linhas = t.split("\n")
    if linhas and RE_PREAMBULO.match(linhas[0]):
        t = "\n".join(linhas[1:]).strip()
    return t


def tem_repeticao(t: str) -> bool:
    linhas = [l.strip() for l in t.splitlines() if l.strip()]
    seguidas = 1
    for x, y in zip(linhas, linhas[1:]):
        seguidas = seguidas + 1 if (x == y and len(x) > 3) else 1
        if seguidas >= 4:
            return True
    b = t.encode("utf-8")
    return len(b) > 800 and len(b) / len(zlib.compress(b)) > 4.0


def problemas_sem_referencia(texto: str, fim: str | None) -> list[str]:
    p = []
    if fim == "length":
        p.append("truncado")
    if not texto.strip():
        p.append("vazio")
    if RE_CJK.search(texto):
        p.append("cjk")
    if tem_repeticao(texto):
        p.append("repeticao")
    return p


class ServidorFora(Exception):
    pass


class Interrompido(Exception):
    pass


class ErroPagina(Exception):
    pass


class Renderizador:
    """Processos que desenham as páginas. Se um deles morrer (página que derruba o MuPDF),
    recria o conjunto e tenta de novo uma vez; na segunda falha a página vira 'erro'."""

    def __init__(self, n_procs: int):
        self.n, self.trava, self.geracao = n_procs, asyncio.Lock(), 0
        self.pool = self._novo()

    def _novo(self):
        return ProcessPoolExecutor(max_workers=self.n, mp_context=mp.get_context("spawn"),
                                   initializer=_init_processo_render)

    async def png(self, caminho: str, pagina: int, lado_px: int) -> bytes:
        from concurrent.futures.process import BrokenProcessPool
        loop = asyncio.get_running_loop()
        for tentativa in (1, 2):
            geracao = self.geracao
            try:
                return await loop.run_in_executor(self.pool, _render_png_cache, caminho, pagina, lado_px)
            except BrokenProcessPool:
                async with self.trava:
                    if self.geracao == geracao:  # só o primeiro que notar recria
                        log(f"Processo de renderização caiu (perto de {Path(caminho).name}:{pagina}); recriando...")
                        self.pool.shutdown(wait=False, cancel_futures=True)
                        self.pool, self.geracao = self._novo(), self.geracao + 1
                if tentativa == 2:
                    raise ErroPagina("renderização: o processo do MuPDF caiu nesta página")
        raise ErroPagina("renderização falhou")

    def fechar(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


@dataclass
class Estado:
    total: int
    inicio: float = field(default_factory=time.monotonic)
    ok: int = 0
    sinalizadas: int = 0
    erros: int = 0
    abortado: bool = False
    ultimo_aviso_servidor: float = 0.0

    @property
    def feitas(self):
        return self.ok + self.sinalizadas + self.erros

    def linha(self) -> str:
        dt = time.monotonic() - self.inicio
        taxa = self.feitas / dt * 60 if dt > 0 else 0
        resta = (self.total - self.feitas) / taxa if taxa > 0 else float("inf")
        eta = f"{resta:.0f} min" if resta != float("inf") else "?"
        return (f"{self.feitas}/{self.total} páginas | ok={self.ok} sinalizadas={self.sinalizadas} "
                f"erros={self.erros} | {taxa:.1f} pág/min | faltam ~{eta}")


class Termico:
    """Pausa os envios se a GPU passar da temperatura máxima (só funciona na máquina da GPU)."""

    def __init__(self, temp_max, indice):
        self.temp_max, self.h, self.trava = temp_max, None, asyncio.Lock()
        if temp_max is None:
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(indice)
            log(f"Controle térmico ativo: pausa acima de {temp_max}°C (GPU {indice})")
        except Exception as e:
            log(f"AVISO: controle térmico desativado (instale nvidia-ml-py e rode na máquina da GPU): {e}")

    def temperatura(self):
        return self.nv.nvmlDeviceGetTemperature(self.h, self.nv.NVML_TEMPERATURE_GPU)

    async def aguardar(self, parar: asyncio.Event):
        if self.h is None:
            return
        async with self.trava:
            t = self.temperatura()
            if t < self.temp_max:
                return
            log(f"GPU a {t}°C (limite {self.temp_max}°C): pausando novos envios...")
            while t > self.temp_max - 5 and not parar.is_set():
                await asyncio.sleep(15)
                t = self.temperatura()
            log(f"GPU a {t}°C: retomando.")


async def chamar_servidor(cli, url, payload, chave, a, estado: Estado, parar: asyncio.Event):
    """Uma requisição. Servidor fora do ar NÃO gasta tentativa da página: espera e tenta de novo."""
    import httpx
    t_queda, atraso = None, 2.0
    while True:
        try:
            r = await cli.post(url, json=payload, headers={"X-Pagina": chave})
        except httpx.ConnectTimeout as e:
            falha = repr(e)
        except httpx.TimeoutException as e:
            raise ErroPagina(f"timeout ({a.timeout}s): {e!r}")
        except (httpx.NetworkError, httpx.RemoteProtocolError) as e:
            falha = repr(e)
        else:
            if r.status_code in (502, 503, 504):
                falha = f"HTTP {r.status_code}"
            elif r.status_code >= 400:
                raise ErroPagina(f"HTTP {r.status_code}: {r.text[:300]}")
            else:
                try:
                    j = r.json()
                    ch = j["choices"][0]
                    return ch["message"].get("content") or "", ch.get("finish_reason"), j.get("usage") or {}
                except Exception as e:
                    raise ErroPagina(f"resposta inválida: {e!r} {r.text[:200]}")
        agora_m = time.monotonic()
        t_queda = t_queda or agora_m
        if agora_m - t_queda > a.espera_servidor:
            raise ServidorFora(falha)
        if parar.is_set():
            raise Interrompido()
        if agora_m - estado.ultimo_aviso_servidor > 30:
            estado.ultimo_aviso_servidor = agora_m
            log(f"Servidor indisponível ({falha}); aguardando até {a.espera_servidor:.0f}s...")
        await asyncio.sleep(atraso + random.random())
        atraso = min(atraso * 2, 30.0)


async def processar_pagina(cli, url, modelo, png: bytes, item, a, prompt, estado, parar):
    doc_id, pagina, texto_nativo = item["doc_id"], item["pagina"], item["texto_nativo"] or ""
    chave = f"{doc_id}:{pagina}"
    texto_prompt = prompt
    if a.ancora and texto_nativo.strip():
        texto_prompt += PROMPT_ANCORA.format(texto=texto_nativo[: a.ancora_max_chars])
    b64 = base64.b64encode(png).decode("ascii")
    melhor, ultimo_erro, max_tok, t0 = None, None, a.max_tokens, time.monotonic()
    tentativas = 0
    for tent in range(1, a.tentativas + 1):
        if parar.is_set() and tent > 1:
            return None  # parando: não grava um resultado "pela metade"; a página fica pendente
        tentativas = tent
        payload = {
            "model": modelo,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": texto_prompt},
            ]}],
            "temperature": 0.0 if tent == 1 else a.temp_retentativa,
            "max_tokens": max_tok,
        }
        payload.update(a.extra)
        try:
            conteudo, fim, uso = await chamar_servidor(cli, url, payload, chave, a, estado, parar)
        except ErroPagina as e:
            ultimo_erro = str(e)
            await asyncio.sleep(min(2 ** tent, 20) + random.random())
            continue
        texto = pos_processar(conteudo)
        probs = problemas_sem_referencia(texto, fim)
        cand = {"texto": texto, "fim": fim, "problemas": probs, "uso": uso, "tentativa": tent}
        if melhor is None or (len(probs), "truncado" in probs) < (len(melhor["problemas"]),
                                                                 "truncado" in melhor["problemas"]):
            melhor = cand
        if not probs:
            break
        if "truncado" in probs:
            max_tok = min(max_tok * 2, a.max_tokens_teto)
    meta = {"modelo": modelo, "lado_px": a.lado_px, "prompt_sha1": hashlib.sha1(texto_prompt.encode()).hexdigest()[:10],
            "ancora": bool(a.ancora), "segundos": round(time.monotonic() - t0, 2), "versao": VERSAO}
    if melhor is None:
        return {"status": "erro", "texto": None, "tentativas": tentativas, "fim": None,
                "erro": ultimo_erro, "problemas": [], "meta": meta}
    meta.update(uso=melhor["uso"], tentativa_escolhida=melhor["tentativa"])
    return {"status": "ok" if not melhor["problemas"] else "sinalizada", "texto": melhor["texto"],
            "tentativas": tentativas, "fim": melhor["fim"], "erro": ultimo_erro,
            "problemas": melhor["problemas"], "meta": meta}


def selecionar_pendentes(con, a, fonte: str) -> list[dict]:
    if a.paginas:
        rows, vistas = [], set()
        for linha in Path(a.paginas).read_text(encoding="utf-8").splitlines():
            if ":" not in linha.strip() or linha.strip() in vistas:
                continue
            vistas.add(linha.strip())
            d, p = linha.strip().rsplit(":", 1)
            if not p.strip().isdigit():
                log(f"AVISO: linha inválida em {a.paginas}: {linha!r}")
                continue
            r = con.execute("SELECT p.doc_id, p.pagina, p.categoria, p.texto_nativo, d.caminho FROM paginas p "
                            "JOIN docs d USING(doc_id) WHERE p.doc_id=? AND p.pagina=?", (d, int(p))).fetchone()
            if r is None:
                log(f"AVISO: {linha.strip()} não está na triagem — ignorada")
            else:
                rows.append(r)
    else:
        cats = {"precisa": ("vlm", "misto"), "vlm": ("vlm",), "todas": ("ok", "misto", "vlm")}[a.selecao]
        sql = ("SELECT p.doc_id, p.pagina, p.categoria, p.texto_nativo, d.caminho FROM paginas p "
               f"JOIN docs d USING(doc_id) WHERE p.categoria IN ({','.join('?' * len(cats))})")
        params = list(cats)
        if a.docs:
            ids = [x.strip() for x in a.docs.split(",") if x.strip()]
            sql += f" AND p.doc_id IN ({','.join('?' * len(ids))})"
            params += ids
        rows = con.execute(sql + " ORDER BY p.doc_id, p.pagina", params).fetchall()
    feitos = {(r[0], r[1]): r[2] for r in
              con.execute("SELECT doc_id, pagina, status FROM saidas WHERE fonte=?", (fonte,))}
    pend = []
    for r in rows:
        st = feitos.get((r["doc_id"], r["pagina"]))
        if not a.refazer_tudo:
            if st == "ok" or (st == "sinalizada" and not a.refazer_sinalizadas):
                continue
        pend.append(dict(r))
    return pend[: a.limite] if a.limite else pend


async def descobrir_modelo(cli, base: str, pedido: str | None, espera: float) -> str:
    import httpx
    t0 = time.monotonic()
    while True:
        try:
            r = await cli.get(f"{base}/models")
            if 400 <= r.status_code < 500 and r.status_code not in (404, 408, 429):
                raise SystemExit(f"O servidor respondeu HTTP {r.status_code} em {base}/models "
                                 f"(precisa de --api-key? URL certa?): {r.text[:200]}")
            r.raise_for_status()
            ids = [m.get("id") for m in r.json().get("data", [])]
            break
        except (httpx.HTTPError, ValueError) as e:
            if time.monotonic() - t0 > espera:
                raise ServidorFora(f"não consegui falar com {base}/models: {e!r}")
            log(f"Aguardando o servidor em {base} ficar pronto ({e.__class__.__name__})...")
            await asyncio.sleep(10)
    if pedido:
        if ids and pedido not in ids:
            log(f"AVISO: o servidor lista {ids}; usando mesmo assim '{pedido}'")
        return pedido
    if len(ids) == 1:
        return ids[0]
    raise SystemExit(f"Informe --modelo. O servidor oferece: {ids}")


async def executar_vlm(a) -> int:
    import httpx
    con = abrir_db(a.db)
    base = a.url.rstrip("/")
    prompt = Path(a.prompt_arquivo).read_text(encoding="utf-8") if a.prompt_arquivo else (a.prompt or PROMPT_PADRAO)
    cabecalhos = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
    limites = httpx.Limits(max_connections=a.concorrencia * 2, max_keepalive_connections=a.concorrencia)
    # trust_env=False: ignora HTTP(S)_PROXY do sistema (o servidor do modelo é local/interno)
    async with httpx.AsyncClient(timeout=httpx.Timeout(a.timeout, connect=10.0), limits=limites,
                                 headers=cabecalhos, trust_env=a.usar_proxy) as cli:
        try:
            modelo = await descobrir_modelo(cli, base, a.modelo, a.espera_servidor)
        except ServidorFora as e:
            log(f"ERRO: {e}")
            return 2
        fonte = a.fonte or f"vlm:{apelido_modelo(modelo)}"
        pend = selecionar_pendentes(con, a, fonte)
        if not pend:
            log(f"Nada pendente para a fonte '{fonte}'.")
            return 0
        log(f"Modelo '{modelo}' -> fonte '{fonte}': {len(pend)} páginas pendentes, "
            f"concorrência {a.concorrencia}, lado {a.lado_px}px")
        estado = Estado(total=len(pend))
        parar = asyncio.Event()
        loop = asyncio.get_running_loop()

        def ao_sinal():
            if parar.is_set():
                log("Saída forçada. O que já terminou está salvo.")
                os._exit(130)
            log("Parando: termino as páginas em andamento e saio (Ctrl+C de novo força).")
            parar.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, ao_sinal)
            except (NotImplementedError, RuntimeError):
                pass
        termico = Termico(a.temp_max_gpu, a.gpu_indice)
        fila: asyncio.Queue = asyncio.Queue()
        for item in pend:
            fila.put_nowait(item)
        url = f"{base}/chat/completions"

        def salvar(item, res):
            con.execute(
                "INSERT OR REPLACE INTO saidas(doc_id,pagina,fonte,status,texto,tentativas,fim,erro,problemas,"
                "meta,atualizado_em) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (item["doc_id"], item["pagina"], fonte, res["status"], res["texto"], res["tentativas"],
                 res["fim"], res["erro"], json.dumps(res["problemas"]),
                 json.dumps(res["meta"], ensure_ascii=False), agora()))
            con.commit()  # checkpoint: cada página fica salva assim que termina

        rend = Renderizador(a.procs_render)

        async def trabalhador():
            while not parar.is_set():
                try:
                    item = fila.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await termico.aguardar(parar)
                    png = await rend.png(item["caminho"], item["pagina"], a.lado_px)
                    res = await processar_pagina(cli, url, modelo, png, item, a, prompt, estado, parar)
                except Interrompido:
                    return  # parando enquanto o servidor estava fora: a página fica pendente
                except ServidorFora as e:
                    if not estado.abortado:
                        log(f"Servidor fora do ar por mais de {a.espera_servidor:.0f}s ({e}). "
                            f"Encerrando; rode de novo depois — o progresso está salvo.")
                    estado.abortado = True
                    parar.set()
                    return
                except Exception as e:
                    res = {"status": "erro", "texto": None, "tentativas": 0, "fim": None,
                           "erro": f"{e.__class__.__name__}: {e}"[:500], "problemas": [], "meta": {}}
                if res is None:
                    continue
                salvar(item, res)
                if res["status"] == "ok":
                    estado.ok += 1
                elif res["status"] == "sinalizada":
                    estado.sinalizadas += 1
                    if a.verboso:
                        log(f"  sinalizada {item['doc_id']}:{item['pagina']} {res['problemas']}")
                else:
                    estado.erros += 1
                    log(f"  erro {item['doc_id']}:{item['pagina']}: {res['erro']}")

        async def progresso():
            while True:
                await asyncio.sleep(a.intervalo)
                log(estado.linha())

        prog = asyncio.create_task(progresso())
        try:
            await asyncio.gather(*(trabalhador() for _ in range(a.concorrencia)))
        finally:
            prog.cancel()
            rend.fechar()
    log("FIM: " + estado.linha())
    if estado.erros:
        log("Páginas com erro serão tentadas de novo na próxima execução do mesmo comando.")
    if estado.abortado:
        return 2
    return 130 if parar.is_set() else 0


def cmd_vlm(a) -> int:
    try:
        a.extra = json.loads(a.extra_json) if a.extra_json else {}
    except json.JSONDecodeError as e:
        raise SystemExit(f"--extra-json inválido: {e}")
    try:
        return asyncio.run(executar_vlm(a))
    except KeyboardInterrupt:
        log("Interrompido. O que já terminou está salvo; rode o mesmo comando para continuar.")
        return 130


# =============================================================== 4. IMPORTAR
def cmd_importar(a) -> int:
    if a.fonte == "nativo":
        raise SystemExit("'nativo' é reservado para o texto da triagem.")
    con = abrir_db(a.db)
    regs = []
    if a.dir:
        for f in sorted(Path(a.dir).glob("*.txt")):
            m = re.match(r"^(.+)__p(\d+)$", f.stem)
            if m:
                regs.append((m.group(1), int(m.group(2)), f.read_text(encoding="utf-8")))
    if a.jsonl:
        with open(a.jsonl, encoding="utf-8") as f:
            for linha in f:
                if linha.strip():
                    o = json.loads(linha)
                    regs.append((o["doc_id"], int(o["pagina"]), o["texto"]))
    validos = 0
    for doc_id, pag, texto in regs:
        if not con.execute("SELECT 1 FROM paginas WHERE doc_id=? AND pagina=?", (doc_id, pag)).fetchone():
            log(f"AVISO: {doc_id}:{pag} não existe na triagem — ignorado")
            continue
        con.execute("INSERT OR REPLACE INTO saidas(doc_id,pagina,fonte,status,texto,meta,atualizado_em) "
                    "VALUES (?,?,?,?,?,?,?)", (doc_id, pag, a.fonte, "ok", texto,
                                               json.dumps({"importado_de": a.dir or a.jsonl}), agora()))
        validos += 1
    con.commit()
    log(f"{validos} página(s) importadas como fonte '{a.fonte}'.")
    return 0 if validos else 1


# ================================================================ 5. VALIDAR
def carregar_fonte(con, fonte: str, como_referencia: bool = False) -> dict:
    """{(doc_id, pagina): (texto, bandas)}. O nativo só serve de referência em páginas ok/misto."""
    if fonte == "nativo":
        cats = ("ok", "misto") if como_referencia else ("ok", "misto", "vlm")
        q = (f"SELECT doc_id, pagina, texto_nativo, n_topo, n_base FROM paginas WHERE n_chars > 0 "
             f"AND categoria IN ({','.join('?' * len(cats))})")
        return {(r[0], r[1]): (r[2], (r[3] or 0, r[4] or 0)) for r in con.execute(q, cats)}
    return {(r[0], r[1]): (r[2] or "", None) for r in con.execute(
        "SELECT doc_id, pagina, texto FROM saidas WHERE fonte=? AND status IN ('ok','sinalizada')", (fonte,))}


def contexto_do_documento(con, doc_id: str, n_paginas: int, alternativos=()) -> Contexto:
    """Aprende cabeçalho/rodapé com o texto nativo (posição conhecida). Se o documento for quase todo
    sem texto nativo (escaneado), aprende com as transcrições `alternativos` (sem geometria).
    O vocabulário (para desfazer hifenização) usa o nativo e as transcrições de VLM do documento
    (nunca o gabarito, para não contaminar a avaliação)."""
    nativos = [(r[0], (r[1] or 0, r[2] or 0)) for r in con.execute(
        "SELECT texto_nativo, n_topo, n_base FROM paginas WHERE doc_id=? AND categoria IN ('ok','misto')",
        (doc_id,))]
    alternativos = [t for t in alternativos if t]
    usar_nativo = len(nativos) >= 0.5 * n_paginas or not alternativos
    ctx = contexto_doc(nativos if usar_nativo else [(t, None) for t in alternativos], n_paginas)
    for (t,) in con.execute("SELECT texto FROM saidas WHERE doc_id=? AND fonte LIKE 'vlm:%' "
                            "AND status IN ('ok','sinalizada')", (doc_id,)):
        ctx.vocab.update(w.lower() for w in RE_PALAVRA.findall(t or ""))
    return ctx


def contextos_por_doc(con, textos_por_doc: dict | None = None) -> dict:
    textos_por_doc = textos_por_doc or {}
    return {d["doc_id"]: contexto_do_documento(con, d["doc_id"], d["n_paginas"],
                                               textos_por_doc.get(d["doc_id"], ()))
            for d in con.execute("SELECT doc_id, n_paginas FROM docs").fetchall()}


def comparar(ref: str, hip: str, Lev) -> dict:
    tr, th = tokens(ref), tokens(hip)
    cr, ch = Counter(tr), Counter(th)
    inter = sum((cr & ch).values())
    recall = inter / len(tr) if tr else (1.0 if not th else 0.0)
    prec = inter / len(th) if th else (1.0 if not tr else 0.0)
    nr, nh = Counter(RE_NUMERO.findall(ref)), Counter(RE_NUMERO.findall(hip))
    falt, extra = nr - nh, nh - nr
    a_, b_ = " ".join(ref.split()), " ".join(hip.split())
    return {
        "recall": round(recall, 4), "precisao": round(prec, 4),
        "cer": round(Lev.distance(a_, b_) / max(1, len(a_)), 4),
        "wer": round(Lev.distance(tr, th) / max(1, len(tr)), 4),
        "tokens_ref": len(tr), "tokens_hip": len(th),
        "tokens_faltando": len(tr) - inter, "tokens_extras": len(th) - inter,
        "nums_faltando": sum(falt.values()), "nums_extras": sum(extra.values()),
        "ex_faltando": " ".join(list(falt.elements())[:6]), "ex_extras": " ".join(list(extra.elements())[:6]),
    }


def _q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))] if v else float("nan")


def cmd_validar(a) -> int:
    try:
        from rapidfuzz.distance import Levenshtein as Lev
    except ImportError:
        raise SystemExit("Instale rapidfuzz: pip install rapidfuzz")
    con = abrir_db(a.db)
    refs = carregar_fonte(con, a.ref, como_referencia=True)
    hips = carregar_fonte(con, a.hip)
    cats = {(r[0], r[1]): r[2] for r in con.execute("SELECT doc_id, pagina, categoria FROM paginas")}
    # contra o gabarito, toda página do gabarito conta (se a hipótese não tem a página, ela "zerou");
    # contra o nativo, compara só onde as duas fontes existem.
    chaves = sorted(set(refs) & set(hips)) if a.ref == "nativo" else sorted(refs)
    if not chaves:
        log(f"Nenhuma página com '{a.ref}' e '{a.hip}' ao mesmo tempo.")
        return 1
    por_doc = defaultdict(list)
    for (doc_id, _), (t, _) in hips.items():
        por_doc[doc_id].append(t)
    ctx = contextos_por_doc(con, por_doc)
    linhas, sinalizadas = [], 0
    con.execute("DELETE FROM validacao WHERE ref=? AND hip=?", (a.ref, a.hip))
    for k in chaves:
        c = ctx.get(k[0])
        ref_l = limpar_pagina(refs[k][0], c, bandas=refs[k][1])
        hip_bruto, hip_bandas = hips.get(k, ("", None))
        hip_l = limpar_pagina(hip_bruto, c, bandas=hip_bandas)
        m = comparar(ref_l, hip_l, Lev)
        s = [] if k in hips else ["ausente"]  # mínimo absoluto de tokens evita acusar páginas curtas à toa
        if m["recall"] < a.min_recall and m["tokens_faltando"] >= a.min_tokens_sinal:
            s.append("omissao")
        if m["precisao"] < a.min_precisao and m["tokens_extras"] >= a.min_tokens_sinal:
            s.append("adicao")
        if m["nums_faltando"]:
            s.append("numero_faltando")
        if m["nums_extras"]:
            s.append("numero_extra")
        if k in hips:
            s += [p for p in problemas_sem_referencia(hip_bruto, None)
                  if p != "vazio" or hip_bruto.strip() != SENTINELA_VAZIA]
        sinalizadas += bool(s)
        con.execute("INSERT OR REPLACE INTO validacao VALUES (?,?,?,?,?,?,?)",
                    (k[0], k[1], a.ref, a.hip, cats.get(k), json.dumps(m), json.dumps(s)))
        linhas.append({"doc_id": k[0], "pagina": k[1], "categoria": cats.get(k), **m, "sinais": " ".join(s)})
    con.commit()
    saida = a.saida or f"validacao_{slug(a.hip)}_vs_{slug(a.ref)}.csv"
    with open(saida, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    def resumo(nome, ls):
        rec = [x["recall"] for x in ls]
        pre = [x["precisao"] for x in ls]
        cer = [x["cer"] for x in ls]
        sin = sum(1 for x in ls if x["sinais"])
        print(f"  {nome:<10} {len(ls):>6}  {statistics.median(rec):>8.3f} {_q(rec, .05):>8.3f}  "
              f"{statistics.median(pre):>8.3f} {_q(pre, .05):>8.3f}  {statistics.median(cer):>7.3f}  "
              f"{100 * sin / len(ls):>6.1f}%")

    print(f"\nValidação: hipótese '{a.hip}' x referência '{a.ref}'  (textos limpos dos dois lados)")
    print(f"  {'categoria':<10} {'págs':>6}  {'recall':>8} {'p5':>8}  {'precisão':>8} {'p5':>8}  "
          f"{'CER':>7}  {'sinal.':>7}")
    resumo("TODAS", linhas)
    for cat in ("ok", "misto", "vlm", "vazia"):
        ls = [x for x in linhas if x["categoria"] == cat]
        if ls:
            resumo(cat, ls)
    cont = Counter(s for x in linhas for s in x["sinais"].split())
    print("  sinais: " + (", ".join(f"{k}={v}" for k, v in cont.most_common()) or "nenhum"))
    if a.ref == "nativo":
        print("  Obs.: em páginas 'misto' o VLM lê texto dentro de imagens que o nativo não tem, então\n"
              "  'adicao' e 'numero_extra' são esperados ali; confira no HTML do comando revisar.")
    print(f"  Relatório por página: {saida}\n")
    return 0


# ================================================================ 6. REVISAR
def diff_html(a: str, b: str) -> str:
    """Diferença palavra a palavra (linhas em branco não contam como diferença)."""
    a, b = re.sub(r"\n\s*\n+", "\n", a), re.sub(r"\n\s*\n+", "\n", b)
    ta, tb = re.findall(r"\S+|\n", a), re.findall(r"\S+|\n", b)
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)

    def fmt(ts):
        return " ".join("<br>" if x == "\n" else html.escape(x) for x in ts)

    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out.append(fmt(ta[i1:i2]))
        if op in ("delete", "replace"):
            out.append(f"<del>{fmt(ta[i1:i2])}</del>")
        if op in ("insert", "replace"):
            out.append(f"<ins>{fmt(tb[j1:j2])}</ins>")
    return " ".join(out)


def cmd_revisar(a) -> int:
    con = abrir_db(a.db)
    rows = con.execute("SELECT v.*, d.caminho FROM validacao v JOIN docs d USING(doc_id) "
                       "WHERE v.ref=? AND v.hip=?", (a.ref, a.hip)).fetchall()
    if not rows:
        log("Rode `validar` com as mesmas --ref/--hip antes.")
        return 1
    if a.somente_sinalizadas:
        rows = [r for r in rows if json.loads(r["sinais"])]
    rnd = random.Random(a.seed)
    rows = list(rows)
    rnd.shuffle(rows)
    rows = sorted(rows[: a.n], key=lambda r: (-len(json.loads(r["sinais"])), r["doc_id"], r["pagina"]))
    refs = carregar_fonte(con, a.ref, como_referencia=True)
    hips = carregar_fonte(con, a.hip)
    por_doc = defaultdict(list)
    for (doc_id, _), (t, _) in hips.items():
        por_doc[doc_id].append(t)
    ctx = contextos_por_doc(con, por_doc)
    blocos = []
    vazio = ("", None)
    for r in rows:
        k = (r["doc_id"], r["pagina"])
        c = ctx.get(r["doc_id"])
        (tr, br), (th, bh) = refs.get(k, vazio), hips.get(k, vazio)
        m, s = json.loads(r["metricas"]), json.loads(r["sinais"])
        jpg = base64.b64encode(render_png(r["caminho"], r["pagina"], 1100, "jpg")).decode()
        blocos.append(f"""<section><h2>{html.escape(r['doc_id'])} · página {r['pagina']} ·
<span class=cat>{r['categoria']}</span> {' '.join(f'<span class=s>{x}</span>' for x in s)}</h2>
<p class=m>recall {m['recall']:.3f} · precisão {m['precisao']:.3f} · CER {m['cer']:.3f} · WER {m['wer']:.3f}
· números faltando {m['nums_faltando']} ({html.escape(m['ex_faltando'])}) · extras {m['nums_extras']}
({html.escape(m['ex_extras'])})</p>
<div class=g><img src="data:image/jpeg;base64,{jpg}"><div class=t>
{diff_html(limpar_pagina(tr, c, bandas=br), limpar_pagina(th, c, bandas=bh))}</div></div></section>""")
    pagina = f"""<!doctype html><meta charset=utf-8><title>Revisão {html.escape(a.hip)}</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}}
section{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:0 0 20px}}
h2{{font-size:15px;margin:0 0 4px}} .m{{color:#555;margin:0 0 8px;font-size:12px}}
.g{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px}}
img{{width:100%;border:1px solid #ccc}} .t{{white-space:normal;font-size:13px}}
del{{background:#fdd;color:#900}} ins{{background:#dfd;color:#060;text-decoration:none}}
.cat{{background:#eef;padding:1px 6px;border-radius:4px}} .s{{background:#fec;padding:1px 6px;border-radius:4px;margin-left:4px}}
@media(max-width:800px){{.g{{grid-template-columns:1fr}}}}</style>
<h1>Revisão: <ins>{html.escape(a.hip)}</ins> × <del>{html.escape(a.ref)}</del></h1>
<p>Vermelho riscado = está na referência e faltou na hipótese. Verde = a hipótese escreveu e a
referência não tem. Os dois textos já passaram pela mesma limpeza.</p>{''.join(blocos)}"""
    saida = a.saida or f"revisao_{slug(a.hip)}.html"
    Path(saida).write_text(pagina, encoding="utf-8")
    log(f"{len(blocos)} páginas em {saida}")
    return 0


# =============================================================== 7. EXPORTAR
def escolher_texto(p, saidas_pag: dict, vlm_fonte, fonte_unica, excluir_sinalizadas):
    """Devolve (fonte, texto, status, bandas) ou None.
    Política padrão: nativo nas páginas 'ok'; VLM nas 'vlm'; VLM (ou nativo, se faltar) nas 'misto'."""
    def de(fonte):
        if fonte == "nativo":
            if (p["n_chars"] or 0) > 0:
                return ("nativo", p["texto_nativo"], "ok", (p["n_topo"] or 0, p["n_base"] or 0))
            return None
        s = saidas_pag.get(fonte)
        if s and s["status"] in ("ok", "sinalizada") and not (excluir_sinalizadas and s["status"] == "sinalizada"):
            return (fonte, s["texto"] or "", s["status"], None)
        return None

    if fonte_unica:
        return de(fonte_unica)
    cat = p["categoria"]
    if cat == "ok":
        return de("nativo")
    if cat == "misto":
        return (de(vlm_fonte) if vlm_fonte else None) or de("nativo")
    if cat == "vlm":
        return de(vlm_fonte) if vlm_fonte else None
    return None  # vazia


def juntar_paginas(textos: list[str], marcar: bool) -> str:
    if marcar:
        return "\n\f\n".join(textos)
    saida = ""
    for t in textos:
        if not t:
            continue
        if saida:
            fim, ini = saida.rstrip()[-1:], t.lstrip()[:1]
            # parágrafo que continua na página seguinte
            sep = " " if (fim and fim not in ".:;!?)\"»”" and ini.islower()) else "\n"
            saida = saida.rstrip() + sep + t.lstrip()
        else:
            saida = t
    return saida.strip() + "\n"


def cmd_exportar(a) -> int:
    con = abrir_db(a.db)
    fontes_vlm = [r[0] for r in con.execute("SELECT DISTINCT fonte FROM saidas WHERE fonte LIKE 'vlm:%'")]
    vlm_fonte = a.vlm
    if not vlm_fonte and not a.fonte_unica:
        if len(fontes_vlm) == 1:
            vlm_fonte = fontes_vlm[0]
        elif len(fontes_vlm) > 1:
            raise SystemExit(f"Há várias fontes VLM {fontes_vlm}; escolha uma com --vlm.")
    pasta = Path(a.saida)
    pasta.mkdir(parents=True, exist_ok=True)
    docs = con.execute("SELECT * FROM docs WHERE triado_em IS NOT NULL ORDER BY doc_id").fetchall()
    manifesto = {"gerado_em": agora(), "versao": VERSAO, "vlm_fonte": vlm_fonte,
                 "fonte_unica": a.fonte_unica, "docs": []}
    faltando_total = 0
    with open(pasta / "corpus.jsonl", "w", encoding="utf-8") as fj:
        for d in docs:
            pags = con.execute("SELECT * FROM paginas WHERE doc_id=? ORDER BY pagina", (d["doc_id"],)).fetchall()
            saidas = defaultdict(dict)
            for s in con.execute("SELECT * FROM saidas WHERE doc_id=?", (d["doc_id"],)):
                saidas[s["pagina"]][s["fonte"]] = s
            escolhas = {p["pagina"]: escolher_texto(p, saidas[p["pagina"]], vlm_fonte, a.fonte_unica,
                                                    a.excluir_sinalizadas) for p in pags}
            ctx = contexto_do_documento(con, d["doc_id"], d["n_paginas"],
                                        [e[1] for e in escolhas.values() if e and e[0] != "nativo"])
            usados, faltando, vazias, sinal, textos = Counter(), [], 0, 0, []
            for p in pags:
                e = escolhas[p["pagina"]]
                if e is None:
                    if p["categoria"] == "vazia":
                        vazias += 1
                    else:
                        faltando.append(p["pagina"])
                    continue
                fonte, bruto, status, bandas = e
                limpo = limpar_pagina(bruto, ctx, a.mascarar_cpf, bandas=bandas)
                usados[fonte] += 1
                sinal += status == "sinalizada"
                if not limpo:
                    vazias += 1
                textos.append(limpo)
                fj.write(json.dumps({"doc_id": d["doc_id"], "pagina": p["pagina"], "fonte": fonte,
                                     "status": status, "categoria": p["categoria"], "texto": limpo},
                                    ensure_ascii=False) + "\n")
            final = juntar_paginas(textos, a.marcar_paginas)
            (pasta / f"{d['doc_id']}.txt").write_text(final, encoding="utf-8")
            residuo = contar_lixo(final)
            faltando_total += len(faltando)
            manifesto["docs"].append({
                "doc_id": d["doc_id"], "arquivo": d["caminho"], "sha256": d["sha256"],
                "paginas_pdf": d["n_paginas"], "paginas_exportadas": len(textos), "paginas_vazias": vazias,
                "paginas_faltando": faltando, "fontes_usadas": dict(usados), "sinalizadas_usadas": sinal,
                "residuo_lixo": residuo, "caracteres": len(final),
                "boilerplate_removido": dict(ctx.freq.most_common(25)),
                "avisos": json.loads(d["avisos"] or "[]")})
            fl = f" | FALTANDO {len(faltando)} págs: {faltando[:10]}{'...' if len(faltando) > 10 else ''}" \
                if faltando else ""
            log(f"{d['doc_id']}: {len(textos)}/{d['n_paginas']} págs exportadas (vazias={vazias}, "
                f"fontes={dict(usados)}, sinalizadas={sinal}, resíduo={residuo}){fl}")
    (pasta / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"Saída em {pasta}/ (um .txt por documento, corpus.jsonl e manifesto.json)")
    if faltando_total and not a.permitir_faltantes:
        log(f"ATENÇÃO: {faltando_total} página(s) sem texto aceitável. Rode o `vlm` para elas "
            f"ou use --permitir-faltantes.")
        return 1
    return 0


# ================================================================= 8. STATUS
def cmd_status(a) -> int:
    con = abrir_db(a.db)
    docs = con.execute("SELECT * FROM docs ORDER BY doc_id").fetchall()
    if not docs:
        log("Banco vazio: rode `triagem` primeiro.")
        return 1
    fontes = [r[0] for r in con.execute("SELECT DISTINCT fonte FROM saidas ORDER BY fonte")]
    print(f"\n{'documento':<32} {'págs':>6} {'ok':>6} {'misto':>6} {'vlm':>6} {'vazia':>6}  triagem")
    for d in docs:
        c = Counter(dict(con.execute("SELECT categoria, COUNT(*) FROM paginas WHERE doc_id=? "
                                     "GROUP BY categoria", (d["doc_id"],)).fetchall()))
        print(f"{d['doc_id'][:32]:<32} {d['n_paginas']:>6} {c['ok']:>6} {c['misto']:>6} {c['vlm']:>6} "
              f"{c['vazia']:>6}  {'completa' if d['triado_em'] else 'EM ANDAMENTO'}")
    for f in fontes:
        st = Counter(dict(con.execute("SELECT status, COUNT(*) FROM saidas WHERE fonte=? GROUP BY status",
                                      (f,)).fetchall()))
        pend = con.execute("SELECT COUNT(*) FROM paginas p WHERE p.categoria IN ('vlm','misto') AND NOT EXISTS "
                           "(SELECT 1 FROM saidas s WHERE s.doc_id=p.doc_id AND s.pagina=p.pagina AND "
                           "s.fonte=? AND s.status IN ('ok','sinalizada'))", (f,)).fetchone()[0]
        extra = f" | páginas vlm/misto ainda sem saída: {pend}" if f.startswith("vlm:") else ""
        print(f"fonte {f}: ok={st['ok']} sinalizadas={st['sinalizada']} erros={st['erro']}{extra}")
    print()
    return 0


# ====================================================================== CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="pipeline.sqlite", help="banco SQLite com todo o estado (padrão: %(default)s)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("triagem", help="analisa as páginas e extrai o texto nativo")
    t.add_argument("entradas", nargs="+", help="arquivos PDF e/ou pastas")
    t.add_argument("--faixa-borda", type=float, default=0.08,
                   help="fração da altura considerada cabeçalho/rodapé (padrão: %(default)s)")
    t.add_argument("--min-chars", type=int, default=40, help="abaixo disso a página tem 'pouco texto'")
    t.add_argument("--max-invalidos", type=float, default=0.05, help="fração máx. de caracteres inválidos (\ufffd, área privada)")
    t.add_argument("--min-stop", type=float, default=0.02,
                   help="fração mín. de palavras comuns (de, que, the...) para o texto ser considerado legível")
    t.add_argument("--min-vogais", type=float, default=0.30,
                   help="fração mín. de vogais entre as letras (texto real em PT/EN fica perto de 0,45)")
    t.add_argument("--max-img", type=float, default=0.30, help="cobertura de imagem a partir da qual a página é 'misto'")
    t.add_argument("--min-desenhos", type=int, default=50, help="nº de vetores que indica texto convertido em curvas")
    t.add_argument("--tabelas-para-vlm", action="store_true", help="manda páginas com cara de tabela para o VLM ('misto')")
    t.add_argument("--refazer", action="store_true", help="descarta tudo do documento e tria de novo")

    s = sub.add_parser("amostra", help="sorteia páginas e gera modelos de gabarito")
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--saida", default="amostra")
    s.add_argument("--lado-px", type=int, default=1600)
    s.add_argument("--sobrescrever", action="store_true", help="reescreve .txt já existentes (cuidado!)")

    v = sub.add_parser("vlm", help="transcreve páginas com um VLM local (assíncrono, com checkpoint)")
    v.add_argument("--url", default="http://localhost:8000/v1", help="base da API compatível com OpenAI")
    v.add_argument("--modelo", help="nome do modelo no servidor (se omitido e houver só um, usa ele)")
    v.add_argument("--fonte", help="nome da fonte no banco (padrão: vlm:<modelo>)")
    v.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    v.add_argument("--usar-proxy", action="store_true", help="respeita HTTP(S)_PROXY do ambiente (padrão: ignora)")
    v.add_argument("--selecao", choices=["precisa", "vlm", "todas"], default="precisa",
                   help="precisa = vlm+misto (padrão); vlm = só as sem texto utilizável; todas = tudo")
    v.add_argument("--paginas", help="arquivo com 'doc_id:pagina' por linha (ex.: amostra/paginas.txt)")
    v.add_argument("--docs", help="restringe a estes doc_id (separados por vírgula)")
    v.add_argument("--limite", type=int, help="processa no máximo N páginas (teste)")
    v.add_argument("--concorrencia", type=int, default=8, help="requisições simultâneas (padrão: %(default)s)")
    v.add_argument("--procs-render", type=int, default=2, help="processos que renderizam páginas")
    v.add_argument("--lado-px", type=int, default=1600, help="maior lado da imagem enviada (padrão: %(default)s)")
    v.add_argument("--max-tokens", type=int, default=4096)
    v.add_argument("--max-tokens-teto", type=int, default=8192, help="limite ao dobrar max-tokens após truncamento")
    v.add_argument("--tentativas", type=int, default=3)
    v.add_argument("--temp-retentativa", type=float, default=0.2)
    v.add_argument("--timeout", type=float, default=300, help="segundos por requisição")
    v.add_argument("--espera-servidor", type=float, default=1800,
                   help="quanto esperar o servidor voltar antes de encerrar (s)")
    v.add_argument("--prompt", help="prompt inline (ex.: 'OCR:' para PaddleOCR-VL)")
    v.add_argument("--prompt-arquivo", help="arquivo com o prompt")
    v.add_argument("--ancora", action="store_true", help="inclui o texto nativo no prompt como referência")
    v.add_argument("--ancora-max-chars", type=int, default=6000)
    v.add_argument("--extra-json", help='campos extras do payload, ex.: \'{"repetition_penalty": 1.05}\'')
    v.add_argument("--refazer-sinalizadas", action="store_true")
    v.add_argument("--refazer-tudo", action="store_true")
    v.add_argument("--temp-max-gpu", type=int, help="pausa envios acima desta temperatura (°C; requer nvidia-ml-py)")
    v.add_argument("--gpu-indice", type=int, default=0)
    v.add_argument("--intervalo", type=float, default=30, help="segundos entre linhas de progresso")
    v.add_argument("--verboso", action="store_true")

    i = sub.add_parser("importar", help="carrega gabarito ou saída de outra ferramenta como fonte")
    i.add_argument("--fonte", required=True, help="nome da fonte (ex.: gabarito, docling)")
    i.add_argument("--dir", help="pasta com arquivos <doc_id>__pNNNNN.txt")
    i.add_argument("--jsonl", help='arquivo com linhas {"doc_id":..., "pagina":..., "texto":...}')

    va = sub.add_parser("validar", help="compara uma fonte com uma referência")
    va.add_argument("--hip", required=True, help="fonte avaliada (ex.: vlm:qwen3.5-9b, nativo)")
    va.add_argument("--ref", default="nativo", help="referência: nativo (padrão) ou gabarito")
    va.add_argument("--min-recall", type=float, default=0.95)
    va.add_argument("--min-precisao", type=float, default=0.95)
    va.add_argument("--min-tokens-sinal", type=int, default=3,
                    help="só acusa omissão/adição se faltarem/sobrarem ao menos N palavras")
    va.add_argument("--saida", help="CSV por página")

    r = sub.add_parser("revisar", help="HTML lado a lado para revisão humana")
    r.add_argument("--hip", required=True)
    r.add_argument("--ref", default="nativo")
    r.add_argument("--n", type=int, default=30)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--somente-sinalizadas", action="store_true")
    r.add_argument("--saida")

    e = sub.add_parser("exportar", help="gera .txt por documento + corpus.jsonl + manifesto.json")
    e.add_argument("--saida", default="corpus")
    e.add_argument("--vlm", help="fonte VLM a usar nas páginas misto/vlm")
    e.add_argument("--fonte-unica", help="exporta tudo de uma só fonte (experimentos)")
    e.add_argument("--excluir-sinalizadas", action="store_true")
    e.add_argument("--mascarar-cpf", action="store_true")
    e.add_argument("--marcar-paginas", action="store_true", help="separa páginas com \\f (form feed)")
    e.add_argument("--permitir-faltantes", action="store_true")

    sub.add_parser("status", help="resumo do andamento")

    a = ap.parse_args(argv)
    return {"triagem": cmd_triagem, "amostra": cmd_amostra, "vlm": cmd_vlm, "importar": cmd_importar,
            "validar": cmd_validar, "revisar": cmd_revisar, "exportar": cmd_exportar,
            "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
