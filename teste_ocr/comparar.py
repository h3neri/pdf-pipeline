#!/usr/bin/env python3
"""
comparar.py — compara as transcrições do teste bruto (teste_ocr/saidas/) com o texto nativo do PDF.

Não há gabarito nesta fase. A régua é o texto nativo (a camada de texto do PDF), que é exato nas
páginas nascidas digitais (98,8% de concordância com um OCR independente; veja o CLAUDE.md). Por isso
o resultado se lê por grupo de página (coluna "grupo", tirada do tipo no manifesto.csv):
  completo  o texto todo está na camada de texto: cobertura baixa = o método omitiu;
            precisão baixa = o método acrescentou ou trocou palavras
  imagem    pode haver texto dentro de imagens (prints, slides, capas): o nativo é incompleto e
            precisão abaixo de 1 é esperada, porque o modelo lê o que o nativo não tem
  defeito   o nativo tem defeitos de espaço (palavras grudadas, letras espaçadas): a diferença
            pode ser culpa do nativo, não do método

Métricas por página e método. São "sacos de palavras": a ordem e a estrutura (tabelas) NÃO entram.
  cobertura  fração das palavras do nativo que aparecem na saída (contando repetições)
  precisao   fração das palavras da saída que existem no nativo
  numeros    fração dos números do nativo que aparecem iguais na saída ("2024" -> "2021" e
             "250,000" -> "250.000" contam como erro; nas palavras, quase não aparecem)
Uma página com 99,8% de cobertura pode ter uma palavra trocada que inverte o sentido. Por isso há
as armadilhas (trechos conferidos na imagem) e o --detalhe.

Uso (na pasta pipeline/):
  python teste_ocr/comparar.py                           # todos os métodos de teste_ocr/saidas/
  python teste_ocr/comparar.py --metodos gemma4-e4b__gemma_fiel paddleocr-vl-1.6__paddleocr_ocr
  python teste_ocr/comparar.py --detalhe 13              # palavras que faltam/sobram na página 13
  python teste_ocr/comparar.py --csv experimento/teste_ocr_metricas.csv   # só números: pode ir ao git

Dependências: só a biblioteca padrão do Python 3.9+.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Optional

AQUI = Path(__file__).resolve().parent
RE_PALAVRA = re.compile(r"\w+")
RE_NUMERO = re.compile(r"\d+(?:[.,]\d+)*")
SINAIS = ("truncado", "repeticao", "vazio", "sem_texto", "cerca_markdown", "cjk")

# grupo de cada tipo do manifesto (veja a docstring)
GRUPO = {
    "texto_em_imagem": "imagem", "slide": "imagem", "design_com_fundo": "imagem",
    "capa_ou_divisor": "imagem", "em_branco": "imagem", "fluxograma": "imagem",
    "defeito_do_nativo": "defeito",
}


# ----------------------------------------------------------------------------- leitura

def palavras(texto: str) -> Counter:
    """Palavras normalizadas (NFKC, minúsculas). Sequências só de "_" (campos ___) não contam."""
    t = unicodedata.normalize("NFKC", texto).lower()
    return Counter(p for p in RE_PALAVRA.findall(t) if not set(p) <= {"_"})


def numeros(texto: str) -> Counter:
    return Counter(RE_NUMERO.findall(unicodedata.normalize("NFKC", texto)))


def ler_log(pasta: Path) -> dict:
    """execucao.jsonl -> {pagina: último registro} (vale a última tentativa de cada página)."""
    log = {}
    arq = pasta / "execucao.jsonl"
    if arq.exists():
        for linha in open(arq, encoding="utf-8"):
            if linha.strip():
                r = json.loads(linha)
                log[r["pagina"]] = r
    return log


def paginas_de(pasta: Path) -> dict:
    """{numero da página: caminho do txt} para os pNNNNN.txt da pasta."""
    out = {}
    for f in pasta.glob("p*.txt"):
        m = re.fullmatch(r"p(\d+)\.txt", f.name)
        if m:
            out[int(m.group(1))] = f
    return out


def ler_armadilhas(arq: Optional[Path]) -> list:
    if not arq or not arq.exists():
        return []
    with open(arq, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("pagina", "").strip()]


# ----------------------------------------------------------------------------- comparação

def comparar_pagina(ref: str, hip: str) -> dict:
    a, b = palavras(ref), palavras(hip)
    comum = sum((a & b).values())
    na, nb = numeros(ref), numeros(hip)
    return {
        "palavras_ref": sum(a.values()), "palavras": sum(b.values()),
        "cobertura": comum / sum(a.values()) if a else None,
        "precisao": comum / sum(b.values()) if b else None,
        "numeros_ref": sum(na.values()),
        "numeros": sum((na & nb).values()) / sum(na.values()) if na else None,
    }


def fmt(x: Optional[float], casas: int = 3) -> str:
    return "-" if x is None else f"{x:.{casas}f}"


def media(vals) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def tabela(linhas: list, cols: list) -> None:
    larg = {c: max(len(c), *(len(str(l.get(c, ""))) for l in linhas)) for c in cols}
    print("  ".join(c.ljust(larg[c]) for c in cols))
    for l in linhas:
        print("  ".join(str(l.get(c, "")).ljust(larg[c]) for c in cols))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saidas", default=str(AQUI / "saidas"), help="pasta das saídas (padrão: teste_ocr/saidas)")
    ap.add_argument("--manifesto", default=str(AQUI / "manifesto.csv"))
    ap.add_argument("--ref", default="nativo", help="método usado como régua (padrão: nativo)")
    ap.add_argument("--metodos", nargs="*", help="pastas de saidas/ a comparar (padrão: todas menos a régua)")
    ap.add_argument("--armadilhas", default=str(AQUI / "armadilhas.csv"),
                    help="CSV pagina,esperado,nota com trechos conferidos na imagem")
    ap.add_argument("--detalhe", type=int, metavar="PAGINA", help="mostra palavras e números que faltam/sobram")
    ap.add_argument("--csv", help="grava as métricas por página e método (só números) neste arquivo")
    a = ap.parse_args()

    saidas = Path(a.saidas)
    pasta_ref = saidas / a.ref
    if not pasta_ref.is_dir():
        print(f"Não achei {pasta_ref}. Rode antes: python teste_ocr/transcrever.py nativo")
        return 1
    ref = paginas_de(pasta_ref)
    tipos = {}
    if Path(a.manifesto).exists():
        with open(a.manifesto, encoding="utf-8") as f:
            tipos = {int(r["pagina_teste"]): r["tipo"] for r in csv.DictReader(f)}
    metodos = a.metodos or sorted(p.name for p in saidas.iterdir()
                                  if p.is_dir() and p.name != a.ref and paginas_de(p))
    if not metodos:
        print(f"Nenhum método em {saidas}/ além de {a.ref}.")
        return 1

    # ---- métricas por página
    linhas, logs, textos = [], {}, {}
    texto_ref = {n: f.read_text(encoding="utf-8") for n, f in ref.items()}
    for m in metodos:
        pasta = saidas / m
        pags = paginas_de(pasta)
        logs[m] = ler_log(pasta)
        textos[m] = {n: f.read_text(encoding="utf-8") for n, f in pags.items()}
        for n in sorted(pags):
            if n not in texto_ref:
                continue
            r = logs[m].get(n, {})
            tipo = tipos.get(n, "")
            linhas.append({
                "metodo": m, "pagina": n, "tipo": tipo, "grupo": GRUPO.get(tipo, "completo"),
                **comparar_pagina(texto_ref[n], textos[m][n]),
                "segundos": r.get("segundos"), "tokens_entrada": r.get("tokens_entrada"),
                "tokens_saida": r.get("tokens_saida"), "motivo_parada": r.get("motivo_parada"),
                "sinais": ",".join(r.get("sinais", [])),
            })

    # ---- resumo por método
    print(f"Régua: {a.ref} ({len(ref)} páginas). Grupo 'completo' = páginas cujo texto está todo na camada de texto.\n")
    resumo = []
    for m in metodos:
        L = [l for l in linhas if l["metodo"] == m]
        comp = [l for l in L if l["grupo"] == "completo"]
        comp_ok = [l for l in comp if "truncado" not in l["sinais"]]
        sinais = Counter(s for l in L for s in l["sinais"].split(",") if s)
        seg = [l["segundos"] for l in L if l["segundos"] is not None]
        resumo.append({
            "metodo": m, "paginas": len(L),
            "completo: cobertura": fmt(media(l["cobertura"] for l in comp)),
            "sem truncadas": fmt(media(l["cobertura"] for l in comp_ok)),
            "precisao": fmt(media(l["precisao"] for l in comp)),
            "numeros": fmt(media(l["numeros"] for l in comp)),
            ">=0.99": f"{sum(1 for l in comp if (l['cobertura'] or 0) >= .99 and (l['precisao'] or 0) >= .99)}/{len(comp)}",
            "s/pag (mediana)": fmt(statistics.median(seg), 1) if seg else "-",
            "tokens saida": sum(l["tokens_saida"] or 0 for l in L),
            **{s: sinais.get(s, 0) for s in ("truncado", "repeticao", "vazio")},
        })
    tabela(resumo, list(resumo[0]))

    # ---- por página
    print("\ncobertura/precisao por página (- = método não rodou a página)")
    cab = ["pag", "tipo"] + [m.split("__")[0][:18] for m in metodos]
    por_pag = []
    for n in sorted(texto_ref):
        lin = {"pag": n, "tipo": tipos.get(n, "")[:20]}
        for m, c in zip(metodos, cab[2:]):
            l = next((x for x in linhas if x["metodo"] == m and x["pagina"] == n), None)
            lin[c] = "-" if l is None else f"{fmt(l['cobertura'], 2)}/{fmt(l['precisao'], 2)}" + (
                " T" if "truncado" in l["sinais"] else "")
        por_pag.append(lin)
    tabela(por_pag, cab)
    print("T = truncada (bateu no max_tokens; quase sempre laço de repetição)")

    # ---- armadilhas
    arm = ler_armadilhas(Path(a.armadilhas))
    if arm:
        print(f"\nArmadilhas ({a.armadilhas}): o trecho conferido na imagem aparece na saída tantas vezes")
        print("quantas aparece no nativo?  ok = sim   k/n = só k de n   ERRO = nenhuma   - = não rodou a página")
        print("T = a página foi truncada (o trecho pode estar depois do corte)\n")
        cab = ["pag", "esperado"] + [m.split("__")[0][:18] for m in metodos] + ["nota"]
        lin_arm = []
        for r in arm:
            n, esp = int(r["pagina"]), unicodedata.normalize("NFC", r["esperado"])
            n_ref = max(1, unicodedata.normalize("NFC", texto_ref.get(n, "")).count(esp))
            lin = {"pag": n, "esperado": esp[:28], "nota": r.get("nota", "")[:60]}
            for m, c in zip(metodos, cab[2:-1]):
                t = textos[m].get(n)
                if t is None:
                    lin[c] = "-"
                    continue
                k = unicodedata.normalize("NFC", t).count(esp)
                lin[c] = ("ok" if k >= n_ref else f"{k}/{n_ref}" if k else "ERRO") + (
                    " T" if "truncado" in logs[m].get(n, {}).get("sinais", []) else "")
            lin_arm.append(lin)
        tabela(lin_arm, cab)

    # ---- detalhe de uma página
    if a.detalhe:
        n = a.detalhe
        if n not in texto_ref:
            print(f"\nPágina {n} não existe na régua.")
        else:
            print(f"\nDetalhe da página {n} ({tipos.get(n, '')}): diferenças de cada método em relação a {a.ref}")
            pr, nr = palavras(texto_ref[n]), numeros(texto_ref[n])
            for m in metodos:
                t = textos[m].get(n)
                if t is None:
                    print(f"\n  {m}: não rodou esta página")
                    continue
                ph, nh = palavras(t), numeros(t)
                print(f"\n  {m}")
                print(f"    faltam  : {dict((pr - ph).most_common(25))}")
                print(f"    sobram  : {dict((ph - pr).most_common(25))}")
                if nr - nh or nh - nr:
                    print(f"    números que faltam: {dict((nr - nh).most_common(15))}")
                    print(f"    números que sobram: {dict((nh - nr).most_common(15))}")

    # ---- CSV
    if a.csv:
        cols = ["metodo", "pagina", "tipo", "grupo", "palavras_ref", "palavras", "cobertura", "precisao",
                "numeros_ref", "numeros", "segundos", "tokens_entrada", "tokens_saida", "motivo_parada", "sinais"]
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for l in linhas:
                w.writerow({c: (round(l[c], 4) if isinstance(l[c], float) else l[c]) for c in cols})
        print(f"\n-> {a.csv} ({len(linhas)} linhas, só números e rótulos; nenhum texto dos documentos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
