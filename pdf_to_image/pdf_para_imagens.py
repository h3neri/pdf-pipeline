#!/usr/bin/env python3
"""
pdf_para_imagens.py — converte páginas de PDFs em imagens PNG/JPG com PyMuPDF.

Exemplos:
  python pdf_para_imagens.py doc.pdf                       # todas as páginas, maior lado 1600 px
  python pdf_para_imagens.py pdfs/ --saida imagens         # todos os PDFs de uma pasta
  python pdf_para_imagens.py doc.pdf --paginas 1-10,25     # só algumas páginas (1-based)
  python pdf_para_imagens.py doc.pdf --dpi 200             # por DPI em vez de tamanho fixo
  python pdf_para_imagens.py doc.pdf --processos 4         # paralelo (um processo por bloco)

Se for interrompido, rode de novo: páginas já salvas são puladas.
Dependência: pymupdf (instale com ./setup.sh na raiz do projeto; veja o README)
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf  # em versões antigas: import fitz as pymupdf


def matriz_para(pagina, lado_px=None, dpi=None):
    """Matrix de escala. O PDF mede em pontos (1 pt = 1/72 pol), então zoom 1 = 72 DPI."""
    if dpi:
        zoom = dpi / 72
    else:
        zoom = lado_px / max(pagina.rect.width, pagina.rect.height)
    return pymupdf.Matrix(zoom, zoom)


def interpretar_paginas(texto, total):
    """'1-10,25' -> [0..9, 24] (índices 0-based que o PyMuPDF usa)."""
    if not texto:
        return list(range(total))
    paginas = set()
    for parte in texto.split(","):
        parte = parte.strip()
        if "-" in parte:
            ini, fim = parte.split("-")
            paginas.update(range(int(ini), int(fim) + 1))
        elif parte:
            paginas.add(int(parte))
    return sorted(p - 1 for p in paginas if 1 <= p <= total)


def converter_bloco(caminho_pdf, indices, pasta, args):
    """Converte uma lista de páginas. Roda num processo próprio (cada processo abre o PDF)."""
    feitas = puladas = 0
    with pymupdf.open(caminho_pdf) as doc:
        for i in indices:
            destino = pasta / f"p{i + 1:05d}.{args.formato}"   # p00001.png: ordena certo
            if destino.exists() and not args.sobrescrever:
                puladas += 1
                continue
            pagina = doc.load_page(i)
            pix = pagina.get_pixmap(matrix=matriz_para(pagina, args.lado_px, args.dpi),
                                    alpha=False,                       # fundo branco, sem transparência
                                    colorspace=pymupdf.csGRAY if args.cinza else pymupdf.csRGB)
            tmp = destino.with_suffix(".tmp")
            if args.formato == "png":
                pix.save(tmp, output="png")
            else:
                pix.save(tmp, output="jpg", jpg_quality=args.qualidade)
            tmp.replace(destino)   # grava em .tmp e renomeia: nunca fica imagem pela metade
            feitas += 1
    return feitas, puladas


def converter_pdf(caminho_pdf, args):
    with pymupdf.open(caminho_pdf) as doc:
        if doc.needs_pass:
            print(f"  pulando {caminho_pdf.name}: protegido por senha")
            return
        total = doc.page_count
    indices = interpretar_paginas(args.paginas, total)
    pasta = Path(args.saida) / caminho_pdf.stem
    pasta.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.processos <= 1:
        feitas, puladas = converter_bloco(caminho_pdf, indices, pasta, args)
    else:
        # divide as páginas em blocos contíguos, um por processo
        n = args.processos
        blocos = [indices[k::n] for k in range(n)]
        feitas = puladas = 0
        with ProcessPoolExecutor(max_workers=n) as ex:
            for f, p in ex.map(converter_bloco, [caminho_pdf] * n, blocos, [pasta] * n, [args] * n):
                feitas, puladas = feitas + f, puladas + p
    dt = time.time() - t0
    print(f"  {caminho_pdf.name}: {feitas} novas, {puladas} já existiam, "
          f"{total} no PDF — {dt:.1f}s ({feitas / max(dt, 1e-9):.1f} pág/s) -> {pasta}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entradas", nargs="+", help="PDFs e/ou pastas com PDFs")
    ap.add_argument("--saida", default="imagens")
    ap.add_argument("--lado-px", type=int, default=1600, help="maior lado da imagem em pixels")
    ap.add_argument("--dpi", type=int, help="usa DPI em vez de --lado-px")
    ap.add_argument("--paginas", help="ex.: 1-10,25 (1-based)")
    ap.add_argument("--formato", choices=["png", "jpg"], default="png")
    ap.add_argument("--qualidade", type=int, default=90, help="qualidade do JPG")
    ap.add_argument("--cinza", action="store_true", help="tons de cinza (arquivos ~3x menores)")
    ap.add_argument("--processos", type=int, default=1)
    ap.add_argument("--sobrescrever", action="store_true")
    args = ap.parse_args()

    pdfs = []
    for e in args.entradas:
        p = Path(e)
        pdfs += sorted(p.rglob("*.pdf")) if p.is_dir() else [p]
    for pdf in pdfs:
        converter_pdf(pdf, args)


if __name__ == "__main__":
    sys.exit(main())
