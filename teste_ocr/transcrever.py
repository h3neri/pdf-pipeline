#!/usr/bin/env python3
"""
transcrever.py — teste bruto da transcrição imagem -> txt (independente do pdf_pipeline.py).

Três subcomandos:
  nativo   extração direta: o texto que já está dentro do PDF (PyMuPDF), uma página por txt
  vlm      manda cada imagem a um servidor compatível com OpenAI (vLLM) com um prompt e grava a resposta
  resumo   conta, por método, páginas feitas, sinais de problema, tempo e tokens

Nada é limpo nem corrigido: o txt é a resposta do modelo como veio. Os "sinais" (truncado, cerca de
Markdown, caracteres chineses, repetição...) só são anotados no log, para a análise depois.

Fluxo (na pasta pipeline/, veja teste_ocr/LEIA-ME.md):
  python pdf_to_image/pdf_para_imagens.py teste_ocr/amostra_teste.pdf --saida teste_ocr/imagens --lado-px 1920
  python teste_ocr/transcrever.py nativo
  python teste_ocr/transcrever.py vlm --prompt teste_ocr/prompts/gemma_fiel.txt
  python teste_ocr/transcrever.py resumo

Saída: teste_ocr/saidas/<metodo>/p00001.txt ... + config.json (como foi rodado) + execucao.jsonl (uma
linha por página: tempo, tokens, motivo de parada, sinais). Rodar de novo continua de onde parou: páginas
que já têm txt são puladas (use --refazer para sobrescrever).

Códigos de saída: 0 terminou, 1 alguma página falhou, 2 servidor fora do ar, 130 Ctrl+C.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

AQUI = Path(__file__).resolve().parent
VERSAO = "1.0"
RE_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
RE_IMAGEM = re.compile(r"^p(\d+)\.(png|jpg|jpeg)$", re.I)


# ----------------------------------------------------------------------------- utilidades

def agora() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def gravar_atomico(caminho: Path, texto: str) -> None:
    """Grava num arquivo temporário e renomeia: se o processo morrer no meio, não sobra txt pela metade."""
    tmp = caminho.with_suffix(caminho.suffix + ".tmp")
    tmp.write_text(texto, encoding="utf-8")
    os.replace(tmp, caminho)


def nome_seguro(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "modelo"


def faixa_paginas(spec: Optional[str]) -> Optional[set]:
    """'1-5,12' -> {1,2,3,4,5,12}"""
    if not spec:
        return None
    out = set()
    for parte in spec.split(","):
        parte = parte.strip()
        if "-" in parte:
            a, b = parte.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif parte:
            out.add(int(parte))
    return out


def sinais(texto: str, motivo_parada: Optional[str]) -> list:
    """Anota problemas comuns de VLM. Não muda o texto: só marca para a análise."""
    s = []
    t = texto.strip()
    if not t:
        s.append("vazio")
    if t == "[SEM TEXTO]":
        s.append("sem_texto")
    if motivo_parada == "length":
        s.append("truncado")
    if "```" in texto:
        s.append("cerca_markdown")
    if RE_CJK.search(texto):
        s.append("cjk")
    linhas = [l.strip() for l in texto.splitlines() if len(l.strip()) >= 20]
    if linhas and Counter(linhas).most_common(1)[0][1] >= 5:
        s.append("repeticao")
    return s


# ----------------------------------------------------------------------------- nativo

def cmd_nativo(a) -> int:
    import pymupdf
    pdf = Path(a.pdf)
    saida = Path(a.saidas) / ("nativo_ordenado" if a.ordenar else "nativo")
    saida.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf)
    for i, page in enumerate(doc, 1):
        # sort=False: ordem em que o texto aparece no PDF (o "bruto"); sort=True: de cima para baixo
        gravar_atomico(saida / f"p{i:05d}.txt", page.get_text(sort=a.ordenar))
    (saida / "config.json").write_text(json.dumps({
        "metodo": saida.name, "pdf": str(pdf), "paginas": doc.page_count, "pymupdf": pymupdf.__version__,
        "get_text_sort": a.ordenar, "data": agora(), "programa": f"transcrever.py {VERSAO}",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{doc.page_count} páginas -> {saida}/")
    return 0


# ----------------------------------------------------------------------------- vlm

class ServidorForaDoAr(Exception):
    pass


async def descobrir_modelo(cliente, url: str) -> str:
    r = await cliente.get(f"{url}/models", timeout=30)
    r.raise_for_status()
    modelos = [m["id"] for m in r.json().get("data", [])]
    if not modelos:
        raise RuntimeError("o servidor não lista nenhum modelo em /models")
    if len(modelos) > 1:
        print(f"AVISO: o servidor tem {len(modelos)} modelos {modelos}; usando {modelos[0]} (escolha com --modelo)")
    return modelos[0]


async def versao_servidor(cliente, url: str) -> Optional[str]:
    raiz = url[:-3] if url.endswith("/v1") else url        # o vLLM responde /version na raiz
    try:
        r = await cliente.get(f"{raiz}/version", timeout=10)
        return r.json().get("version") if r.status_code == 200 else None
    except Exception:
        return None


async def transcrever_pagina(cliente, a, modelo: str, prompt: str, img: Path) -> dict:
    """Uma requisição por imagem. Tenta de novo em erro de rede/servidor; nunca altera a resposta."""
    import httpx
    mime = "image/png" if img.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(img.read_bytes()).decode()
    corpo = {
        "model": modelo,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},   # imagem antes do texto
            {"type": "text", "text": prompt},
        ]}],
        "temperature": a.temperatura,
        "max_tokens": a.max_tokens,
        "seed": 0,
    }
    ultimo_erro, falhas_conexao = None, 0
    for tentativa in range(1, a.tentativas + 1):
        t0 = time.monotonic()
        try:
            r = await cliente.post(f"{a.url}/chat/completions", json=corpo, timeout=a.timeout)
            if r.status_code >= 500 or r.status_code == 429:
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}: {r.text[:200]}", request=r.request, response=r)
            if r.status_code >= 400:          # erro do pedido (ex.: imagem grande demais): não adianta repetir
                return {"erro": f"HTTP {r.status_code}: {r.text[:500]}", "tentativas": tentativa}
            j = r.json()
            esc = j["choices"][0]
            msg = esc.get("message") or {}
            texto = msg.get("content") or ""
            raciocinio = msg.get("reasoning_content") or msg.get("reasoning") or ""
            uso = j.get("usage") or {}
            return {
                "texto": texto, "tentativas": tentativa, "segundos": round(time.monotonic() - t0, 2),
                "motivo_parada": esc.get("finish_reason"), "tokens_entrada": uso.get("prompt_tokens"),
                "tokens_saida": uso.get("completion_tokens"), "chars_raciocinio": len(raciocinio),
            }
        except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
            falhas_conexao += 1
            ultimo_erro = f"{type(e).__name__}: {e}"
        except (httpx.TimeoutException, httpx.HTTPStatusError, KeyError, ValueError) as e:
            ultimo_erro = f"{type(e).__name__}: {e}"
        if tentativa < a.tentativas:
            await asyncio.sleep(min(60, 2 ** tentativa))
    if falhas_conexao == a.tentativas:
        raise ServidorForaDoAr(ultimo_erro)
    return {"erro": ultimo_erro, "tentativas": a.tentativas}


async def rodar_vlm(a) -> int:
    import httpx
    pasta_img = Path(a.imagens)
    imagens = {}
    for f in sorted(pasta_img.iterdir()) if pasta_img.is_dir() else []:
        m = RE_IMAGEM.match(f.name)
        if m:
            imagens[int(m.group(1))] = f
    if not imagens:
        print(f"Nenhuma imagem pNNNNN.png em {pasta_img}. Gere com: python pdf_to_image/pdf_para_imagens.py "
              f"teste_ocr/amostra_teste.pdf --saida teste_ocr/imagens --lado-px 1920")
        return 1
    filtro = faixa_paginas(a.paginas)
    prompt_arq = Path(a.prompt)
    prompt = prompt_arq.read_text(encoding="utf-8").strip()

    async with httpx.AsyncClient(headers={"Authorization": "Bearer EMPTY"}) as cliente:
        try:
            modelo = a.modelo or await descobrir_modelo(cliente, a.url)
        except httpx.HTTPError as e:
            print(f"Não consegui falar com o servidor em {a.url} ({type(e).__name__}). Ele está no ar?")
            return 2
        metodo = a.metodo or f"{nome_seguro(modelo)}__{prompt_arq.stem}"
        saida = Path(a.saidas) / metodo
        saida.mkdir(parents=True, exist_ok=True)

        # config.json: tudo o que define a execução. Se mudar algo, use outro --metodo, senão as páginas
        # antigas (feitas com a configuração anterior) se misturam com as novas.
        config = {
            "metodo": metodo, "modelo": modelo, "url": a.url, "versao_servidor": await versao_servidor(cliente, a.url),
            "prompt_arquivo": str(prompt_arq), "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "temperatura": a.temperatura, "max_tokens": a.max_tokens, "imagens": str(pasta_img),
            "programa": f"transcrever.py {VERSAO}", "observacao": a.obs,
        }
        arq_cfg = saida / "config.json"
        if arq_cfg.exists():
            antigo = json.loads(arq_cfg.read_text(encoding="utf-8"))
            so_algumas = a.refazer and filtro is not None
            chaves = ["modelo", "prompt_sha256", "temperatura", "max_tokens", "imagens"]
            if so_algumas:
                # refazer páginas truncadas com limite maior é seguro: com temperatura 0, uma página que
                # terminou antes do limite sairia igual. O limite usado fica no log de cada página.
                chaves.remove("max_tokens")
                config["max_tokens"] = antigo.get("max_tokens", a.max_tokens)
            dif = [k for k in chaves if antigo.get(k) != config[k]]
            if dif and (not a.refazer or so_algumas):   # refazer só algumas misturaria configurações
                print(f"ERRO: {saida} já foi feita com outra configuração ({', '.join(dif)}). "
                      f"Use outro --metodo, ou --refazer sem --paginas para refazer todas.")
                return 1
            config["data_inicio"] = antigo.get("data_inicio", agora())
        else:
            config["data_inicio"] = agora()
        gravar_atomico(arq_cfg, json.dumps(config, ensure_ascii=False, indent=2))

        pendentes = [n for n in sorted(imagens) if (filtro is None or n in filtro)
                     and (a.refazer or not (saida / f"p{n:05d}.txt").exists())]
        print(f"método {metodo}: {len(imagens)} imagens, {len(pendentes)} a fazer "
              f"(modelo {modelo}, servidor {config['versao_servidor'] or '?'})")
        if not pendentes:
            return 0

        log = open(saida / "execucao.jsonl", "a", encoding="utf-8")
        sem = asyncio.Semaphore(a.concorrencia)
        feitas, falhas, t_ini = 0, 0, time.monotonic()
        fora_do_ar = asyncio.Event()

        async def uma(n: int):
            nonlocal feitas, falhas
            async with sem:
                if fora_do_ar.is_set():
                    return
                try:
                    res = await transcrever_pagina(cliente, a, modelo, prompt, imagens[n])
                except ServidorForaDoAr as e:
                    if not fora_do_ar.is_set():        # várias requisições caem juntas: avisa uma vez
                        fora_do_ar.set()
                        print(f"\nServidor fora do ar ({e}). O progresso está salvo: rode o mesmo comando de novo.")
                    return
                reg = {"pagina": n, "imagem": imagens[n].name, "data": agora(), "max_tokens": a.max_tokens,
                       **{k: v for k, v in res.items() if k != "texto"}}
                if "texto" in res:
                    gravar_atomico(saida / f"p{n:05d}.txt", res["texto"])
                    reg["sinais"] = sinais(res["texto"], res.get("motivo_parada"))
                    reg["palavras"] = len(res["texto"].split())
                    feitas += 1
                else:
                    falhas += 1
                log.write(json.dumps(reg, ensure_ascii=False) + "\n")
                log.flush()
                el = time.monotonic() - t_ini
                marca = f" [{', '.join(reg['sinais'])}]" if reg.get("sinais") else ("  ERRO: " + reg["erro"][:120] if "erro" in reg else "")
                print(f"  p{n:05d}  {res.get('segundos', 0):6.1f}s  {reg.get('palavras', 0):5d} palavras"
                      f"{marca}   ({feitas + falhas}/{len(pendentes)}, {60 * (feitas + falhas) / el:.1f} pág/min)")

        try:
            await asyncio.gather(*(uma(n) for n in pendentes))
        finally:
            log.close()
    print(f"\nfeitas {feitas}, com erro {falhas}. Veja: python teste_ocr/transcrever.py resumo")
    if fora_do_ar.is_set():
        return 2
    return 1 if falhas else 0


def cmd_vlm(a) -> int:
    try:
        return asyncio.run(rodar_vlm(a))
    except KeyboardInterrupt:
        print("\nInterrompido. As páginas já gravadas ficam; rode o mesmo comando para continuar.")
        return 130


# ----------------------------------------------------------------------------- resumo

def cmd_resumo(a) -> int:
    raiz = Path(a.saidas)
    pastas = sorted(p for p in raiz.iterdir() if p.is_dir()) if raiz.is_dir() else []
    if not pastas:
        print(f"Nada em {raiz}/ ainda.")
        return 0
    linhas = []
    for p in pastas:
        txts = sorted(p.glob("p*.txt"))
        ultimos = {}
        if (p / "execucao.jsonl").exists():
            for l in open(p / "execucao.jsonl", encoding="utf-8"):
                r = json.loads(l)
                ultimos[r["pagina"]] = r                      # vale a última tentativa de cada página
        cont = Counter(s for r in ultimos.values() for s in r.get("sinais", []))
        seg = [r["segundos"] for r in ultimos.values() if "segundos" in r]
        palavras = sum(len(t.read_text(encoding="utf-8").split()) for t in txts)
        linhas.append({
            "metodo": p.name, "paginas": len(txts), "erros": sum(1 for r in ultimos.values() if "erro" in r),
            "palavras": palavras, "s_mediana": round(statistics.median(seg), 1) if seg else "",
            "s_max": round(max(seg), 1) if seg else "",
            "tokens_saida": sum(r.get("tokens_saida") or 0 for r in ultimos.values()),
            **{s: cont.get(s, 0) for s in ("truncado", "vazio", "sem_texto", "cerca_markdown", "cjk", "repeticao")},
        })
    cols = list(linhas[0])
    larg = {c: max(len(c), *(len(str(l[c])) for l in linhas)) for c in cols}
    print("  ".join(c.ljust(larg[c]) for c in cols))
    for l in linhas:
        print("  ".join(str(l[c]).ljust(larg[c]) for c in cols))
    with open(raiz / "resumo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(linhas)
    print(f"\n-> {raiz / 'resumo.csv'}   (páginas com sinal: veja execucao.jsonl de cada método)")
    return 0


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saidas", default=str(AQUI / "saidas"), help="pasta das saídas (padrão: teste_ocr/saidas)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("nativo", help="extração direta do texto que está dentro do PDF")
    n.add_argument("--pdf", default=str(AQUI / "amostra_teste.pdf"))
    n.add_argument("--ordenar", action="store_true",
                   help="ordena o texto de cima para baixo (grava em saidas/nativo_ordenado)")
    n.set_defaults(func=cmd_nativo)

    v = sub.add_parser("vlm", help="manda as imagens a um servidor compatível com OpenAI (vLLM)")
    v.add_argument("--prompt", required=True, help="arquivo com o prompt (ex.: teste_ocr/prompts/gemma_fiel.txt)")
    v.add_argument("--imagens", default=str(AQUI / "imagens" / "amostra_teste"))
    v.add_argument("--url", default="http://127.0.0.1:8000/v1")
    v.add_argument("--modelo", help="nome do modelo no servidor (padrão: o que o servidor listar)")
    v.add_argument("--metodo", help="nome da pasta de saída (padrão: <modelo>__<nome do prompt>)")
    v.add_argument("--paginas", help="só estas páginas do teste, ex.: 1-5,12")
    v.add_argument("--refazer", action="store_true", help="sobrescreve páginas que já têm txt")
    v.add_argument("--concorrencia", type=int, default=4, help="requisições ao mesmo tempo (padrão: %(default)s)")
    v.add_argument("--max-tokens", type=int, default=8192, help="limite da resposta (padrão: %(default)s)")
    v.add_argument("--temperatura", type=float, default=0.0)
    v.add_argument("--timeout", type=float, default=900, help="segundos por requisição (padrão: %(default)s)")
    v.add_argument("--tentativas", type=int, default=4, help="por página, em erro de rede/servidor")
    v.add_argument("--obs", default="", help="anotação livre gravada no config.json (ex.: flags do vllm serve)")
    v.set_defaults(func=cmd_vlm)

    r = sub.add_parser("resumo", help="tabela com páginas, sinais, tempo e tokens de cada método")
    r.set_defaults(func=cmd_resumo)

    a = ap.parse_args()
    if getattr(a, "url", None):
        a.url = a.url.rstrip("/")
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
