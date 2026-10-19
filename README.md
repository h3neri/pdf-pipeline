# pdf_pipeline — PDFs extensos → texto limpo (com VLM local só onde precisa)

Pipeline em um único script (`pdf_pipeline.py`) com estado em SQLite: cada etapa pode ser
interrompida e retomada, e todo texto fica rastreável até `documento:página:fonte`.

```
triagem → (amostra + gabarito) → vlm → validar / revisar → exportar
```

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate      # Python >= 3.9 (testado em 3.10 e 3.13)
pip install pymupdf httpx rapidfuzz
pip install nvidia-ml-py      # opcional: pausa automática por temperatura da GPU
```

O servidor do modelo (vLLM / llama.cpp / Ollama) roda à parte; o script só fala HTTP com ele.

## Teste em 2 minutos, sem GPU

```bash
python teste_local.py gerar-pdf --saida teste.pdf          # PDF sintético com as armadilhas
python teste_local.py servidor-mock --gabarito teste.gabarito.json --porta 8000 &
python pdf_pipeline.py triagem teste.pdf
python pdf_pipeline.py vlm --url http://localhost:8000/v1 --selecao todas
python pdf_pipeline.py validar --hip vlm:mock-ocr
python pdf_pipeline.py revisar --hip vlm:mock-ocr --somente-sinalizadas   # abre revisao_vlm_mock_ocr.html
python pdf_pipeline.py exportar --saida corpus
```

O servidor falso injeta erros de propósito (HTTP 500, laço de repetição, omissão de parágrafo,
número trocado, caracteres chineses); a validação deve acusar as páginas 12 e 13 e o retry deve
consertar a 3 e a 14. Para testar o checkpoint, mate o `vlm` no meio (Ctrl+C, `kill`, `kill -9`) ou
derrube o servidor e rode o mesmo comando de novo.

## Fluxo nos PDFs reais

1. **Triagem** (segundos/minutos, só CPU): `python pdf_pipeline.py triagem pdfs/` e depois
   `python pdf_pipeline.py status`. Categorias: `ok` (texto nativo bom), `misto` (texto bom + imagem
   grande que pode ter texto), `vlm` (sem camada de texto, texto ilegível ou caracteres inválidos),
   `vazia`. Os limiares são heurísticos — calibre olhando algumas páginas de cada categoria.
2. **Gabarito**: `python pdf_pipeline.py amostra --n 40` gera PNG + TXT de páginas sorteadas por
   estrato; a equipe corrige os TXT seguindo `amostra/LEIA-ME.txt` e importa:
   `python pdf_pipeline.py importar --fonte gabarito --dir amostra`.
3. **Experimento** (escolha do método): rode cada candidato só nas páginas do gabarito e compare.
   ```bash
   python pdf_pipeline.py vlm --paginas amostra/paginas.txt --url http://localhost:8000/v1
   python pdf_pipeline.py validar --hip vlm:<modelo> --ref gabarito
   python pdf_pipeline.py validar --hip nativo --ref gabarito
   ```
4. **Produção**: `python pdf_pipeline.py vlm` (padrão: só páginas `vlm` + `misto`), depois
   `validar --hip vlm:<modelo>` (contra o nativo), `revisar --somente-sinalizadas` e
   `exportar --saida corpus`. A exportação falha (código 1) se alguma página não tiver texto
   aceitável — nada some calado.

## Subindo o modelo

Confira as flags na página/receita do modelo para a versão instalada (mudam com frequência).

**RTX A5500 (24 GB, Ampere)** — vLLM:
```bash
# OCR especializado, 0,9B (Apache 2.0). Use o prompt do próprio modelo:
vllm serve PaddlePaddle/PaddleOCR-VL-1.6 --trust-remote-code \
     --max-num-batched-tokens 16384 --no-enable-prefix-caching --mm-processor-cache-gb 0
python pdf_pipeline.py vlm --prompt "OCR:" --concorrencia 16

# OCR especializado, 0,9B (MIT):
vllm serve zai-org/GLM-OCR --served-model-name glm-ocr --port 8000
python pdf_pipeline.py vlm --prompt "Text Recognition:" --concorrencia 16

# VLM generalista (segue o prompt em português do script): ex. Qwen3.5 pequeno ou Gemma 4 12B
# quantizado. Se o modelo tiver modo de "raciocínio", desligue:
python pdf_pipeline.py vlm --extra-json '{"chat_template_kwargs": {"enable_thinking": false}}'
```
Observação: PaddleOCR-VL e GLM-OCR foram feitos para rodar depois de um detector de layout
(recortes de região). Mandar a página inteira funciona, mas compare com o pipeline oficial deles
(`paddleocr doc_parser` / SDK do GLM-OCR) no experimento — dá para importar a saída deles com
`importar --jsonl` e validar do mesmo jeito.

**Titan V (12 GB, Volta, sm_70)** — o vLLM atual exige compute capability 7.5+ e o PyTorch só
mantém a Volta nos builds CUDA 12.6 (até a versão 2.14). Use llama.cpp (compilado com CUDA 12.x) e GGUF:
```bash
llama-server -m PaddleOCR-VL-1.6.gguf --mmproj PaddleOCR-VL-1.6-mmproj.gguf \
             --temp 0 --parallel 4 -c 32768 --port 8000
python pdf_pipeline.py vlm --url http://localhost:8000/v1 --prompt "OCR:" --concorrencia 4
```

## Parâmetros que mais importam no `vlm`

| opção | padrão | quando mexer |
|---|---|---|
| `--concorrencia` | 8 | Suba até as págs/min pararem de crescer (o log do vLLM mostra o uso do KV cache). |
| `--lado-px` | 1600 | Maior lado da imagem. Ajuste ao modelo (olmOCR usa 1288); letra miúda pede mais. |
| `--max-tokens` | 4096 | Se truncar, o retry dobra até `--max-tokens-teto`. |
| `--ancora` | off | Junta o texto nativo ao prompt como referência (ajuda em números; não use no experimento "VLM puro"). |
| `--temp-max-gpu` | off | Ex.: 83 → pausa os envios acima de 83 °C (precisa de nvidia-ml-py, na máquina da GPU). |
| `--espera-servidor` | 1800 | Quanto tempo esperar o servidor voltar antes de encerrar (o progresso fica salvo). |

## Rodando sem babá

- Use `tmux` (ou `nohup`) para a sessão SSH não matar o processo.
- Limite de potência evita superaquecimento melhor que pausa: `sudo nvidia-smi -pl 180`.
- Rode o servidor do modelo como serviço (`systemd` com `Restart=always` ou Docker com
  `--restart unless-stopped`) e o cliente num laço que só para quando termina ou quando você pede:
  ```bash
  while true; do
    python pdf_pipeline.py vlm --url http://localhost:8000/v1; rc=$?
    [ $rc -eq 0 ] || [ $rc -eq 130 ] && break     # 0 = terminou, 130 = Ctrl+C
    sleep 60                                       # 2 = servidor caiu: espera e retoma
  done
  ```

## Onde fica cada coisa

- `pipeline.sqlite`: tabelas `docs`, `paginas` (triagem + texto nativo), `saidas` (uma linha por
  página × fonte, com status `ok` / `sinalizada` / `erro`, problemas, tentativas e metadados do
  modelo/prompt) e `validacao`. Consultas úteis:
  `sqlite3 pipeline.sqlite "select fonte, status, count(*) from saidas group by 1, 2"`.
- `exportar` grava `<doc>.txt` (um parágrafo ou linha de tabela por linha), `corpus.jsonl` (uma
  página por linha, com documento, página e fonte) e `manifesto.json` (páginas exportadas,
  faltantes, sinalizadas, fontes usadas e a lista do que foi removido como cabeçalho/rodapé —
  confira essa lista).

## Comparar com o Docling (ponto de partida — não testado aqui)

```python
# docling_jsonl.py <pdf> <doc_id> <amostra/paginas.txt> <saida.jsonl>
import json, sys
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pdf, doc_id, lista, saida = sys.argv[1:5]
conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(
    pipeline_options=PdfPipelineOptions(do_ocr=False))})          # PDF nativo: sem OCR
with open(saida, "w", encoding="utf-8") as f:
    for linha in open(lista, encoding="utf-8"):
        d, p = linha.strip().rsplit(":", 1)
        if d == doc_id:
            md = conv.convert(pdf, page_range=(int(p), int(p))).document.export_to_markdown()
            f.write(json.dumps({"doc_id": d, "pagina": int(p), "texto": md}, ensure_ascii=False) + "\n")
```
Depois: `importar --fonte docling --jsonl saida.jsonl` e `validar --hip docling --ref gabarito`.

## Limitações conhecidas

- Ordem de leitura do texto nativo = ordem do conteúdo no PDF (cabeçalho/rodapé reposicionados).
  Em layouts complexos, compare com Docling no experimento.
- Tabelas: só agrupa células na mesma altura em `a | b | c`. Use `--tabelas-para-vlm` na triagem
  para mandar páginas com cara de tabela ao VLM.
- Cabeçalho/rodapé: aprendidos pelas linhas da faixa superior/inferior (`--faixa-borda`, 8% da
  altura) que se repetem; se algum sobreviver, aumente a faixa e rode `triagem --refazer`
  (refaz só a triagem; as saídas do VLM continuam valendo).
- Hifenização: `contra-tação` vira `contratação` só se a forma sem hífen aparecer no documento.
