# Teste bruto: imagem → txt

Primeira fase: medir a qualidade da transcrição **sem limpeza nenhuma** e **sem o `pdf_pipeline.py`**.
Tudo o que tem texto é mantido (tabelas, URLs, cabeçalhos, números de página); só ícones e fotos sem
texto podem ser ignorados. A limpeza vem depois, com base no que este teste mostrar.

Métodos comparados, cada um numa pasta de `saidas/`:

| pasta | método |
|---|---|
| `nativo/` | extração direta: o texto que já está dentro do PDF (PyMuPDF, na ordem do arquivo) |
| `nativo_ordenado/` | o mesmo, reordenado de cima para baixo |
| `gemma4-e4b__gemma_fiel/` (e 12b, 31b) | Gemma 4 lendo a imagem, com `prompts/gemma_fiel.txt` |
| `paddleocr-vl-1.6__paddleocr_ocr/` | PaddleOCR-VL-1.6 lendo a imagem, com o prompt do próprio modelo (`OCR:`) |

## Arquivos

- `amostra_teste.pdf`: 47 páginas de 15 tipos (não vai para o git: `*.pdf`). Para recriar igual:
  `python teste_ocr/montar_pdf_teste.py pdf_to_image/`
- `manifesto.csv`: página do teste → documento, página original e tipo.
- `transcrever.py`: o programa do teste (`nativo`, `vlm`, `resumo`).
- `prompts/`: um arquivo por prompt. Para testar outro prompt, crie outro arquivo; o nome dele vira
  parte do nome da pasta de saída.
- `saidas/`: os txt (uma página por arquivo: `p00001.txt` é a página 1 do `amostra_teste.pdf`),
  `config.json` (como foi rodado) e `execucao.jsonl` (tempo, tokens e sinais de cada página).
  **Não vai para o git**: tem o texto dos documentos.

## Roteiro na A5500 (todos os comandos na pasta `pipeline/`)

**0. Levar o PDF de teste.** Ele não está no git. Do seu computador:
```bash
rsync -avP "teste_ocr/amostra_teste.pdf" usuario@maquina:~/pdf-pipeline/teste_ocr/
```

**1. Imagens, com o nosso conversor.** Use 1920 px: com `max_soft_tokens` 1120, a Gemma 4 enxerga até
~1344×1920 px; a 1600 px ela teria de ampliar a imagem. As mesmas imagens servem para todos os modelos.
```bash
source .venv/bin/activate
python pdf_to_image/pdf_para_imagens.py teste_ocr/amostra_teste.pdf --saida teste_ocr/imagens --lado-px 1920
python teste_ocr/transcrever.py nativo
python teste_ocr/transcrever.py nativo --ordenar
```

**2. Gemma 4 E4B** (a mais leve, para validar tudo). Duas janelas do `tmux` (`tmux new -s ocr`):
```bash
# janela 0: servidor, no ambiente do vLLM
~/venv-vllm/bin/vllm serve google/gemma-4-E4B-it --served-model-name gemma4-e4b --host 127.0.0.1 --port 8000 \
     --max-model-len 16384 --gpu-memory-utilization 0.90 \
     --limit-mm-per-prompt '{"image": 1}' \
     --mm-processor-kwargs '{"max_soft_tokens": 1120}' \
     --reasoning-parser gemma4

# janela 1: cliente, no .venv do pipeline. Primeiro 3 páginas, para olhar o resultado:
python teste_ocr/transcrever.py vlm --prompt teste_ocr/prompts/gemma_fiel.txt --paginas 1-3 \
       --obs "E4B BF16, max_soft_tokens 1120, max-model-len 16384"
cat teste_ocr/saidas/gemma4-e4b__gemma_fiel/p00001.txt
# se estiver tudo bem, o resto (as 3 já feitas são puladas):
python teste_ocr/transcrever.py vlm --prompt teste_ocr/prompts/gemma_fiel.txt \
       --obs "E4B BF16, max_soft_tokens 1120, max-model-len 16384"
```

**3. Gemma 4 maior.** Pare o servidor (Ctrl+C na janela 0) e suba o 12B (precisa do vLLM nightly) ou o 31B,
trocando só o checkpoint e o `--served-model-name` (veja "Subindo o modelo" no README). O cliente é o
mesmo comando; a pasta muda sozinha (`gemma4-12b__gemma_fiel`). Se o 31B não subir, tente
`--max-model-len 12288`; se não subir de jeito nenhum, isso já é um resultado.

**4. PaddleOCR-VL-1.6.**
```bash
# janela 0
~/venv-vllm/bin/vllm serve PaddlePaddle/PaddleOCR-VL-1.6 --served-model-name paddleocr-vl-1.6 \
     --host 127.0.0.1 --port 8000 --trust-remote-code \
     --max-num-batched-tokens 16384 --no-enable-prefix-caching --mm-processor-cache-gb 0
# janela 1
python teste_ocr/transcrever.py vlm --prompt teste_ocr/prompts/paddleocr_ocr.txt --obs "página inteira, prompt OCR:"
```
Ele foi treinado para receber recortes de um detector de layout e, na tarefa `OCR:`, reduz a imagem a
~1 megapixel. Página inteira funciona, mas letra miúda pode sofrer: é uma das coisas que o teste mede.

**5. Resumo.**
```bash
python teste_ocr/transcrever.py resumo
```

## Durante a execução

- **Interrupção:** Ctrl+C, queda da conexão SSH (fora do `tmux`) ou servidor caído. Rode o mesmo comando
  de novo: as páginas que já têm txt são puladas. Servidor fora do ar → o programa para com código 2.
- **Página `truncado`:** a resposta bateu no `--max-tokens` (8192). Refaça só ela com um limite maior:
  `--paginas 16 --refazer --max-tokens 16384` (o limite usado fica no log da página).
- **Mudou prompt, modelo, temperatura ou imagens?** Use outro `--metodo` (ou outro nome de arquivo de
  prompt). O programa recusa misturar configurações na mesma pasta.
- Os **sinais** (`truncado`, `vazio`, `sem_texto`, `cerca_markdown`, `cjk`, `repeticao`) só marcam a
  página no log. O txt fica exatamente como o modelo respondeu.
- Anote a versão do vLLM (`vllm --version`; ela também vai para o `config.json`) e os flags do
  `vllm serve` no `--obs`.
