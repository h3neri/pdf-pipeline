# Teste bruto: imagem → txt

Primeira fase: medir a qualidade da transcrição **sem limpeza nenhuma** e **sem o `pdf_pipeline.py`**.
Tudo o que tem texto é mantido (tabelas, URLs, cabeçalhos, números de página). Só ícones e fotos sem
texto podem ser ignorados. A limpeza vem depois, com base no que este teste mostrar.

**Resultados e análise (5–6/nov/2026):** [`../experimento/resultados_teste_ocr.md`](../experimento/resultados_teste_ocr.md).

Métodos comparados, cada um numa pasta de `saidas/`:

| pasta | método |
|---|---|
| `nativo/` | extração direta: o texto que já está dentro do PDF (PyMuPDF, na ordem do arquivo) |
| `nativo_ordenado/` | o mesmo, reordenado de cima para baixo |
| `gemma4-e4b__gemma_fiel/`, `gemma4-12b__gemma_fiel/` | Gemma 4 lendo a imagem, com `prompts/gemma_fiel.txt` |
| `gemma4-31b-ctx2700__gemma_fiel/` | Gemma 4 31B com contexto limitado a 2.700 tokens (limite da A5500) |
| `paddleocr-vl-1.6__paddleocr_ocr/` | PaddleOCR-VL-1.6 lendo a imagem, com o prompt do próprio modelo (`OCR:`) |

## Arquivos

- `amostra_teste.pdf`: 47 páginas de 15 tipos. Não vai para o git (`*.pdf`). Para recriar igual:
  `python teste_ocr/montar_pdf_teste.py pdf_to_image/`
- `manifesto.csv`: página do teste → documento, página original e tipo.
- `transcrever.py`: o programa do teste (`nativo`, `vlm`, `resumo`).
- `comparar.py`: compara cada método com o nativo e confere as armadilhas (veja o passo 6).
- `armadilhas.csv`: trechos conferidos na imagem ampliada, em lugares onde algum método errou.
- `prompts/`: um arquivo por prompt. Para testar outro prompt, crie outro arquivo: o nome dele vira
  parte do nome da pasta de saída.
- `saidas/`: os txt, um por página (`p00001.txt` é a página 1 do `amostra_teste.pdf`), mais
  `config.json` (como foi rodado) e `execucao.jsonl` (tempo, tokens e sinais de cada página).
  **Não vai para o git**, porque tem o texto dos documentos.

## Roteiro na A5500 (máquina "chineque")

Todos os comandos rodam na pasta do clone, `~/teste/pdf-pipeline`. Fatos da máquina que os comandos
levam em conta (detalhes no `CLAUDE.md`):
- a máquina é compartilhada e sem sudo;
- a **porta 8000 está ocupada**, então usamos a **8011**;
- não há `nvcc`, por isso o `VLLM_USE_FLASHINFER_SAMPLER=0` antes de cada `vllm serve`.

Use o **tmux**, para que os processos sobrevivam a uma queda do VS Code:
- `tmux new -s ocr` cria a sessão, ou `tmux attach -t ocr` volta para ela;
- a janela 0 fica com o servidor;
- `Ctrl+b c` cria a janela 1, para o cliente;
- `Ctrl+b 0` e `Ctrl+b 1` trocam de janela;
- `Ctrl+b d` sai sem matar nada.

Cole cada comando **numa linha só**: comandos com `\` quebram ao colar. Quando aparecer
`Application startup complete`, o servidor está pronto. Para conferir os números da partida:
`grep -E "loading took|Available KV cache memory|Maximum concurrency" ~/vllm-<modelo>.log`.

**0. Levar o PDF de teste.** Ele não está no git. Do seu computador:
```bash
rsync -avP "teste_ocr/amostra_teste.pdf" g38885519865@chineque:~/teste/pdf-pipeline/teste_ocr/
```

**1. Imagens e nativo.** 1920 px, porque com `max_soft_tokens` 1120 a Gemma 4 enxerga até ~1344×1920 px.
As mesmas imagens servem para todos os modelos.
```bash
source .venv/bin/activate
python pdf_to_image/pdf_para_imagens.py teste_ocr/amostra_teste.pdf --saida teste_ocr/imagens --lado-px 1920
python teste_ocr/transcrever.py nativo
python teste_ocr/transcrever.py nativo --ordenar
```

**2. Gemma 4 E4B** (teste de fumaça; ~6 min para as 47 páginas).
```bash
# janela 0: servidor
VLLM_USE_FLASHINFER_SAMPLER=0 ~/venv-vllm/bin/vllm serve google/gemma-4-E4B-it --served-model-name gemma4-e4b --host 127.0.0.1 --port 8011 --gpu-memory-utilization 0.90 --max-model-len 16384 --max-num-seqs 4 --max-num-batched-tokens 4096 --enforce-eager --limit-mm-per-prompt '{"image": 1, "audio": 0}' --mm-processor-kwargs '{"max_soft_tokens": 1120}' --reasoning-parser gemma4 2>&1 | tee ~/vllm-e4b.log

# janela 1: cliente. Primeiro 3 páginas, para olhar o resultado; depois o mesmo comando sem --paginas.
source .venv/bin/activate
python teste_ocr/transcrever.py vlm --url http://127.0.0.1:8011/v1 --modelo gemma4-e4b --prompt teste_ocr/prompts/gemma_fiel.txt --paginas 1-3 --timeout 1800 --obs "E4B BF16, max_soft_tokens 1120, max-model-len 16384, util 0.90, eager, max-num-seqs 4, batched 4096, audio 0, flashinfer sampler off"
```
`--url` e `--modelo` são **obrigatórios na chineque**. O padrão do script é a porta 8000, que é o
serviço de outra pessoa. Com `--modelo`, um servidor errado responde 404 em vez de transcrever com
outro modelo.

**3. Gemma 4 12B** (4 bits; ~25 min, contando 3 páginas em laço).
O comando do servidor é o do passo 2, trocando só o checkpoint, o nome e o log:
`google/gemma-4-12B-it-qat-w4a16-ct`, `--served-model-name gemma4-12b` e `~/vllm-12b.log`.
No cliente, use `--modelo gemma4-12b` e o `--obs` correspondente.

**4. Gemma 4 31B** (4 bits). Não cabe com contexto útil: o arquivo tem 23,3 GB e o `--cpu-offload-gb`
não libera a GPU com esses pesos. O teto é de ~2.700 tokens de contexto e uma página por vez, e cada
página leva ~2 min.
```bash
# janela 0
VLLM_USE_FLASHINFER_SAMPLER=0 ~/venv-vllm/bin/vllm serve google/gemma-4-31B-it-qat-w4a16-ct --served-model-name gemma4-31b --host 127.0.0.1 --port 8011 --gpu-memory-utilization 0.95 --cpu-offload-gb 10 --max-model-len 2700 --max-num-seqs 1 --max-num-batched-tokens 4096 --enforce-eager --limit-mm-per-prompt '{"image": 1}' --mm-processor-kwargs '{"max_soft_tokens": 1120}' --reasoning-parser gemma4 2>&1 | tee ~/vllm-31b.log

# janela 1 (o --metodo deixa claro que é uma rodada com contexto reduzido)
python teste_ocr/transcrever.py vlm --url http://127.0.0.1:8011/v1 --modelo gemma4-31b --metodo gemma4-31b-ctx2700__gemma_fiel --prompt teste_ocr/prompts/gemma_fiel.txt --concorrencia 1 --max-tokens 1300 --timeout 3600 --obs "31B QAT W4A16, contexto 2700 (limite de hardware), max_tokens 1300, max_soft_tokens 1120, util 0.95, offload 10 (sem efeito), eager, flashinfer sampler off"
```

**5. PaddleOCR-VL-1.6** (~4 min).
```bash
# janela 0
VLLM_USE_FLASHINFER_SAMPLER=0 ~/venv-vllm/bin/vllm serve PaddlePaddle/PaddleOCR-VL-1.6 --served-model-name paddleocr-vl-1.6 --host 127.0.0.1 --port 8011 --trust-remote-code --max-num-batched-tokens 16384 --no-enable-prefix-caching --mm-processor-cache-gb 0 --enforce-eager --gpu-memory-utilization 0.90 2>&1 | tee ~/vllm-paddle.log

# janela 1
python teste_ocr/transcrever.py vlm --url http://127.0.0.1:8011/v1 --modelo paddleocr-vl-1.6 --prompt teste_ocr/prompts/paddleocr_ocr.txt --timeout 1800 --obs "PaddleOCR-VL-1.6, pagina inteira, prompt OCR:, eager, flashinfer sampler off"
```
O Paddle foi treinado para receber recortes de um detector de layout e, na tarefa `OCR:`, reduz a
imagem a ~1 megapixel. Com a página inteira, o impresso miúdo sai bem, mas os prints de tela não.

**6. Resumo e comparação** (pode rodar no seu computador, depois de copiar `saidas/`):
```bash
python teste_ocr/transcrever.py resumo
python teste_ocr/comparar.py
python teste_ocr/comparar.py --detalhe 13
python teste_ocr/comparar.py --csv experimento/teste_ocr_metricas.csv
```
O `--detalhe` mostra as palavras que faltam ou sobram numa página, por método. O `--csv` grava só
números, então pode ir para o git.

Para trazer as saídas para o seu computador, use o VS Code (clique direito em `teste_ocr/saidas`,
depois em Download) ou rode do seu computador:
`rsync -avP g38885519865@chineque:~/teste/pdf-pipeline/teste_ocr/saidas/ teste_ocr/saidas/`

Ao terminar, **pare o servidor** (Ctrl+C na janela 0): a GPU é compartilhada.

## Durante a execução

- **Interrupção:** Ctrl+C, queda da conexão SSH fora do `tmux` ou servidor caído. Rode o mesmo comando
  de novo: as páginas que já têm txt são puladas. Com o servidor fora do ar, o programa para com
  código 2.
- **Cliente "parado" por vários minutos:** quase sempre é uma página em laço de repetição. A janela
  do servidor mostra `Running: 1 reqs` com geração acima de zero. Espere: ela termina em
  `max_tokens` com o sinal `truncado`.
  - **Não refaça páginas em laço.** Com temperatura 0, o laço se repete igual, e ele é um resultado
    do teste.
  - **Use `--timeout 1800`.** No padrão de 900 s, o cliente desiste de um laço lento e tenta de novo,
    até 4 vezes.
  - Nesta amostra, os laços apareceram em sequências longas de pontuação: linha tracejada, campos
    `____` e sumário pontilhado.
- **Página `truncado` legítima:** a maior resposta legítima da amostra teve 2.297 tokens, então
  bater em 8.192 é laço. Se um dia uma página longa for cortada, refaça só ela com um limite maior:
  `--paginas 16 --refazer --max-tokens 16384`. O limite usado fica no log da página.
- **Mudou prompt, modelo, temperatura ou imagens?** Use outro `--metodo` (ou outro nome de arquivo de
  prompt). O programa recusa misturar configurações na mesma pasta.
  - Mas a trava **não vê os flags do servidor** (`max_soft_tokens`, contexto). Mantenha um comando de
    servidor por modelo e copie esses flags no `--obs`.
  - Para conferir: `tokens_entrada` no `execucao.jsonl` fica em ~1,3 mil com `max_soft_tokens` 1120.
- **Sinais** (`truncado`, `vazio`, `sem_texto`, `cerca_markdown`, `cjk`, `repeticao`): só marcam a página
  no log, e o txt fica exatamente como o modelo respondeu. O `repeticao` tem falso positivo em
  formulários com linhas legitimamente iguais (8 linhas de data na p3).
- **O servidor não sobe:** "Engine core initialization failed" é só a mensagem final. A causa está no
  traceback do `EngineCore`:
  `grep -n "EngineCore" ~/vllm-<modelo>.log | grep -v " INFO " | tail -n 40`.
  Causas que já apareceram:
  - **`Could not find nvcc`:** faltou o `VLLM_USE_FLASHINFER_SAMPLER=0`.
  - **`max seq len ... KV cache is needed`:** o contexto não cabe. Reduza o `--max-model-len` e o
    `--max-tokens` do cliente na mesma proporção.
- Anote a versão do vLLM (`vllm --version`, que também vai para o `config.json`) e os flags do
  `vllm serve` no `--obs`.
