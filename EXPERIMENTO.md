# Experimento: qual modelo transcreve as páginas `vlm`/`misto`

**Pergunta:** qual modelo, com qual configuração, deve transcrever as páginas que o texto nativo não resolve?
A decisão sai de números medidos contra um gabarito corrigido à mão, não de impressão.

**Por que só `vlm` e `misto`:** na produção, o VLM só recebe essas páginas (arquitetura híbrida). Nas `ok`,
o texto nativo já vale; medir o modelo nelas mostra pouco.

## 0. Antes de tudo: o experimento compensa?

Depois da `triagem pdfs/`, rode `status` e some as colunas `vlm` + `misto`.

- **Poucas páginas** (algumas dezenas): a escolha do modelo pesa pouco. Rode o melhor candidato que couber,
  revise tudo com `revisar` e siga em frente.
- **Centenas ou mais:** faça o experimento completo abaixo.

## 1. Gabarito (o padrão-ouro)

```bash
python pdf_pipeline.py amostra --n 40
```

- O sorteio distribui as páginas pelos estratos (categoria × tem-lixo × documento). Com 6 documentos, 40
  páginas dão cerca de 1 a 3 por estrato. Confira no log quantas saíram `vlm`/`misto`. Se forem menos de
  ~15, aumente o `--n` (as páginas já corrigidas nunca são sobrescritas).
- A equipe corrige os `.txt` olhando o `.png`, seguindo `amostra/LEIA-ME.txt`. Antes, duas pessoas fazem as
  mesmas 5 páginas e comparam, para alinhar as regras.
- **Corrija o gabarito antes de rodar qualquer modelo** e sem olhar as saídas deles. Quem vê a saída do modelo
  tende a aceitar o que ele escreveu, e o gabarito passa a favorecer esse modelo.
- Importe: `python pdf_pipeline.py importar --fonte gabarito --dir amostra`

## 1.5 Teste de fumaça (enquanto a equipe corrige o gabarito)

Antes de gastar horas de GPU, confira se o servidor, os flags e o cliente funcionam com PDFs reais. Use a
E4B (a mais leve) em ~10 páginas `vlm`/`misto` que **não** estão no gabarito, numa fonte só para isso:

```bash
# janela 0 do tmux: o vllm serve da E4B (seção 3)
# janela 1:
python pdf_pipeline.py amostra --n 12 --saida fumaca --excluir amostra/paginas.txt   # sorteio fora do gabarito
python pdf_pipeline.py vlm --paginas fumaca/paginas.txt --fonte vlm:fumaca-e4b
python pdf_pipeline.py mostrar --fonte vlm:fumaca-e4b --n 12
python pdf_pipeline.py validar --hip vlm:fumaca-e4b --ref nativo   # nas páginas ok/misto, o nativo já serve de régua
```

- **Por que fora do gabarito:** quem corrige uma página não deve ter visto a saída do modelo nela.
- **Por que outra fonte:** `vlm:fumaca-e4b` não se mistura com `vlm:gemma4-e4b` do experimento.
- **O que conferir:** nenhum papo, cerca ```` ``` ```` ou tag de raciocínio no texto (se a checagem `meta_texto`
  pegar, a página aparece `sinalizada`); números iguais aos da imagem; tabelas com ` | `; a velocidade (`pág/min`
  no log). Se algo sistemático aparecer, ajuste o `pos_processar` ou o prompt antes do experimento.
- Para ver uma página lado a lado com a imagem: `python pdf_pipeline.py revisar --hip vlm:fumaca-e4b --ref nativo`.

## 2. Candidatos

**Rodada 1: modelos.** Todos em condições iguais: prompt do próprio modelo, temperatura 0, `--lado-px 1600`,
sem `--ancora`, mesma `--concorrencia` (8). Concorrência igual deixa os tempos comparáveis.

| fonte (nome no banco) | checkpoint | por que testar |
|---|---|---|
| `vlm:gemma4-e4b` | `google/gemma-4-E4B-it` (BF16) | menor e mais rápido; cabe com folga |
| `vlm:gemma4-12b` | `google/gemma-4-12B-it-qat-w4a16-ct` | meio-termo; precisa do vLLM nightly |
| `vlm:gemma4-31b` | `google/gemma-4-31B-it-qat-w4a16-ct` | teto de qualidade; apertado em 24 GB |
| `vlm:paddleocr-vl` | `PaddlePaddle/PaddleOCR-VL-1.6` | OCR especializado, referência |
| `vlm:glm-ocr` | `zai-org/GLM-OCR` | OCR especializado, referência |
| `nativo` | texto do PyMuPDF | mostra quanto se ganha usando VLM |

- O **26B A4B** fica fora: não há checkpoint de 4 bits oficial, e o int8 da receita (~26 GB) não cabe.
- Se o **31B** não subir nem com `--max-model-len 8192`, ele sai do experimento. Isso já é um resultado.

**Rodada 2: ajustes**, só nos 1 ou 2 melhores da rodada 1. Mude uma coisa de cada vez e dê a cada
configuração seu próprio `--fonte`, senão o checkpoint pula as páginas como "já feitas".

| variação | fonte de exemplo | o que responde |
|---|---|---|
| `max_soft_tokens` 560 em vez de 1120 | `vlm:gemma4-12b-560` | perde letra miúda? quanto ganha de velocidade? |
| `--lado-px 1920` e `2339` (200 DPI) em vez de 1600 | `vlm:gemma4-12b-1920` | com 1120 tokens a Gemma vê ~1344×1920 px: a 1600 ela amplia a imagem; a 2339 ela reduz. Desenhar já em 1920 deixa a letra mais nítida? |
| `--ancora` | `vlm:gemma4-12b-ancora` | o texto nativo ajuda nos números ou faz o modelo copiar os erros dele? |
| outro prompt (`--prompt-arquivo`) | `vlm:gemma4-12b-p2` | só se a rodada 1 mostrar um erro sistemático (ex.: tabelas) |

## 3. Como rodar um candidato

Use duas janelas do `tmux`: uma para o servidor, outra para o cliente (veja "Rodando por SSH" no README).

```bash
# janela 0: servidor, no ambiente do vLLM
vllm serve google/gemma-4-E4B-it --served-model-name gemma4-e4b --host 127.0.0.1 --port 8000 \
     --max-model-len 16384 --gpu-memory-utilization 0.90 \
     --limit-mm-per-prompt '{"image": 1}' \
     --mm-processor-kwargs '{"max_soft_tokens": 1120}' \
     --reasoning-parser gemma4

# janela 1: cliente, no .venv do pipeline (a fonte vira vlm:gemma4-e4b)
python pdf_pipeline.py vlm --paginas amostra/paginas.txt --concorrencia 8
python pdf_pipeline.py validar --hip vlm:gemma4-e4b --ref gabarito
```

- `--served-model-name` define o nome curto que vira a fonte no banco.
- **Na primeira vez com cada modelo**, veja uma saída antes de seguir:
  `python pdf_pipeline.py mostrar --fonte vlm:gemma4-e4b --n 1`.
  Se aparecer alguma tag de raciocínio ou um preâmbulo ("Aqui está a transcrição..."), pare e ajuste o
  `pos_processar` antes de continuar.
- Para os modelos de OCR, troque o checkpoint e o nome e passe o prompt deles no cliente
  (`--prompt "OCR:"` ou `--prompt "Text Recognition:"`). Os comandos estão no README.
- Anote a versão do vLLM (`vllm --version`) e os flags usados. Outra versão pode mudar o resultado.

## 4. Comparar e decidir

```bash
python pdf_pipeline.py validar --hip nativo --ref gabarito             # linha de base
mkdir -p experimento
python pdf_pipeline.py comparar --saida experimento/rodada1.csv        # só vlm+misto (padrão)
python pdf_pipeline.py comparar --categorias todas                     # visão geral
```

O `comparar` lê o que cada `validar` gravou e ordena as fontes pela precisão p5. Critérios, nesta ordem:

1. **Eliminatório:** laços de repetição, caracteres chineses, papo do modelo (`meta_texto`) ou páginas truncadas
   que as tentativas extras não resolvem (coluna `sinal.` e o detalhe no CSV do `validar`).
2. **Precisão p5:** a precisão nas piores páginas. Texto inventado entra direto na whitelist e libera respostas
   fora do domínio. É o erro mais caro.
3. **Recall p5:** o que o modelo deixa de fora nas piores páginas. Vocabulário perdido é cobertura perdida.
4. **Números errados:** trocas de valor, como R$ 500,00 → 5.000,00.
5. **Velocidade** (`s/pág`): só desempata. Com 2.322 páginas, mesmo processando todas a 20 s cada, seriam ~13 h.

**Empate técnico:** com ~40 páginas, uma página sozinha mexe na mediana e no p5. Diferenças abaixo de ~0,01
são ruído; nesse caso, fique com o modelo menor ou mais rápido.

Antes de fechar, olhe os erros dos 2 primeiros colocados:

```bash
python pdf_pipeline.py revisar --hip vlm:gemma4-12b --ref gabarito
```

## 5. Registro

Guarde em `experimento/` (só métricas, sem texto dos documentos) o CSV de cada rodada e um
`experimento/resultados.md` com: data, modelo, checkpoint, versão do vLLM, flags, a tabela do `comparar` e a
decisão com o motivo. Isso serve de base para o relatório da IC e para refazer o teste se algo mudar.
