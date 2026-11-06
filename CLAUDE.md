# pdf_pipeline — PDFs públicos (Bradesco + Bacen) → texto limpo

## Contexto
- Iniciação Científica (Poli-USP) em Adversarial AI com o Bradesco, para aprimorar o chatbot **Bia**.
- Objetivo desta parte: transformar PDFs públicos em `.txt` limpo. Esse texto vira a base de uma **whitelist**
  que restringe as respostas do chatbot ao domínio do banco. O material do Bacen entra para enriquecer o
  dataset com termos técnicos de finanças.
- Documentos atuais (6 PDFs em `pdf_to_image/`, **2.322 páginas, 196 MB** no total): `documentos assets 1.pdf`,
  `documentos assets 2.pdf`, `documentos assets 3.pdf` (slides 16:9), `documentos bacen.pdf`,
  `glossario_cidadania_financeira.pdf` e `regulamentos.pdf`. No banco, os `doc_id` ficam sem espaço:
  `documentos_assets_1`, ..., `glossario_cidadania_financeira`, `regulamentos`. Os PDFs antigos
  (`Bradesco_Assets.pdf`, `Bradesco_Seguros.pdf`) saíram do projeto. Os de "assets" e o `regulamentos` são
  vários PDFs juntados (pdf-lib) e repetem páginas entre si: ~290 páginas são cópias idênticas de outras.
- A tentativa anterior (PDF → CSV) quebrava a formatação e mantinha lixo, como os `xxxxx` de campos de assinatura.
- Quem mantém o projeto é o Henry, estudante de Engenharia de Computação que está aprendendo Python e PyMuPDF.
  Fale em português e explique o porquê de cada mudança de forma didática.

## Arquivos
- `pdf_pipeline.py`: o pipeline inteiro, com os subcomandos `triagem`, `amostra`, `vlm`, `importar`,
  `validar`, `comparar`, `revisar`, `exportar`, `status` e `mostrar`. Todo o estado fica em `pipeline.sqlite`.
- `teste_local.py`: `gerar-pdf` cria um PDF sintético de 14 páginas com armadilhas, mais o gabarito.
  `servidor-mock` é um servidor falso, compatível com OpenAI, que injeta erros de propósito.
- `pdf_to_image/pdf_para_imagens.py`: conversor independente de PDF para PNG/JPG, usado para inspeção.
- `README.md`: instalação, fluxo completo, como subir os modelos, SSH, parâmetros e limitações.
- `EXPERIMENTO.md`: plano do teste que escolhe o modelo (candidatos, gabarito, critérios de decisão).
- `teste_ocr/`: **fase atual**, independente do pipeline (veja abaixo e `teste_ocr/LEIA-ME.md`).
- `experimento/resultados_teste_ocr.md`: resultados, análise e conclusões do teste bruto (5–6/nov/2026), com
  `teste_ocr_metricas.csv` (só números) e `teste_ocr_mapa.png`.

## Fase atual: teste bruto imagem → txt (desde 5/nov/2026)
- Decisão do Henry: **não usar o `pdf_pipeline.py` nem a abordagem híbrida agora**. Primeiro medir a qualidade
  bruta da transcrição, comparando vários métodos na mesma amostra: extração direta do PDF, Gemma 4 lendo
  a imagem e PaddleOCR-VL-1.6. Depois, possivelmente, reorganizar o pipeline com uma abordagem múltipla
  (o tempo de GPU não importa; importa o melhor resultado).
- **Sem limpeza nesta fase:** tabelas, URLs, cabeçalhos e números de página são mantidos; só elementos sem
  texto (ícones, fotos) podem ser ignorados. A limpeza é decidida depois, com base nos resultados.
- `teste_ocr/montar_pdf_teste.py` monta `amostra_teste.pdf` (47 páginas, 15 tipos, sem repetidas, semente
  2026; `manifesto.csv` liga cada página à origem). As imagens saem do nosso `pdf_para_imagens.py`
  (`--lado-px 1920`). `teste_ocr/transcrever.py` (`nativo`, `vlm`, `resumo`) grava um txt por página em
  `teste_ocr/saidas/<modelo>__<prompt>/`, sem alterar a resposta; os sinais vão só para o log.
- Prompts em `teste_ocr/prompts/`: `gemma_fiel.txt` (transcrição fiel, tabelas com ` | `, `[SEM TEXTO]`
  para página vazia) e `paddleocr_ocr.txt` (`OCR:`). Servidor: vLLM na A5500; começar pela E4B, depois o
  maior Gemma 4 que couber.
- Testado sem GPU com servidor falso: retomada após Ctrl+C, `kill -9` e queda do servidor.
- `teste_ocr/comparar.py` compara as saídas com o nativo (cobertura, precisão e números por página, em três
  grupos: `completo`, `imagem`, `defeito`) e confere `teste_ocr/armadilhas.csv` (14 trechos verificados na
  imagem ampliada, onde algum método errou). Só biblioteca padrão; roda em qualquer máquina com `saidas/`.
- **Rodadas feitas (5–6/nov, vLLM 0.33.0 na A5500):** E4B, 12B e PaddleOCR-VL-1.6 nas 47 páginas; 31B só em
  7 páginas dirigidas (limite de hardware). Rodada completa do 31B (40 páginas restantes) pendente: o comando
  está no LEIA-ME, e a pasta é `saidas/gemma4-31b-ctx2700__gemma_fiel`. Resultados em `experimento/resultados_teste_ocr.md`.
- **Conclusões até aqui:**
  - Nenhum método domina. Os Gemma erram de forma fluente e invisível: trocam palavras por outras plausíveis
    ("excetuadas"→"executadas"), normalizam nomes e números ("RenovAgro"→"RenovaAgro", "250,000"→"250.000")
    e "corrigem" o original. A 12B inventou um aviso ilegível. O Paddle é literal no impresso nítido, mas
    erra acentos e letras, perde a estrutura das tabelas e não lê prints de tela com a página inteira.
  - O tamanho reduz os erros fluentes. Nas 14 armadilhas: E4B 0, 12B 6, Paddle 11 e 31B 9 (+4 depois do
    corte de contexto).
  - A 12B entrou em laço em 3 páginas com pontuação repetida (tracejado, `____`, sumário pontilhado), o
    Paddle em 1 e a E4B em nenhuma.
  - Todas as armadilhas estão em páginas com camada de texto, onde o nativo acerta: isso reforça o híbrido.
    Os modelos só acrescentam em texto dentro de imagem (~17 páginas reais na produção).
  - Votar entre modelos da mesma família não funciona: E4B e 12B erram as mesmas palavras.

## Decisões de arquitetura do pipeline (não mudar sem discutir)
1. **Híbrido.** O texto nativo do PyMuPDF é usado onde ele é bom. O VLM só recebe as páginas que a triagem
   marca como `vlm` (sem camada de texto, texto ilegível) ou `misto` (imagem grande). Isso evita
   alucinação e custo de GPU nas páginas nativas. Com um generalista como a Gemma, que tende a "corrigir"
   o texto, isso importa ainda mais.
2. **Limpeza determinística em código, nunca pelo VLM:** preenchimentos (`xxxx`, `____`), pontilhados de
   sumário, máscaras (`xxx.xxx.xxx-xx`), cabeçalho/rodapé repetido (aprendido por documento) e
   hifenização (só junta se a palavra sem hífen existir no documento).
3. **Checkpoint em SQLite (WAL).** Cada página é gravada assim que fica pronta. Para retomar, rode o mesmo
   comando de novo. O sha256 liga cada saída à versão do PDF. O checkpoint é por (página, fonte): para testar
   outra configuração do mesmo modelo, use outro `--fonte`, senão as páginas são puladas como "já feitas".
4. **Cliente VLM assíncrono** (httpx): uma imagem por requisição, API compatível com OpenAI (vLLM, llama.cpp
   ou Ollama). Quando o servidor cai, as tentativas da página não são gastas: o programa espera até
   `--espera-servidor` e então sai com código 2. Ctrl+C sai com código 130.
5. **Validação:**
   - recall/precisão de tokens e CER/WER;
   - comparação dos números da página, que pega trocas como R$ 500,00 → 5.000,00;
   - checagens sem referência: saída truncada, vazia, com caracteres chineses, em laço de repetição ou com
     "papo" do modelo (`meta_texto`: cerca ```` ``` ```` que sobrou, "Como um modelo de linguagem...", "texto
     extraído"; só no começo e no fim da resposta). Elas valem também nas páginas `vlm`, que não têm referência.
     Um falso positivo só custa outra tentativa e deixa a página sinalizada; não se apaga o papo automaticamente;
   - gabarito de ~40 páginas estratificadas, corrigidas à mão (`amostra` → `importar --fonte gabarito`);
   - `comparar` junta numa tabela a validação de todas as fontes, ordenada pela precisão p5.
6. **`exportar` falha com código 1** se alguma página ficar sem texto aceitável, a menos que se use `--permitir-faltantes`.

## Situação atual
- **Feito e testado sem GPU**, com o `servidor-mock`: o pipeline inteiro, inclusive a retomada após
  SIGTERM, SIGKILL e queda do servidor, e o `comparar`.
- **Primeiro contato com a Gemma 4 E4B (nov/2026)**, pelo `transformers` num chat interativo (`chat.py`, fora
  do repositório), numa página com texto nativo: 3 palavras trocadas em ~500 (ex.: "diligência nas" →
  "diligência às"), números certos. Com um pedido em tom de conversa, a resposta veio com papo ("Como um modelo
  de linguagem...") e cerca ```` ```txt ````, e o pipeline aceitava isso como `ok`: daí a checagem `meta_texto`.
  Esse chat serve só para explorar; os números do experimento saem do vLLM + `vlm` (prompt, temperatura 0 e
  `max_soft_tokens` controlados).
- **Triagem dos PDFs reais (5/nov/2026, 21 s, só CPU):** `ok` 2.126, `misto` 188, `vlm` 2, `vazia` 6.
  Nenhuma página com codificação quebrada. Medição fora do pipeline (OCR tesseract; contagens em
  `experimento/texto_em_imagem.csv`): das 188 `misto`, 127 só têm fundo decorativo ou foto, 48 têm umas
  poucas palavras (logotipo, título) e 13 têm texto de verdade na imagem (prints de sistema, telas de app);
  mais 3 `ok` do `documentos_assets_2` têm texto em imagem pequena. Das 2 `vlm`, só a capa da cartilha
  (bacen:33) precisa do modelo. Cabeçalho/rodapé aprendidos e lixo detectado conferidos: estão certos.
- **Texto nativo:** em 42 páginas `ok` sorteadas (17.940 palavras), um OCR que só vê os pixels concorda com o
  nativo em 98,8% das palavras, e nas diferenças o erro de letra foi sempre do OCR. Os defeitos do nativo são
  de espaço, não de caractere, e têm conserto determinístico: palavras grudadas (~33 págs. do Bacen; o
  espaço entre palavras é 0,137 × o tamanho da fonte e o MuPDF não o vê), ligadura partida ("fi nanceira",
  ~42 págs.) e letras espaçadas em títulos ("I m p o r t a n t e", ~35 págs.).
- **Atenção:** o `exportar` usa o texto do VLM no lugar do nativo em toda página `misto`; nas 127 sem texto na
  imagem, isso troca texto exato por uma releitura. O teste bruto confirmou o risco: os modelos erram
  justamente onde o nativo acerta. **Proposta:** manter o nativo como base e acrescentar só o texto da imagem
  (linhas do modelo que não existem no nativo), e só nas páginas com texto real em imagem.
- **Testado em 5–6/nov (teste bruto, fora do pipeline):** Gemma 4 E4B/12B/31B e PaddleOCR-VL-1.6 pelo vLLM
  na A5500. **Ainda não testado:** o subcomando `vlm` do pipeline com modelo real, prompts alternativos,
  pausa por temperatura da GPU, pipeline oficial do Paddle (layout + recortes) e o script do Docling.
- **Próximos passos:**
  1. Terminar a rodada completa do 31B, rodar `teste_ocr/comparar.py` e atualizar
     `experimento/resultados_teste_ocr.md` (seções 5, 7 e 10).
  2. Decidir e implementar a regra de mescla das páginas `misto` (acima).
  3. Escolher o modelo para as ~17 páginas com texto em imagem: o 31B, se a rodada completa não mostrar laços
     nem invenção; senão, a E4B. Marcar para revisão humana as páginas onde Gemma e Paddle discordam.
  4. Gabarito humano focado nessas ~17 páginas, e não em 40 estratificadas (seção 0 do `EXPERIMENTO.md`:
     com poucas páginas, o experimento de 5 modelos não compensa).
  5. Testar o pipeline oficial do Paddle (`paddleocr doc_parser`) nas páginas de prints de tela.
  6. Produção: nativo + consertos determinísticos de espaço; modelo só nas páginas escolhidas; `exportar`.

## Hardware e acesso
- **NVIDIA RTX A5500, 24 GB** (Ampere, sm_86, tem BF16; 24.564 MiB no `nvidia-smi`), na máquina Ubuntu
  **compartilhada "chineque"** (usuário `g38885519865`, **sem sudo**). Driver 580, CUDA 13.0. Em 06/11 a GPU
  estava livre (1 MiB usado), mas é compartilhada: pare o servidor (Ctrl+C) ao terminar.
- **Porta 8000 ocupada** por outro serviço da máquina: use `--port 8011` no `vllm serve` e, no cliente,
  sempre `--url http://127.0.0.1:8011/v1 --modelo <nome>` (o padrão do `transcrever.py` é a 8000; com
  `--modelo`, um servidor errado responde 404 em vez de transcrever com o modelo de outra pessoa).
- Clone do repositório na chineque: `~/teste/pdf-pipeline` (com `.venv`). vLLM **0.33.0** em `~/venv-vllm`.
- **Sem CUDA toolkit (`nvcc`)**: a FlashInfer não compila kernels, e o amostrador top-k/top-p dela derruba a
  partida em `determine_available_memory` → `_dummy_sampler_run` ("Could not find nvcc"). Solução: prefixe o
  `vllm serve` com `VLLM_USE_FLASHINFER_SAMPLER=0` (com temperatura 0 o top-k/top-p nem é usado). Alternativa
  não testada: `~/venv-vllm/bin/vllm download-kernels`. `gcc` e `Python.h` existem (o Triton funciona).
- "Engine core initialization failed" é só a mensagem final do processo da API. A causa está no traceback do
  `EngineCore`: `grep -n "EngineCore" log | grep -v " INFO " | tail -n 40`. Rode sempre com `2>&1 | tee log`
  e com o comando **numa linha só** (o `\` de várias linhas quebra ao colar).
- Acesso por **SSH pelo VS Code** (Remote-SSH). Processos longos (`vllm serve`, `vlm`) rodam sempre dentro do
  `tmux`, porque os terminais do VS Code morrem quando a conexão cai.
- O vLLM fica num ambiente virtual **separado** do `.venv` do pipeline (ele fixa versões próprias de PyTorch);
  os dois só conversam por HTTP. Suba-o com `--host 127.0.0.1`: o cliente roda na mesma máquina.
- `sudo nvidia-smi -pl` (limite de potência) não está disponível na chineque (sem sudo); use
  `--temp-max-gpu` no `vlm` se a GPU passar muito tempo acima de ~85 °C.

## Modelos
- **Gemma 4 (principal).** Na A5500 (medido em 06/11, vLLM 0.33.0; comandos em `teste_ocr/LEIA-ME.md`):
  - `google/gemma-4-E4B-it` em BF16: 4,5B "efetivos", mas **8B com embeddings → 14,5 GiB de pesos**. Sobe com
    `--gpu-memory-utilization 0.90`, `--max-num-seqs 4`, `--max-num-batched-tokens 4096`, `--enforce-eager` e
    `--limit-mm-per-prompt '{"image": 1, "audio": 0}'` (o `"audio": 0` evita alocar o codificador de áudio).
    Sobram 6,2 GiB de KV cache (~38 KB/token). Não "cabe com folga".
  - `google/gemma-4-12B-it-qat-w4a16-ct` (4 bits, QAT): **subiu no vLLM 0.33.0 estável** (não precisa de
    nightly). 8,3 GiB de pesos, 12,3 GiB de KV cache (~196 KB/token). Ocupa menos que a E4B em BF16.
  - `google/gemma-4-31B-it-qat-w4a16-ct`: arquivo de **23,3 GB**; ocupa ~19 GiB na GPU e o KV cache custa
    ~860 KB/token. `--cpu-offload-gb` **não reduz** a memória da GPU com pesos W4A16 (19,0 GiB com 6 GB fora,
    19,2 com 10). Contexto máximo ≈ 2.700 tokens (`--gpu-memory-utilization 0.95 --max-model-len 2700
    --max-num-seqs 1`; cliente `--max-tokens 1300 --concorrencia 1`): as páginas com resposta longa são cortadas.
  - Fora: **26B A4B**, que não tem checkpoint W4A16 oficial; o int8 da receita (~26 GB) não cabe.
  - Com a imagem em 1120 tokens, o prompt fica em ~1,3 mil tokens. A maior resposta legítima da amostra teve
    2.297 tokens (tabela densa); passar de ~3 mil é quase sempre laço de repetição.
  - Usa o prompt padrão em português do script.
  - **Imagem:** a Gemma reduz cada imagem a um orçamento de tokens visuais (70, 140, 280, 560 ou 1120; padrão
    280). Para OCR use 1120: `--mm-processor-kwargs '{"max_soft_tokens": 1120}'` no `vllm serve`. Por isso o
    `--lado-px` não define a resolução final que o modelo enxerga.
  - **Raciocínio:** vem desligado por padrão no vLLM; não ligue. Suba com `--reasoning-parser gemma4` para que,
    se algum pensamento vier, ele vá para o campo `reasoning` e não para o texto. O `pos_processar` só remove
    `<think>...</think>`: confira as primeiras saídas reais e, se sobrar tag de raciocínio, ajuste a regex.
- **Referências do experimento** (OCR especializado, 0,9B; rodam na A5500 com vLLM):
  - **PaddleOCR-VL-1.6** (Apache 2.0): `--prompt "OCR:"`. 1,8 GiB de pesos, ~5 s por página. Literal no
    impresso, mas erra acentos e não lê prints de tela com a página inteira (reduz a ~1 MP).
  - **GLM-OCR** (MIT): `--prompt "Text Recognition:"`.
  - Eles foram feitos para receber recortes de um detector de layout. Mandar a página inteira funciona, mas
    dá para comparar com o pipeline oficial (`paddleocr doc_parser ...`) trazendo a saída com `importar --jsonl`.
- **Evitar:**
  - Llama 3.2 Vision e olmOCR 2, que são focados em inglês;
  - MinerU, que exige licença comercial para empresas grandes;
  - Marker, cujos pesos têm uso restrito.

## Convenções
- Código, nomes, mensagens e comentários em **português**. Python 3.9 ou mais novo. Dependências:
  `pymupdf httpx rapidfuzz` (`nvidia-ml-py` é opcional e vale instalar na A5500).
- **Nunca versionar dados:** PDFs, imagens, `amostra/`, `corpus/`, `*.sqlite`, `*.jsonl`, HTML de revisão
  (veja `.gitignore`). O repositório é privado.
- Rode o teste sem GPU antes e depois de qualquer mudança no pipeline:
  ```bash
  python teste_local.py gerar-pdf --saida teste.pdf
  python teste_local.py servidor-mock --gabarito teste.gabarito.json --porta 8000 &
  python pdf_pipeline.py --db teste.sqlite triagem teste.pdf
  python pdf_pipeline.py --db teste.sqlite vlm --url http://localhost:8000/v1 --selecao todas
  python pdf_pipeline.py --db teste.sqlite validar --hip vlm:mock-ocr --ref nativo
  python pdf_pipeline.py --db teste.sqlite mostrar --fonte vlm:mock-ocr --pagina teste:9
  ```
  Resultado esperado: no `vlm`, as tentativas extras corrigem a 3, a 9 (papo + cerca, `meta_texto`) e a 14
  (`sinalizadas=0`; o `mostrar` da 9 dá `tentativas=2` ou mais, e texto limpo). Na validação,
  a 12 aparece com `omissao`, a 13 com o número trocado (500,00 → 5.000,00) e a 8 (`misto`) com `adicao`,
  o que é normal ali, porque o VLM lê o texto da imagem.
- **PyMuPDF:** índices de página começam em 0. Renderize com processos (não threads) e abra o PDF dentro
  de cada processo.
- Faça commits pequenos, com mensagem em português.
