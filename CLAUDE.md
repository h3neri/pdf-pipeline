# pdf_pipeline — PDFs públicos (Bradesco + Bacen) → texto limpo

## Contexto
- Iniciação Científica (Poli-USP) em Adversarial AI com o Bradesco, para aprimorar o chatbot **Bia**.
- Objetivo desta parte: transformar PDFs públicos em `.txt` limpo. Esse texto vira a base de uma **whitelist**
  que restringe as respostas do chatbot ao domínio do banco. O material do Bacen entra para enriquecer o
  dataset com termos técnicos de finanças.
- Documentos atuais (6 PDFs, **2.322 páginas, 674,2 MB** no total): `documentos_assets_1.pdf`,
  `documentos_assets_2.pdf`, `documentos_assets_3.pdf`, `documentos_bacen.pdf`, `glossario_cidadania.pdf` e
  `regulamentos.pdf`. Os PDFs antigos (`Bradesco_Assets.pdf`, `Bradesco_Seguros.pdf`) saíram do projeto.
  Uns 290 KB por página é bastante, o que pode indicar muitas imagens ou páginas escaneadas: a triagem dirá.
- A tentativa anterior (PDF → CSV) quebrava a formatação e mantinha lixo, como os `xxxxx` de campos de assinatura.
- Quem mantém o projeto é o Henry, estudante de Engenharia de Computação que está aprendendo Python e PyMuPDF.
  Fale em português e explique o porquê de cada mudança de forma didática.

## Arquivos
- `pdf_pipeline.py`: o pipeline inteiro, com os subcomandos `triagem`, `amostra`, `vlm`, `importar`,
  `validar`, `comparar`, `revisar`, `exportar` e `status`. Todo o estado fica em `pipeline.sqlite`.
- `teste_local.py`: `gerar-pdf` cria um PDF sintético de 14 páginas com armadilhas, mais o gabarito.
  `servidor-mock` é um servidor falso, compatível com OpenAI, que injeta erros de propósito.
- `pdf_to_image/pdf_para_imagens.py`: conversor independente de PDF para PNG/JPG, usado para inspeção.
- `README.md`: instalação, fluxo completo, como subir os modelos, SSH, parâmetros e limitações.
- `EXPERIMENTO.md`: plano do teste que escolhe o modelo (candidatos, gabarito, critérios de decisão).

## Decisões de arquitetura (não mudar sem discutir)
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
   - checagens sem referência: saída truncada, vazia, com caracteres chineses ou em laço de repetição;
   - gabarito de ~40 páginas estratificadas, corrigidas à mão (`amostra` → `importar --fonte gabarito`);
   - `comparar` junta numa tabela a validação de todas as fontes, ordenada pela precisão p5.
6. **`exportar` falha com código 1** se alguma página ficar sem texto aceitável, a menos que se use `--permitir-faltantes`.

## Situação atual
- **Feito e testado sem GPU**, com o `servidor-mock`: o pipeline inteiro, inclusive a retomada após
  SIGTERM, SIGKILL e queda do servidor, e o `comparar`.
- **Ainda não testado:** PDFs reais, Gemma 4 e os outros modelos reais, prompts por modelo, pausa por
  temperatura da GPU e o script de comparação com o Docling que está no README.
- **Próximos passos** (detalhes em `EXPERIMENTO.md`):
  1. Preparar a A5500: `setup.sh`, um ambiente separado para o vLLM e copiar os PDFs para `pdfs/`.
  2. `triagem pdfs/` e depois `status`, para saber quantas páginas são `vlm` ou `misto`.
  3. `amostra --n 40`, corrigir o gabarito e importar.
  4. Rodar os candidatos nas páginas do gabarito, `validar` cada um e decidir com `comparar`.
  5. Rodar a produção com o vencedor e depois `exportar`.

## Hardware e acesso
- **NVIDIA RTX A5500, 24 GB** (Ampere, sm_86, tem BF16), numa máquina Ubuntu. O vLLM roda direto.
- Acesso por **SSH pelo VS Code** (Remote-SSH). Processos longos (`vllm serve`, `vlm`) rodam sempre dentro do
  `tmux`, porque os terminais do VS Code morrem quando a conexão cai.
- O vLLM fica num ambiente virtual **separado** do `.venv` do pipeline (ele fixa versões próprias de PyTorch);
  os dois só conversam por HTTP. Suba-o com `--host 127.0.0.1`: o cliente roda na mesma máquina.
- `sudo nvidia-smi -pl` (limite de potência) é opcional; só use se a GPU passar muito tempo acima de ~85 °C.

## Modelos
- **Gemma 4 (principal).** Tamanho e quantização são escolhidos no experimento. Na A5500:
  - `google/gemma-4-E4B-it` em BF16 — cabe com folga;
  - `google/gemma-4-12B-it-qat-w4a16-ct` (4 bits, treinado com quantização) — precisa do vLLM nightly ou da
    imagem Docker `vllm/vllm-openai:gemma4-unified` (receita do vLLM, nov/2026);
  - `google/gemma-4-31B-it-qat-w4a16-ct` — uns 18 GB só de pesos, pouca sobra para KV cache; pode não subir.
  - Fora: **26B A4B**, que não tem checkpoint W4A16 oficial; o int8 da receita (~26 GB) não cabe.
  - Usa o prompt padrão em português do script.
  - **Imagem:** a Gemma reduz cada imagem a um orçamento de tokens visuais (70, 140, 280, 560 ou 1120; padrão
    280). Para OCR use 1120: `--mm-processor-kwargs '{"max_soft_tokens": 1120}'` no `vllm serve`. Por isso o
    `--lado-px` não define a resolução final que o modelo enxerga.
  - **Raciocínio:** vem desligado por padrão no vLLM; não ligue. Suba com `--reasoning-parser gemma4` para que,
    se algum pensamento vier, ele vá para o campo `reasoning` e não para o texto. O `pos_processar` só remove
    `<think>...</think>`: confira as primeiras saídas reais e, se sobrar tag de raciocínio, ajuste a regex.
- **Referências do experimento** (OCR especializado, 0,9B; rodam na A5500 com vLLM):
  - **PaddleOCR-VL-1.6** (Apache 2.0): `--prompt "OCR:"`.
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
  ```
  Resultado esperado: no `vlm`, as tentativas extras corrigem a 3 e a 14 (`sinalizadas=0`). Na validação,
  a 12 aparece com `omissao`, a 13 com o número trocado (500,00 → 5.000,00) e a 8 (`misto`) com `adicao`,
  o que é normal ali, porque o VLM lê o texto da imagem.
- **PyMuPDF:** índices de página começam em 0. Renderize com processos (não threads) e abra o PDF dentro
  de cada processo.
- Faça commits pequenos, com mensagem em português.
