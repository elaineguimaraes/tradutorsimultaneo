# Tradutor Simultâneo 🎧 EN → BR

<!-- screenshot: adicionar imagem da interface e da legenda -->

Tradução simultânea, em tempo real, de **qualquer áudio do seu PC** (lives do
YouTube, vídeos, podcasts) para **voz falada em português do Brasil**, com
uma **legenda flutuante** que fica por cima de qualquer janela.

**A motivação:** o YouTube já traduz e legenda vídeos gravados automaticamente,
mas **transmissões ao vivo ficam descobertas**. Quem acompanha lives em inglês
não tem legenda nem dublagem disponível. É o caso, por exemplo, das lives de
trading, com o mercado aberto e sem tempo para pausar e ler. Este projeto nasceu para
preencher essa lacuna: ele ouve o que está tocando no PC e devolve voz e
legenda em português, com poucos segundos de atraso. Custo zero: a
transcrição e a tradução rodam localmente na CPU, sem nuvem paga; a voz usa o
Edge-TTS da Microsoft (gratuito).

Em destaque: o produto foi **otimizado para o jargão do mercado financeiro em
transmissões voltadas para trading** (day trade, análise técnica, notícias de
mercado). Funciona com qualquer conteúdo em inglês, mas o glossário e as
regras de correção foram afinados nesse domínio: mais de **1.200 regras**
(≈100 termos protegidos que ficam em inglês, 22 tickers expandidos para o
nome da empresa, ≈590 traduções fixas de expressões e ≈510 correções
aplicadas depois da tradução) fazem "earnings" virar "lucros/balanço",
"high/low" virar "topo/fundo", "short/long" virar "venda/compra", "stop"
continuar "stop", tickers como AAPL virarem "Apple" na fala, e assim por
diante.

## English summary

Tradutor Simultâneo is a Windows, CPU-only application that translates any
audio playing on the PC into spoken Brazilian Portuguese in real time, with a
floating on-screen subtitle. The pipeline is fully local except for the
neural voice: faster-whisper (speech recognition) → Opus-MT tc-big en→pt via
CTranslate2 (translation) → Edge-TTS (speech synthesis, free, requires
internet). It was originally built and tuned for English-language trading
livestreams (day-trading commentary, technical analysis, market news), which
is why the glossary and post-translation fix-up rules are especially strong
in that domain, though the app works with any English audio source. All
documentation is in Brazilian Portuguese; this paragraph is here for
international readers and recruiters browsing the repository.

## Como funciona

```
apps → CABLE Input → LoopbackCapture (48 kHz) ─┬─ passthrough → OutputMixer (ducking) → alto-falantes
                                                └─ 16 kHz mono → SpeechSegmenter (Silero VAD)
                                                     → Transcriber (faster-whisper base, int8, CPU)
                                                     → Glossary (máscara de jargão)
                                                     → Translator (Opus-MT tc-big en→pt-BR via CTranslate2)
                                                     → Glossary (restaura + corrige)  → legenda (overlay)
                                                     → TtsSpeaker (Edge-TTS pt-BR, velocidade adaptativa)
                                                     → OutputMixer.enqueue_tts()
```

- Latência típica de ponta a ponta: 3-5 segundos entre a fala original e a
  voz traduzida.
- A voz acelera sozinha (+10% / +25% / +40%) conforme o atraso acumulado
  cresce, para tentar alcançar o "ao vivo"; acima de um limite de atraso, o
  app descarta áudio antigo e pula direto para o presente (a legenda mantém
  o texto de tudo que foi dito).
- O áudio original continua audível ao fundo, em volume reduzido (ducking)
  enquanto a voz traduzida fala.
- Sem internet, a síntese de voz cai automaticamente para a voz offline do
  Windows (SAPI). A legenda continua funcionando normalmente.
- A legenda aparece na tela cerca de 1-2 segundos antes da voz começar a
  falar aquele trecho.

## Requisitos

| Item | Detalhe |
|---|---|
| Sistema operacional | Windows 10/11 (usa captura WASAPI loopback e a voz offline SAPI, exclusivos do Windows) |
| Python | 3.13 (as versões das dependências estão fixadas para ela; outra versão do Python pode não instalar) |
| CPU | Sem necessidade de GPU; recomendado 6-8 núcleos (referência: um Ryzen 7 5700U traduz em tempo real com folga) |
| Disco | ~2 GB para os modelos (Whisper + Opus-MT) |
| Internet | Necessária para baixar os modelos na instalação e para a voz neural do Edge-TTS (há fallback offline para a voz) |
| Driver de áudio | VB-CABLE (gratuito) |

## Instalação

### Forma automática (recomendada)

```powershell
git clone https://github.com/elaineguimaraes/tradutorsimultaneo.git
cd tradutorsimultaneo
```

Se você não tem o Git instalado, baixe o projeto pelo botão verde
**Code → Download ZIP** na página do GitHub e extraia numa pasta (por
exemplo, `C:\TradutorSimultaneo`).

Depois, dois cliques em **`instalar.bat`**. Ele faz tudo sozinho, em 5 passos:

1. Verifica se o Python está instalado; se não estiver, instala via `winget`.
2. Cria o ambiente virtual (`.venv`) e instala as dependências do
   `requirements.txt`.
3. Verifica se o driver VB-CABLE já está instalado; se não estiver, baixa e
   instala (pede confirmação de UAC).
4. Baixa e prepara os modelos: Whisper `base` e o tradutor Opus-MT tc-big
   (download de ~1,2 GB na primeira vez). A conversão do tradutor para o
   formato CTranslate2 acontece uma única vez e leva alguns minutos.
5. Testa a síntese de voz.

**Atenção:** o instalador do VB-CABLE costuma deixar o "CABLE" como saída de
áudio padrão do Windows. Depois da instalação, vá em
`Configurações > Sistema > Som > Saída` e escolha seus alto-falantes/fone de
volta, senão o PC fica mudo.

### Forma manual (se preferir, ou se o `.bat` falhar)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Instale o driver [VB-CABLE](https://vb-audio.com/Cable/) (gratuito, site
oficial). Depois, prepare os modelos:

```powershell
.\.venv\Scripts\python.exe scripts\preparar_modelos.py
```

Pronto. Siga para **Roteamento de áudio**, logo abaixo, e depois abra o
`run.bat`.

## Roteamento de áudio (configuração única)

O app captura o áudio pelo **VB-CABLE**. Escolha UMA das três opções abaixo:

### Opção 1: traduzir só um programa (recomendado para quem está operando)

No Windows 11: `Configurações → Sistema → Som → Mixer de volume` → no
programa desejado (por exemplo, o navegador), em "Dispositivo de saída",
escolha **CABLE Input (VB-Audio Virtual Cable)**.

Só esse programa é traduzido; o restante do som do PC (alertas do home
broker, música, notificações) toca normalmente nos alto-falantes e não passa
pelo tradutor.

### Opção 2: traduzir tudo que toca no PC

`Configurações → Sistema → Som → Saída` → escolha **CABLE Input**.

Todo o som do PC passa pelo tradutor, que devolve o áudio original mais a
tradução nos seus alto-falantes/fone. Lembre de voltar a saída padrão para os
alto-falantes ao fechar o app, senão o PC fica mudo.

### Sem VB-CABLE (modo alternativo)

Se o CABLE não estiver instalado ou for removido, o app captura o loopback
do alto-falante padrão diretamente. Funciona, mas com uma limitação: para não
traduzir a própria voz do app, tudo que é detectado como português é
ignorado.

## Uso no dia a dia

1. Dê dois cliques em **`run.bat`**.
2. Clique em **Iniciar**. Na primeira vez aparece "carregando modelos…" por
   alguns segundos; nas próximas é mais rápido.
3. Toque o áudio que quiser traduzir (no programa que você direcionou para o
   CABLE na seção *Roteamento de áudio*, acima). Em poucos segundos a voz em
   português começa e a legenda aparece na tela.

### Controles do painel

| Controle | O que faz |
|---|---|
| Iniciar / Pausar | liga/desliga o tradutor |
| Ao vivo | descarta a fila de voz atrasada e pula direto para o "agora" |
| Volume original | quanto do áudio original você continua ouvindo |
| Volume tradução | volume da voz em português |
| Ducking | quanto o áudio original ABAIXA enquanto a voz traduzida está falando |
| Voz | escolha entre Francisca, Thalita ou Antônio (vozes neurais pt-BR) |
| Velocidade | velocidade base da voz: 1,0 / 1,25 / 1,5 |
| Tema | conjunto de regras de jargão em uso (arquivos em `glossarios/`) |
| A- / A+ | diminui/aumenta o tamanho da fonte da legenda |
| Clique atravessa a legenda | o mouse passa direto pela legenda, sem atrapalhar cliques em janelas por baixo |
| Mostrar/ocultar legenda | liga ou desliga a legenda flutuante |

A legenda flutuante fica sempre por cima de qualquer janela; pode ser
arrastada para qualquer posição da tela (a posição escolhida é salva). Ela
some sozinha depois de 6 segundos de silêncio e reaparece assim que chega
texto novo.

### Linha de status

Um exemplo: `● Ouvindo (EN) · atraso 3,2s · voz +10%` significa: idioma
detectado é inglês, a voz está 3,2 segundos atrás do que está tocando ao
vivo, e a velocidade da voz foi acelerada em 10% para tentar alcançar. Quando
o locutor faz uma pausa, o atraso tende a diminuir sozinho.

## Glossários por tema

O diferencial do projeto é o glossário: um conjunto de regras que corrige o
comportamento padrão do tradutor automático para o jargão real de um
domínio específico. O app vem com dois temas prontos, cada um num arquivo
`glossarios/<nome>.json`:

| Tema | Arquivo | Conteúdo |
|---|---|---|
| `trading` (padrão) | `glossarios/trading.json` | jargão de mercado financeiro/day trade (ver exemplos abaixo) |
| `geral` | `glossarios/geral.json` | neutro, sem regras específicas; usa só o modelo de tradução |

A escolha do tema é feita pelo combo **Tema** no painel ou pela chave
`glossario` do `config.json`. Com o app rodando, a troca vale na hora, a
partir da próxima frase; antes do primeiro Iniciar, vale quando o app iniciar.

Cada arquivo de tema tem os campos `nome` e `descricao` (usados no combo da
interface), o booleano `regras_de_mercado` e quatro listas:

| Lista | Efeito |
|---|---|
| `proteger` | termos que ficam em inglês, sem tradução (nomes, siglas, jargão sem equivalente natural) |
| `tickers` | símbolos de bolsa expandidos para o nome da empresa na fala (AAPL → Apple) |
| `traduzir` | traduções fixas aplicadas antes de o texto chegar ao modelo de tradução |
| `corrigir` | substituições aplicadas na saída em português, depois da tradução |

**As listas do arquivo substituem (não somam a) os padrões embutidos** em
`src/tradutor/glossary.py` (que correspondem ao tema `trading`). Se você
remover um termo do JSON, ele deixa de valer, mesmo que exista um padrão
equivalente no código.

Além das quatro listas, existe um conjunto de correções fixas de CONTEXTO de
mercado financeiro que não vêm do JSON (ficam embutidas no código, em
`src/tradutor/glossary.py`): "trade" como substantivo, "the high"/"the low"
virando topo/fundo, número solto nunca ser lido como idade e a concordância
de gênero dos substantivos de mercado. O campo `regras_de_mercado` liga
(`true`) ou desliga (`false`) esse conjunto inteiro. Temas de outras áreas
(que não sejam mercado financeiro) devem usar `false`, porque essas regras
fixas fariam sentido só no domínio de trading (por exemplo, "the high" só
deve virar "topo" quando o assunto é o preço de um ativo).

Exemplos do tema `trading` (mais de **1.200 regras** ao todo: ≈100 termos
protegidos, 22 tickers expandidos, ≈590 traduções fixas e ≈510 correções
pós-tradução):

| Inglês | Português | Observação |
|---|---|---|
| `short` / `long` | venda/vendido, compra/comprado | tipos de operação |
| `trade` | operação / operar | substantivo e verbo tratados separadamente |
| `earnings` | lucros | resultado da empresa; nunca "ganhos" |
| `earnings call` | teleconferência de resultados | |
| `oil` | petróleo | |
| bullish / bearish | compradores / vendedores | |
| `high` / `low` | topo / fundo | não "máxima/mínima" |
| `hit` (o topo/fundo) | bater | "bateu o topo" |
| `spider` | SPY | apelido de mercado do ETF |
| AAPL, e outros tickers famosos | Apple, etc. | expandidos para o nome da empresa |
| `stop` | stop | nunca traduzido |
| `stopped` | stopado | não "nocauteado" |
| `breakout` / `breakdown` | rompimento / quebra | |
| `pop` / `rip` | repique | |
| `dip` | queda | |
| `be up` / `be down` | no lucro / no negativo | |

A lista completa do tema `trading` está em `docs/DECISOES-TECNICAS.md`.

Para auditar a qualidade da tradução, mude `gravar_log` para `true` no
`config.json` (vem desligado por padrão), reinicie o app e deixe-o rodando
por alguns minutos: o arquivo `traducoes.log`, na raiz
do projeto, grava as quatro etapas de cada frase processada: o texto em
inglês ouvido, o texto mascarado enviado ao tradutor, a saída crua do
tradutor e o texto final em português falado. É a melhor ferramenta para
identificar onde o glossário precisa de um ajuste.

Mais detalhes técnicos sobre como o glossário resolve casos difíceis (ordem
das regras, concordância de gênero, ambiguidades de "long"/"short" etc.) em
[`docs/DECISOES-TECNICAS.md`](docs/DECISOES-TECNICAS.md).

## Configuração (`config.json`)

Criado automaticamente na primeira execução, na raiz do projeto. **Não é
versionado**, porque guarda estado específico da máquina onde o app roda
(dispositivo de áudio, posição da legenda, volumes).

| Chave | Padrão | Significado |
|---|---|---|
| `capture_device_hint` | `null` | pista de nome do dispositivo de captura (auto-detecção do CABLE) |
| `output_device` | `null` | índice do dispositivo de saída (legado; resolvido a partir de `output_device_name` no boot) |
| `output_device_name` | `null` | nome do dispositivo de saída (é o que vale; o índice é derivado dele) |
| `gain_original` | `0.30` | volume do áudio original |
| `gain_tts` | `1.00` | volume da voz traduzida |
| `duck_level` | `0.85` | fração de REDUÇÃO do original enquanto a voz traduzida fala |
| `whisper_model` | `"base"` | modelo do faster-whisper usado na transcrição |
| `tts_voice` | `"pt-BR-FranciscaNeural"` | voz do Edge-TTS |
| `tts_speed` | `1.0` | velocidade base da voz (1,0 / 1,25 / 1,5) |
| `silence_ms` | `450` | silêncio (ms) que fecha um segmento de fala no VAD |
| `max_segment_s` | `7.0` | duração máxima (s) de um segmento antes de ser cortado |
| `rate_ladder` | `[[2.0, 10], [5.0, 25], [9.0, 40]]` | escada de atraso (s) → aceleração da voz (%) |
| `max_backlog_s` | `12.0` | atraso acumulado (s) acima do qual o app pula para o ao vivo |
| `gravar_log` | `false` | `true` grava `traducoes.log` com as 4 etapas de cada frase (auditoria de tradução) |
| `glossario` | `"trading"` | tema do glossário (nome do arquivo em `glossarios/`, sem `.json`) |
| `show_subtitles` | `true` | exibe a legenda flutuante |
| `subtitle_font_size` | `18` | tamanho da fonte da legenda |
| `subtitle_pos` | `null` | posição salva da legenda na tela |
| `overlay_click_through` | `false` | clique atravessa a legenda (não intercepta o mouse) |

## Solução de problemas

| Sintoma | Causa / solução |
|---|---|
| PC sem som | A saída padrão do Windows ficou no CABLE Input. Vá em `Configurações → Som → Saída` e escolha os alto-falantes |
| App não lista o CABLE | O Windows enumera dispositivos na abertura do processo. Feche e abra o app de novo |
| Sem voz traduzida, legenda OK | Falta de internet para o Edge-TTS. Procure "só voz offline" no `tradutor.log`: o app já caiu para a voz SAPI offline |
| Primeira frase demora para sair | Carregamento dos modelos na primeira execução; estabiliza logo depois |
| Tradução ruim em algum jargão específico | Edite o arquivo do tema em `glossarios/` e audite o `traducoes.log` para ver onde a tradução desviou |
| Tradução sai em português de Portugal ("registou", "ações" viram "acções") | O tradutor Opus-MT não ficou pronto na instalação e o app está usando o Argos de reserva. Rode `.venv\Scripts\python.exe scripts\preparar_modelos.py` e confira se imprime `backend=ct2` |
| Ao reabrir, avisa "já está aberto" | Um processo anterior ficou preso ao fechar. Feche pelo Gerenciador de Tarefas e abra de novo |
| App trava | Olhe o `tradutor.log` primeiro, que é o log geral da aplicação |
| Voz cortando/atrasando muito, tudo lento ao mesmo tempo | Sintoma de pico de CPU de outro processo (ex.: antivírus); o app se recupera sozinho quando a CPU libera |

## Para desenvolvedores

| Módulo | Responsabilidade |
|---|---|
| `audio_io.py` | captura WASAPI loopback e mixer de saída (ducking + voz) |
| `segmenter.py` | segmentação de fala (VAD) e transcrição (faster-whisper) |
| `glossary.py` | máscara de jargão, tradução fixa e correções pós-tradução |
| `translate_tts.py` | tradução (Opus-MT/CTranslate2, com Argos Translate como reserva) e síntese de voz (Edge-TTS/SAPI) |
| `gui.py` | interface gráfica (Tkinter): painel de controle e legenda flutuante |
| `main.py` | orquestração: liga captura → VAD → ASR → tradução → TTS → mixer + GUI |
| `contracts.py` | contratos/protocolos entre módulos, a fonte da verdade da arquitetura |
| `config.py` | configuração persistida (`config.json`) |
| `dsp.py` | utilitários de processamento de sinal compartilhados (downmix, resample) |

`contracts.py` é o ponto de partida recomendado para entender a arquitetura:
cada módulo implementa as interfaces de lá, e a integração em `main.py` não
usa nada além delas.

### Testes

Cada arquivo em `tests/` é executável de forma independente (sem pytest):

```powershell
.venv\Scripts\python.exe tests\test_glossary.py
.venv\Scripts\python.exe tests\test_segmenter.py
.venv\Scripts\python.exe tests\test_audio_io.py --offline
.venv\Scripts\python.exe tests\test_translate_tts.py     # baixa modelo / precisa de internet p/ TTS
.venv\Scripts\python.exe tests\test_gui.py               # revisão visual
.venv\Scripts\python.exe tests\test_e2e.py               # requer VB-CABLE + internet
```

Logs de diagnóstico: `tradutor.log` (log geral, rotativo) e `traducoes.log`
(auditoria de tradução, 4 etapas por frase), ambos gerados na raiz e não
versionados. Documentação técnica completa (decisões de modelo,
anti-repetição, manutenção do glossário, armadilhas conhecidas) em
[`docs/DECISOES-TECNICAS.md`](docs/DECISOES-TECNICAS.md).

## Contribuindo

Há dois caminhos para contribuir com o projeto:

1. **Novo tema (outro segmento):** copie `glossarios/geral.json` (já vem com
   `regras_de_mercado` em `false`) para `glossarios/<seu-tema>.json` (nome
   curto, minúsculo, sem espaços ou acentos), preencha os campos `nome` e
   `descricao`, adicione as regras (`proteger`, `tickers`, `traduzir`,
   `corrigir`), rode `tests\test_glossary.py` (ele valida todos os temas em
   `glossarios/`) e abra um Pull Request só com esse arquivo. Dicas: comece
   pela lista `proteger` (o que deve ficar em inglês) e pela lista `corrigir`
   (o que o tradutor erra); use o `traducoes.log` para achar os erros de
   tradução do seu domínio; mantenha `regras_de_mercado` em `false`, a menos
   que o seu tema também seja sobre mercado financeiro; não inclua dados
   pessoais ou de terceiros.
2. **Melhorias no tema `trading` ou no código:** abra um Pull Request com o
   trecho do `traducoes.log` (o texto em inglês ouvido e o texto final em
   português) que mostra o erro, junto com a regra proposta.

Sugestões e problemas: abra uma Issue no repositório.

Veja também [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Limitações conhecidas

- Funciona apenas no Windows (depende de captura WASAPI loopback e da voz
  offline SAPI).
- Traduz apenas inglês → português do Brasil: o modelo de tradução é
  específico para esse par de idiomas; o Whisper detecta o idioma falado e
  ignora o que não for inglês.
- Latência de alguns segundos é inerente ao pipeline (segmentação de fala →
  transcrição → tradução → síntese de voz). Não é um bug pontual a corrigir.
- A voz neural depende de internet; sem conexão, a qualidade cai para a voz
  offline do Windows.
- A qualidade da transcrição cai com música alta de fundo ou vários locutores
  falando ao mesmo tempo.

## Créditos e licenças

Este projeto é distribuído sob a licença MIT (veja [`LICENSE`](LICENSE)).

Componentes de terceiros utilizados:

| Componente | Uso | Licença |
|---|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | transcrição de fala | MIT |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | motor de inferência do tradutor | MIT |
| [Opus-MT tc-big en-pt](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-pt) (Helsinki-NLP) | modelo de tradução | CC-BY-4.0 |
| [Silero VAD](https://github.com/snakers4/silero-vad) | detecção de atividade de voz | MIT |
| [Argos Translate](https://github.com/argosopentech/argos-translate) | tradução de reserva | MIT |
| [edge-tts](https://github.com/rany2/edge-tts) | síntese de voz neural pt-BR | GPL-3.0 (dependência via pip) |
| [pyttsx3](https://github.com/nateshmbhat/pyttsx3) | síntese de voz offline (SAPI) | MPL-2.0 |
| [VB-CABLE](https://vb-audio.com/Cable/) | driver de áudio virtual | licença própria da VB-Audio, instalado separadamente |

## Outros projetos

[LangCoach](https://langcoach.ia.br): treino de entrevistas de emprego em
inglês pelo WhatsApp. Você recebe perguntas por áudio e texto, responde por
voz e ganha feedback sobre o inglês, o conteúdo e a forma como apresentou suas
experiências e resultados. Ao final de cada sessão chega um relatório em PDF
com os pontos a melhorar, que serve de guia de estudo com professor
particular, com IA ou em cursos.
