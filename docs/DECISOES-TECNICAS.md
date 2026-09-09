# Decisões técnicas e notas de manutenção

Este documento reúne o "porquê" por trás das escolhas de implementação do
Tradutor Simultâneo, útil para quem for dar manutenção no código ou entender
por que certas soluções, aparentemente estranhas, existem.

## 1. Visão geral do pipeline

O app roda em threads separadas conectadas por filas (`queue.Queue`), uma por
estágio do pipeline. `src/tradutor/contracts.py` é a fonte da verdade da
arquitetura: cada módulo implementa as interfaces de lá, e a integração
(`main.py`) não usa nada além delas.

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

Nenhum estágio pode usar `queue.put()` bloqueante: um estágio mais lento (por
exemplo o TTS aguardando rede) travaria o pipeline inteiro. Por isso todo
enfileiramento passa por `Pipeline._drop_oldest_put`, que descarta o item mais
antigo da fila em vez de bloquear.

## 2. Decisões de modelo

**Transcrição (ASR):** `faster-whisper` no modelo `base`, quantizado em int8,
com 8 threads: RTF (tempo de processamento / duração do áudio) ≈ 0,32 num
Ryzen 7 5700U, ou seja, processa bem mais rápido que o tempo real. O modelo
`small` é mais preciso, mas tem RTF ≈ 1,0 em CPU: no limite do tempo real, sem
folga para picos de carga da máquina. O decoder usa uma escada de temperatura
(0,0 / 0,2 / 0,4) combinada com `compression_ratio_threshold` e
`repetition_penalty=1,1` para conter loops de repetição do modelo; como rede
de segurança adicional, `collapse_repetitions()` roda depois do ASR.

**Tradução (MT):** Opus-MT tc-big en→pt, convertido para CTranslate2 int8
(backend `"ct2"` em `translate_tts.py`), carregado a partir de
`models/opus-mt-en-pt-ct2/` (gerado por `scripts/preparar_modelos.py`, não
versionado, veja o README). O primeiro token da sequência de origem é o
marcador `>>pob<<`, que seleciona a variante pt-BR do modelo (multi-alvo);
sem ele o modelo tende a pt-PT. Por isso o Argos Translate, usado nas
primeiras versões, foi rebaixado a reserva automática: entra em ação apenas
quando `models/` não existe (instalador não rodou ou falhou).

Detalhes de decodificação:
- `repetition_penalty=1,2` no MT, nunca `no_repeat_ngram_size`, porque essa
  opção mutila os tokens de proteção do glossário (`XPROTECTEDnX`), que
  compartilham subpalavras entre si na tokenização.
- A tradução acontece FRASE a frase: `_SENT_RE` separa o segmento e todas as
  frases são traduzidas numa única chamada em lote. O Marian (arquitetura do
  Opus-MT) é treinado sentence-level; alimentá-lo com um segmento de várias
  frases fragmentado pelo ASR era o gatilho mais comum de loops de paráfrase.
- `max_decoding_length` é limitado a ≈ 1,6× o tamanho da origem, o que corta
  qualquer eco que sobreviva às defesas anteriores.
- `intra_threads=4` no `ctranslate2.Translator`, para não disputar todos os
  núcleos de CPU com o Whisper (que já usa 8 threads).
- Latência típica: 350-500 ms por frase no MT (contra ~250 ms do antigo
  Argos). Cabe no orçamento de tempo real, já que o TTS sozinho leva 1,5-3 s.

## 3. Anti-repetição em três pontos

Ecos de repetição podem entrar em qualquer estágio, e cada um tem sua própria
defesa:

1. **ASR:** `collapse_repetitions(min_repeats=3)`: o Whisper eventualmente
   repete a mesma palavra/frase 3+ vezes seguidas em áudio ruidoso.
2. **Entrada do MT:** `collapse_repetitions(min_repeats=2)` no texto que sai
   do ASR antes de mascarar/traduzir, pois um eco do tipo "resistance
   resistance resistance" faz o Marian alucinar na tradução (ex.: virar
   "resistência a água"). Além disso, conectores soltos no fim do segmento
   são removidos (`_TRAIL_CONN_RE`: "I mean", "so", "and"...) porque penduram
   e geram eco de paráfrase na tradução. Muletas de fala em inglês ("uh",
   "um") também são removidas antes de traduzir (`_strip_fillers`), pois o
   Marian entra em loop com elas.
3. **Saída do MT:** `collapse_repetitions(min_repeats=2)` de novo, mais
   `_dedupe_adjacent_sentences`: na tradução, uma palavra repetida 3 vezes
   nunca é ênfase legítima, mas repetida 2 vezes pode ser (por isso o limiar
   mais baixo que no ASR).

## 4. Glossário: como funciona e como manter

O glossário é organizado por tema: cada arquivo `glossarios/<tema>.json` tem
quatro listas (`proteger`, `tickers`, `traduzir` e `corrigir`) que
**substituem** (não somam a) os padrões embutidos em `src/tradutor/glossary.py`,
que correspondem ao tema `trading` (fallback se o arquivo do tema sumir ou
ficar corrompido). O app vem com dois temas: `trading` (padrão, jargão de
mercado financeiro) e `geral` (neutro, sem regras, serve de modelo para
novos temas). A regra "editar os dois arquivos juntos" (o JSON e os padrões
embutidos em `glossary.py`) vale só para o tema `trading`; um novo tema não
mexe em `glossary.py`, só no próprio JSON.

`glossary.py` expõe dois utilitários para lidar com temas:

- `theme_path(name)`: resolve o nome de um tema (ex.: `"geral"`) para o
  caminho completo do arquivo (`glossarios/geral.json`); usa `os.path.basename`
  para não permitir escapar do diretório.
- `list_themes()`: lista `(nome_do_arquivo, rótulo)` de todos os temas
  válidos em `glossarios/`, lendo o campo `nome` de cada JSON para o rótulo.
  Arquivos inválidos são ignorados (com aviso no log) em vez de derrubar a
  lista inteira. O tema padrão (`trading`) sempre vem primeiro; os demais em
  ordem alfabética pelo rótulo.

O campo booleano `regras_de_mercado` no JSON do tema (lido em
`Glossary.__init__`, guardado em `self._market`, padrão `True` quando o
campo não existe, seja porque o arquivo é antigo ou porque o arquivo do tema
nem existe) liga ou desliga um conjunto de regras fixas de CONTEXTO de
mercado financeiro que NÃO vêm das quatro listas do JSON, e sim de regexes
embutidos no código:

- Em `Glossary.mask()`, as substituições `_TRADE_NOUN_RE` ("trade" como
  substantivo) e `_HILO_RE` ("the high"/"the low" -> topo/fundo) só rodam
  quando `self._market` é `True`. `_TIME_RE` (hora), as entradas do tema, os
  tickers e o auto-mask de nomes próprios continuam de fora dessa condição
  (não são específicos de mercado).
- Em `Glossary.fix()`, tudo que vem depois do gerúndio (o bloco do "anos
  fantasma", "aos N", ordinal, "às 900", "graus", "acima/abaixo de", "no N da
  manhã", "resistência a", eco de número, "mínimo/máximo de", a concordância
  de artigo `_fem_article`/`_masc_article`, "todos/todas", `_ADJ_NOUN_RE`/
  `_NOUN_ADJ_RE` e "comércio -> trade") foi extraído para um método privado
  `_fix_market(text, source)`, chamado por `fix()` só quando `self._market`
  é `True`. A parte genérica de `fix()` (`_fix_unk`, o laço de `self._fix_re`,
  o gerúndio e as regras "é/são + gerúndio") continua rodando sempre, porque
  não é específica de mercado.

`trading.json` tem `regras_de_mercado: true`; `geral.json` tem `false`. Um
tema novo de outro domínio (medicina, games, futebol) deve copiar `geral.json`
como ponto de partida e manter o campo em `false`, porque essas regras fixas
só fazem sentido no domínio de trading (por exemplo, "the high" só deve virar
"topo" quando o assunto é o preço de um ativo, não em qualquer contexto).

A troca de tema em tempo real (`Pipeline.set_glossary`, chamado pelo combo
"Tema" da GUI) roda o carregamento do JSON numa thread separada, porque
compilar as centenas de regexes do glossário leva cerca de 0,1 s e travaria a
interface. A troca do atributo `self._glossary` é atômica (reatribuição de
referência em Python), e o estágio de tradução sempre lê `self._glossary` no
início de cada frase, então a frase em curso termina com o tema antigo e só a
próxima já usa o novo.

Mecânica de proteção: antes de mandar o texto para o MT, ocorrências dos
termos protegidos são trocadas por tokens `XPROTECTEDnX` (esse formato
sobrevive ao Argos e ao Opus-MT; formatos como `[[0]]` não sobrevivem à
tokenização). Depois da tradução os tokens são restaurados com a grafia
original: a restauração é case-insensitive porque o Opus-MT às vezes devolve
o token em minúsculas.

Pontos de atenção específicos:

- `auto_proteger_nomes` está desligado por padrão: a heurística de detectar
  nomes próprios por maiúscula inicial acabava congelando palavras comuns em
  inglês só por estarem no início de frase ou em ênfase ("Congrats", "Looks",
  o "Don" de "Don't"). O Opus-MT já preserva nomes e siglas bem sozinho, sem
  essa heurística. Não religar sem revisar esse comportamento.
- `trade` como SUBSTANTIVO é tratado como palavra reservada em inglês ("um
  trade A plus", "o pior trade", "trades de qualidade"), pois é assim que o
  jargão de mercado brasileiro fala. Reconhecido por `_TRADE_NOUN_RE`, que
  casa o determinante/adjetivo à esquerda (`a`, `the`, `worst`, `quality`,
  `of`...) e mascara só a palavra, deixando o artigo de fora do token para o
  MT concordar o gênero (com o artigo dentro do token saía "a qualidade do
  uma operação"). O VERBO continua sendo traduzido normalmente ("they just
  trade the levels" → "operam"), justamente porque a regra exige um
  determinante/adjetivo à esquerda que o verbo não tem. Compostos macro
  (`trade war`, `free trade`) e `day trade` são mascarados antes e ficam
  intactos.
- No mapa `traduzir`, evite palavras soltas que também funcionam como verbo
  (`trade`, `long`, `short`, `pop`): o token de proteção congela qualquer
  forma da palavra, inclusive quando ela aparece como verbo na frase ("they
  just trade" → "eles apenas operação"). Prefira frases mais específicas
  ("this trade", "went long") e resolva o verbo com uma regra em `corrigir`
  na volta (por exemplo, "negociar" → "operar", "fuga" → "rompimento").
- O mapa `corrigir` troca a palavra e deixa o artigo que o MT gerou para trás
  (ex.: "um mergulho" → "um queda"); `_fem_article`/`_masc_article` em
  `glossary.py` corrigem a concordância depois. Dentro do mapa, formas COM
  artigo precisam vir ANTES da forma nua, porque o dicionário é aplicado na
  ordem de definição: "baixa noturna" tem que vir antes de "a baixa
  noturna", senão a substituição da forma nua roda primeiro e deixa "A fundo
  da noite". O artigo "a" sozinho ficou de fora da regra de concordância
  masculina porque "analisar a fundo" é português legítimo.
- `_HILO_RE` faz `the/a/this` + `high/low` virar topo/fundo só com lookahead
  positivo, para não mascarar expressões como "a high probability setup" ou
  "the high side", em que high/low é adjetivo, não substantivo de preço.
- Entradas do mapa `traduzir` terminadas em `long`/`short` (e as que começam
  com essas palavras) ganham lookaround automático via `_compile_entry`, para
  não mascarar expressões de tempo/duração sem relação com posição comprada/
  vendida ("my long time discomfort", "I'm down to 2 micros" não devem virar
  "no negativo"). É uma heurística que cobre só fim de frase e uma lista de
  palavras, não é análise sintática completa. Limitação conhecida: entradas
  com determinante ("the long", "my long", "a short") não casam quando a
  posição aparece no fim da frase ("I closed my long."); o custo foi aceito
  para evitar mascarar falsos positivos como "a short break" ou "this long
  red candle".
- O Opus-MT interpreta número solto como idade ("at 57" → "aos 57 anos") no
  contexto de mercado, onde na verdade é preço ou nível, nunca idade.
  `Glossary.fix(text, source)` recebe o texto original em inglês e só remove
  esse "anos" fantasma quando o texto de origem não contém `year`, `yrs` ou
  `decade`.
- O caractere "⁇" (código de token desconhecido do CT2, U+2047) e resíduos de
  token mutilado (`ED0X`, `0X`, `900x`) são limpos em `_fix_unk`/`unmask`.
- O verbo `trade` é mascarado já conjugado pelo sujeito da frase ("you trade"
  → "você opera"), porque o MT não conjuga o conteúdo de um token protegido:
  a conjugação certa tem que estar pronta antes de mascarar.

## 5. Voz

O Edge-TTS oferece três vozes pt-BR, todas suportadas pela GUI
(`App._VOICES`): Francisca, Thalita (o id real é
`pt-BR-ThalitaMultilingualNeural`, não `ThalitaNeural`) e Antônio.
`set_tts_voice` troca `TtsSpeaker.voice` em tempo real: o valor é lido a cada
síntese, então a troca vale a partir da próxima frase. A velocidade base
(`tts_speed`, 1,0 / 1,25 / 1,5, escolhida na GUI) é somada pela escada de
atraso em `_adaptive_rate()`, que acelera a voz para recuperar atraso em
relação ao áudio ao vivo, limitada a no máximo +100%. Se a rede falhar, o
fallback offline usa a SAPI do Windows com a voz Maria, independente da
escolha feita na GUI.

## 6. Áudio

A captura depende do VB-CABLE (driver de áudio virtual) e detecta o
dispositivo CABLE automaticamente. O roteamento típico é: saída padrão do
Windows → CABLE Input; o mixer do app devolve o som (original + tradução) nos
alto-falantes reais. `output_device_name` é a fonte da verdade da
configuração de saída: o índice numérico do dispositivo (`output_device`) é
resolvido a partir do nome no boot do app, porque o índice WASAPI pode mudar
entre reinicializações do Windows ou entre máquinas diferentes. Por isso
`config.json` não é versionado: ele guarda estado específico de cada máquina
(dispositivo, posição da legenda, volumes).

## 7. Logs de diagnóstico

Dois arquivos de log rotativos são gravados na raiz do projeto (fora do
controle de versão):

- `tradutor.log`: log geral da aplicação; primeiro lugar a olhar em caso de
  travamento ou comportamento estranho.
- `traducoes.log`: grava as quatro etapas de cada frase processada: o texto
  em inglês ouvido pelo ASR, o texto mascarado que chega ao MT, a saída crua
  do MT (antes de restaurar/corrigir) e o texto final em português falado.
  É a ferramenta principal para auditar qualidade de tradução: basta deixar o
  app rodando por alguns minutos e ler esse arquivo. Controlado pela chave
  `gravar_log` no `config.json` (desligado por padrão, para não encher o
  disco em uso normal); quando desligado, o `Glossary.apply` nem monta a
  mensagem de log.

## 8. Armadilhas conhecidas

- Uma GUI Tkinter lançada por um shell de automação (sem sessão de desktop
  interativa) abre numa estação de janela invisível; o app deve ser aberto
  pelo `run.bat` numa sessão de desktop normal, não por script/automação.
- A cauda pendurada de um segmento cortado no meio é limpa por
  `_strip_trailing_junk`, que roda em laço até estabilizar. Duas regras
  trabalham juntas: `_TRAIL_CONN_RE` (conectores e muletas de fala, roda
  sempre) e `_TRAIL_CUT_RE` (preposições e pronomes como `into`, `up`, `its`,
  aplicada só quando o texto não termina em pontuação, senão mutilaria uma
  frase completa como "The market is up."). Um único passo não é suficiente:
  "running into a" perde o "a" e deixa a preposição solta, que sozinha já é
  gatilho de eco na tradução, daí o laço.
- O slider de ducking da GUI representa REDUÇÃO do volume original enquanto a
  voz traduzida fala, e é invertido internamente para o ganho do mixer: não
  confundir com "quanto mais alto o slider, mais volume".
- `get_ui_state().running` já fica `True` DURANTE o carregamento dos modelos:
  é o que mantém o botão em "Pausar" nesse período. Qualquer código que
  precise saber que a captura de fato abriu deve esperar o campo `status`
  começar com "ouvindo", não apenas checar `running`.
- Nenhum estágio do pipeline pode usar `queue.put()` bloqueante (ver seção 1).

## 9. Robustez

O processo eleva sua própria prioridade para `ABOVE_NORMAL` no Windows
(`_boost_priority()` em `main.py`) por causa de picos de CPU causados por
processos externos (o suspeito mais comum é a verificação periódica de um
antivírus): nesses picos, o ASR passa a levar de 26 a 32 segundos para
processar um segmento curto (RTF > 4), e MT e TTS ficam lentos ao mesmo
tempo: quando tudo desacelera junto, o gargalo é a máquina, não um estágio
específico do pipeline. O app se recupera sozinho: quando o atraso acumulado
ultrapassa `max_backlog_s`, ele descarta áudio antigo e pula para o "ao
vivo".

Diagnóstico de travamento por sintoma:

- **Janela não responde:** normalmente é o QuickEdit do console do Windows
  (desligado por `_disable_console_quickedit`) ou uma parada síncrona longa
  (movida para thread própria).
- **GUI responde, mas nada é traduzido:** suspeitar de TTS/rede (procurar
  "só voz offline pelos próximos" no log, que indica o disjuntor do Edge-TTS
  acionado) ou de captura de áudio parada, sinalizada pelo watchdog com
  "captura sem áudio há Xs, reabrindo" após 20 s sem dados.
- **Voz duplicada após pausar e retomar:** sintoma de workers "fantasmas" que
  não terminaram; o contador `_generation` existe justamente para impedir
  isso.
- **App fecha, mas ao reabrir avisa "já está aberto":** o processo anterior
  ficou preso na finalização; por isso `main()` termina com `os._exit(0)`.

Qualquer funcionalidade que dependa de rede precisa de um caminho offline de
reserva, porque conexões instáveis (falhas intermitentes de DNS) são um
cenário esperado, não uma exceção.

## 10. Convenções de tradução do jargão financeiro

O produto foi afinado para transmissões de trading e análise técnica em
inglês (tema `trading`). Tabela de referência (todas as regras vivem em
`glossarios/trading.json` e nos padrões de `src/tradutor/glossary.py`):

| Inglês | Português | Observação |
|---|---|---|
| `short` / `long` | venda/vendido, compra/comprado | tipos de operação, sempre traduzidos |
| `trade` | operação / operar | ver seção 4 sobre substantivo vs. verbo |
| `earnings` | lucros | resultado/balanço da empresa; NUNCA "ganhos" (isso é `gain`) |
| `earnings call` | teleconferência de resultados | composto de mercado |
| `after earnings` | depois do balanço | composto de mercado |
| `earnings reaction` | reação ao balanço | composto de mercado |
| `oil` | petróleo | |
| bullish / bearish | compradores / vendedores | tom do mercado |
| `high` / `low` | topo / fundo | não "máxima/mínima" |
| `hit` (o topo/fundo) | bater | "bateu o topo" |
| `spider` | SPY | apelido de mercado para o ETF |
| tickers famosos (AAPL, etc.) | nome da empresa (Apple) | expandidos para leitura em voz |
| `pivot` | pivô | |
| `order` | ordem | |
| `fill` | execução da ordem | "my fill" = minha ordem executada |
| `stopped` / `knocked out` | stopado | não "nocauteado" |
| `stop` | stop | NUNCA traduzido ("meu stop", "stop loss") |
| `breakout` | rompimento | |
| `breakdown` | quebra | |
| `pop` / `rip` | repique | |
| `dip` | queda | |
| `vomit` | despencar | |
| `be up` / `be down` | estar no lucro / no negativo | |
| `move` | movimento | |
| `play` | jogada | |
| `break even` | break even | protegido, fica em inglês |

Jargão que permanece em inglês por convenção de mercado: `pips`, `ticks`,
`VWAP`, `SPY`, `market makers`, entre outros termos técnicos sem tradução
natural no dia a dia de quem opera. Nomes próprios, empresas e instituições
(Federal Reserve, Amazon, Target etc.) nunca são traduzidos.

A saída é sempre em português do Brasil, nunca português europeu: formas como
"bilhões", "registrou" e o gerúndio ("subindo") são preferidas às variantes
de Portugal.
