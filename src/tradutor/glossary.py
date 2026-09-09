"""Glossário de termos protegidos: nomes próprios que NÃO devem ser traduzidos.

Antes da tradução, ocorrências dos termos são substituídas por tokens que o
modelo comprovadamente preserva (``XPROTECTEDnX``); depois são restauradas com
a grafia original. Um mapa de correções pós-tradução serve de rede de
segurança (ex.: "Reserva Federal" -> "Federal Reserve").

O usuário pode editar ``glossario.json`` na raiz do projeto:
    {
      "proteger": ["Federal Reserve", "Fed", ...],
      "corrigir": {"Reserva Federal": "Federal Reserve", ...}
    }
As listas do arquivo SUBSTITUEM as padrão (para remover um termo, basta
apagá-lo do arquivo).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Tuple

log = logging.getLogger("tradutor.glossario")

_TIME_RE = re.compile(r"\b(\d{1,2})\s*([AaPp])\.?[Mm]\.?(?!\w)")

# "trade" SUBSTANTIVO fica em inglês, pois é como o mercado brasileiro fala
# ("um trade A plus", "o pior trade", "trades de qualidade"). O VERBO continua
# traduzido ("they just trade the levels" -> "operam"), por isso o substantivo
# é reconhecido pelo determinante/adjetivo à esquerda, e não como palavra solta
# no mapa (palavra solta congelaria a forma errada; ver docs/DECISOES-TECNICAS.md).
# O artigo NÃO entra no token: quem concorda gênero é o MT ("the quality of a
# trade" -> "a qualidade de um trade"; com o artigo dentro saía "do uma
# operação").
_TRADE_DET = (
    r"a|an|the|this|that|these|those|my|your|his|her|its|our|their|one"
    r"|first|last|best|worst|good|bad|great|nice|quality|green|red|winning"
    r"|losing|perfect|solid|big|small|every|each|any|some|no|more|most|plus|of"
    r"|biggest|final|next|scalp|swing|similar|same|whole|entire|another|other"
    r"|new|easy|single|actual|real|live|paper|smallest|largest|cleanest"
    r"|riskiest|safest|only")
_TRADE_NOUN_RE = re.compile(r"(?i)\b(" + _TRADE_DET + r")(\s+)(trades?)\b")

# high/low com determinante -> topo/fundo (mesmo padrão de `_TRADE_NOUN_RE`):
# o determinante fica FORA do token porque é o MT quem concorda o artigo
# ("a high probability setup" não pode virar "a topo probability setup" -
# por isso o lookahead positivo exige que o high/low termine ali mesmo, como
# substantivo, e não como adjetivo de outra palavra).
_HILO_DET = (
    r"a|an|the|this|that|these|those|its|our|their|another|other|major|minor"
    r"|key|last|next|new|fresh|first|second|third|same|big|solid|clean|nice"
    r"|good|clear|significant|important")
_HILO_RE = re.compile(
    r"(?i)\b(" + _HILO_DET + r")(\s+)(highs?|lows?)\b"
    r"(?=\s*(?:[.,!?;:)]|$|of\b|from\b|at\b|here\b|there\b|is\b|was\b|are\b"
    r"|were\b|and\b|to\b|in\b|on\b|by\b|that\b|which\b|for\b|\d|again\b|yet\b"
    r"|so\b|but\b|right\b|around\b|near\b|out\b|back\b|today\b|yesterday\b"
    r"|too\b|as\b|though\b|because\b|before\b|after\b|with\b|into\b))")
_HILO_PT = {"high": "topo", "highs": "topos", "low": "fundo", "lows": "fundos"}


# Entradas do "traduzir" terminadas em "long"/"short" (substantivo solto)
# ganham lookahead para não mascarar expressões de tempo/direção que nada têm
# a ver com posição comprada/vendida ("my long time discomfort", "things
# ripped this long", "the long and short of it", "it took that long", "this
# long red candle", "a short break"). A lista de palavras cobre o que segue
# "long"/"short" quando NÃO é o sentido financeiro; "$" indica fim de
# frase/pontuação (só se aplica às entradas com determinante, ver abaixo,
# frases como "I'm long." não podem ser barradas por isso).
_LS_WORDLIST = (
    r"time|term|run|while|ago|way|story|enough|standing|haul|list"
    r"|distance|lived|answer|shot|side|cut|form|day|days|week|weeks|month"
    r"|months|year|years|hour|hours|minute|minutes|ish|as|and|or|before"
    r"|after|red|green|candle|squeeze|break|wick|bar|tail|\d")
# Para entradas com DETERMINANTE à esquerda ("a short", "the long", "this
# play"...): barra também fim de frase/pontuação, porque nesse caso o
# substantivo solto quase sempre é seguido de outro substantivo comum
# ("a short break", "this long red candle") quando não é posição.
_LS_NOT_AFTER_DET = r"(?!\s*(?:$|[,.;!?])|\s+(?:" + _LS_WORDLIST + r"))"
# Para entradas que COMEÇAM com pronome/verbo ("I'm long", "was short", "went
# long"...): o sentido financeiro é normal em fim de frase ("I'm long.",
# "who is short?"): só barra quando seguido de uma das palavras da lista.
_LS_NOT_AFTER_PRON = r"(?!\s+(?:" + _LS_WORDLIST + r"))"
# Prefixos que indicam sentido financeiro completo mesmo no fim da frase
# (pronome/verbo à esquerda de long/short).
_LS_PRON_PREFIXES = (
    "i'm ", "i am ", "was ", "is ", "are ", "still ", "who ", "who's ",
    "currently ", "went ", "go ", "going ", "took ", "take ", "i went ",
)
# Entradas que COMEÇAM com "long "/"short " ganham lookbehind para não
# mascarar advérbios de intensidade + "long" no sentido de duração
# ("how long the market stays…" não pode virar "how comprar o…"). NÃO inclui
# "that ": "that short squeeze" é o protegido "short squeeze" (financeiro),
# não "that" + duração.
_LS_NOT_BEFORE = (
    r"(?<![Hh]ow )(?<![Tt]oo )(?<![Ss]o )(?<![Aa]s )(?<![Vv]ery )")
# "I'm down"/"we're down"/... não é prejuízo quando seguido de destino/valor.
_DOWN_NOT_AFTER = r"(?!\s+(?:to|too|there|here|with|for|from)\b)"
# "I'm up"/"we're up"/... não é lucro quando seguido de destino/tempo.
_UP_NOT_AFTER = r"(?!\s+(?:to|there|here|early|late|for|by|next|against)\b)"


def _compile_entry(t: str) -> "re.Pattern":
    """Compila o regex de uma entrada de `proteger`/`traduzir`.

    Heurística: cobre só fim de frase e as palavras da lista, não é uma
    análise sintática completa. Entradas terminadas em "long"/"short" e
    começadas por "long "/"short " ganham lookaround extra para não capturar
    expressões de tempo/duração que usam essas palavras sem relação com
    posição comprada/vendida (ver docs/DECISOES-TECNICAS.md). Entradas que começam com
    pronome/verbo ("I'm long", "was short"...) têm sentido financeiro
    completo mesmo em fim de frase, então só barram as palavras da lista
    (não fim de frase); entradas com determinante ("a short", "this play")
    barram também fim de frase, pois aí o substantivo solto costuma vir
    seguido de outro substantivo comum quando não é posição. Entradas
    terminadas em " down"/" up" (ex.: "I'm down") não capturam quando
    seguidas de destino/tempo ("I'm down to 2 micros").
    """
    tl = t.lower()
    pattern = r"\b" + re.escape(t) + r"\b"
    if tl.endswith("long") or tl.endswith("short"):
        if tl.startswith(_LS_PRON_PREFIXES):
            pattern += _LS_NOT_AFTER_PRON
        else:
            pattern += _LS_NOT_AFTER_DET
    if tl.startswith("long ") or tl.startswith("short "):
        pattern = _LS_NOT_BEFORE + pattern
    if tl.endswith(" down"):
        pattern += _DOWN_NOT_AFTER
    elif tl.endswith(" up"):
        pattern += _UP_NOT_AFTER
    return re.compile(pattern, re.IGNORECASE)

# O mapa "corrigir" troca a PALAVRA e deixa o artigo do MT para trás
# ("um mergulho" -> "um queda"). `_fem_article` acerta a concordância.
_FEM_ART = {"um": "uma", "o": "a", "este": "esta", "esse": "essa",
            "aquele": "aquela", "meu": "minha", "seu": "sua",
            "nosso": "nossa",
            # plurais
            "os": "as", "estes": "estas", "esses": "essas",
            "aqueles": "aquelas", "meus": "minhas", "seus": "suas",
            "nossos": "nossas", "todos": "todas"}


def _fem_article(m: "re.Match") -> str:
    art = _FEM_ART.get(m.group(1).lower(), m.group(1))
    if m.group(1)[0].isupper():
        art = art[0].upper() + art[1:]
    return f"{art} {m.group(2)}"


# Sentido inverso ("a baixa noturna" -> "a fundo da noite"). O artigo "a"
# solto fica DE FORA: "analisar a fundo" é português legítimo e viraria
# "analisar o fundo".
_MASC_ART = {"uma": "um", "esta": "este", "essa": "esse",
             "aquela": "aquele", "minha": "meu", "sua": "seu",
             "nossa": "nosso",
             # plurais
             "as": "os", "estas": "estes", "essas": "esses",
             "aquelas": "aqueles", "minhas": "meus", "suas": "seus",
             "nossas": "nossos", "todas": "todos"}


def _masc_article(m: "re.Match") -> str:
    art = _MASC_ART.get(m.group(1).lower(), m.group(1))
    if m.group(1)[0].isupper():
        art = art[0].upper() + art[1:]
    return f"{art} {m.group(2)}"


# Concordância de gênero do adjetivo depois/antes de um substantivo feminino
# que a máscara/correção restaurou já flexionado (o mapa "corrigir" troca só
# a palavra: "uma quebra falso" precisa virar "uma quebra falsa").
_ADJ_FEM = {
    "falso": "falsa", "pequeno": "pequena", "limpo": "limpa", "bom": "boa",
    "novo": "nova", "grande": "grande", "comprador": "compradora",
    "vendedor": "vendedora", "pendente": "pendente", "ativo": "ativa",
    "executado": "executada", "parcial": "parcial", "belo": "bela",
    "bonito": "bonita",
}

_NOUN_ADJ_RE = re.compile(
    r"\b(quebras?|quedas?|operaç(?:ão|ões)|ordens?|posiç(?:ão|ões)"
    r"|execuç(?:ão|ões)|jogadas?|vendas?|compras?|tendências?)(?!-)"
    r" ([A-Za-zÀ-ÿ]+)(s?)\b", re.IGNORECASE)

_ADJ_NOUN_RE = re.compile(
    r"\b(falso|pequeno|bom|novo|grande|bonito|belo)"
    r" (quebras?|quedas?|operaç(?:ão|ões)|ordens?|jogadas?|execuç(?:ão|ões))(?!-)\b",
    re.IGNORECASE)


def _agree_noun_adj(m: "re.Match") -> str:
    noun, adj, adj_s = m.group(1), m.group(2), m.group(3)
    fem = _ADJ_FEM.get(adj.lower())
    if fem is None:
        return m.group(0)
    if adj[0].isupper():
        fem = fem[0].upper() + fem[1:]
    # pluraliza o adjetivo se o substantivo está no plural (termina em "s")
    # e o adjetivo feminino ainda não - "as quedas pequeno" -> "pequenas"
    plural = adj_s or (noun.endswith("s") and not fem.endswith("s"))
    if plural and not fem.endswith("s"):
        fem += "s"
    return f"{noun} {fem}"


def _agree_adj_noun(m: "re.Match") -> str:
    adj, noun = m.group(1), m.group(2)
    fem = {"falso": "falsa", "pequeno": "pequena", "bom": "boa",
           "novo": "nova", "grande": "grande", "bonito": "bonita",
           "belo": "bela"}.get(adj.lower(), adj)
    if adj[0].isupper():
        fem = fem[0].upper() + fem[1:]
    # pluraliza o adjetivo se o substantivo está no plural (termina em "s")
    # e o adjetivo feminino ainda não - "falso quebras" -> "falsas quebras"
    if noun.endswith("s") and not fem.endswith("s"):
        fem += "s"
    return f"{fem} {noun}"


DEFAULT_PROTECT: List[str] = [
    "Federal Reserve", "Fed chair", "Fed", "FOMC", "Treasuries",
    "Nasdaq", "Dow Jones", "S&P 500", "S&P", "Wall Street",
    "Jerome Powell", "Powell",
    # jargão de trading que o mercado BR usa em inglês
    "Market Makers", "Market Maker", "short squeeze",
    "pips", "pip", "ticks", "tick", "VWAP",
    "day trading", "day trade", "day traders", "day trader",
    "traders", "trader", "scalping", "scalps", "scalp",
    # tickers/índices falados com frequência
    "SPY", "QQQ", "DXY", "MFF", "momentum",
    "rally", "rallies", "CPI",
    # jargão, produtos e nomes de plataformas que ficam em inglês
    "break even", "breakeven", "break-even", "break evens",
    "Discord", "Bookmap", "book map", "Tradeify", "Tradify", "OnlyFans",
    "only fans", "Texas Roadhouse", "CrowdStrike",
    "prop firm", "prop firms", "prop account", "prop accounts",
    "drawdown", "drawdowns", "slippage", "setup", "setups", "stream",
    "streams", "streamer", "streamers",
    "DM", "DMs", "front running", "front run", "front-run", "dark pool",
    "dark pools",
    "swing trading", "swing trade", "swing trades", "swing trader",
    "swing traders", "trade copier",
    "MFFs", "put credit spread", "put credit spreads", "call credit spread",
    "call credit spreads",
    "credit spread", "credit spreads", "call spread", "put spread",
    "call spreads", "put spreads",
    "gap", "gaps", "tickers", "ticker", "strike", "strikes", "Mag 7", "NQ",
    "MNQ", "ES futures", "options flow",
    "high water mark", "Breakout prop", "Breakout Prop firm",
]

# Tickers expandidos para o nome da empresa (case-SENSITIVE: "COIN" ticker
# não pode casar com a palavra "coin"). Editável em glossario.json:"tickers".
DEFAULT_TICKERS: Dict[str, str] = {
    "AAPL": "Apple", "TSLA": "Tesla", "NVDA": "Nvidia", "MSFT": "Microsoft",
    "AMZN": "Amazon", "GOOGL": "Google", "GOOG": "Google", "NFLX": "Netflix",
    "INTC": "Intel", "PLTR": "Palantir", "COIN": "Coinbase",
    "BABA": "Alibaba", "JPM": "JPMorgan", "WMT": "Walmart", "XOM": "Exxon",
    "MCD": "McDonald's", "PYPL": "PayPal", "ABNB": "Airbnb",
    "HOOD": "Robinhood", "GME": "GameStop", "SPX": "S&P 500",
    "NDX": "Nasdaq 100",
}

# Termos com TRADUÇÃO FIXA: substituídos por token antes do MT e restaurados
# já em português (o tradutor automático erraria: "short" -> "curto").
# Frases mais longas têm prioridade sobre palavras soltas.
DEFAULT_TRANSLATE: Dict[str, str] = {
    "I'm short": "estou vendido",
    "I am short": "estou vendido",
    "I'm long": "estou comprado",
    "I am long": "estou comprado",
    "going short": "vendendo",
    "go short": "vender",
    "going long": "comprando",
    "I went long": "comprei",
    "went long": "comprou",
    "I went short": "vendi",
    "went short": "vendeu",
    "my short": "minha venda",
    "my long": "minha compra",
    "go long": "comprar",
    "short on": "vendido no",
    "long on": "comprado no",
    "short position": "posição de venda",
    "long position": "posição de compra",
    "buying the dip": "comprando na queda",
    "buy the dip": "comprar na queda",
    "selling the rip": "vendendo no repique",
    "sell the rip": "vender no repique",
    "rallying": "subindo",
    "off to the races": "disparado",
    "op-ed": "artigo de opinião",
    "op ed": "artigo de opinião",
    "trading history": "histórico de trading",
    "pivot points": "pontos de pivô",
    "pivot point": "ponto de pivô",
    "pivots": "pivôs",
    "pivot": "pivô",
    "spiders": "SPY",
    "spider": "SPY",
    # high = topo, low = fundo
    "all-time highs": "topos históricos",
    "all-time high": "topo histórico",
    "record highs": "topos históricos",
    "record high": "topo histórico",
    "new highs": "novos topos",
    "new high": "novo topo",
    "new lows": "novos fundos",
    "new low": "novo fundo",
    "higher highs": "topos ascendentes",
    "lower lows": "fundos descendentes",
    "swing high": "topo",
    "swing low": "fundo",
    "high of the day": "topo do dia",
    "low of the day": "fundo do dia",
    "overnight lows": "fundos da noite",
    "overnight low": "fundo da noite",
    "overnight highs": "topos da noite",
    "overnight high": "topo da noite",
    "the overnight session": "o overnight",
    "overnight session": "overnight",
    "the high from yesterday": "o topo de ontem",
    "the low from yesterday": "o fundo de ontem",
    "yesterday's high": "o topo de ontem",
    "yesterday's low": "o fundo de ontem",
    "looking to play": "querendo operar",
    "want to play": "quiser operar",
    "the EMAs": "as médias móveis",
    "EMAs": "médias móveis",
    "the 50 EMA": "a média móvel de 50",
    "50 EMA": "média móvel de 50",
    "the 200 EMA": "a média móvel de 200",
    "200 EMA": "média móvel de 200",
    "nasic": "Nasdaq",
    "add to the position": "aumentar a posição",
    "add to position": "aumentar a posição",
    "add to my position": "aumentar minha posição",
    "add to the winner": "aumentar a posição vencedora",
    "add to the short": "aumentar a venda",
    "add to the long": "aumentar a compra",
    "higher lows": "fundos ascendentes",
    "lower highs": "topos descendentes",
    "wicked out": "stopado pelo pavio",
    # earnings = resultado/balanço da empresa (NUNCA "ganhos", que é gain)
    "earnings call": "teleconferência de resultados",
    "earnings report": "balanço",
    "earnings season": "temporada de balanços",
    "earnings reaction": "reação ao balanço",
    "earnings winners": "destaques do balanço",
    "earnings losers": "decepções do balanço",
    "reported earnings": "divulgou o balanço",
    "report earnings": "divulgar o balanço",
    "reporting earnings": "divulgando o balanço",
    "missing on earnings": "decepcionando no balanço",
    "miss on earnings": "decepcionar no balanço",
    "beat on earnings": "superar no balanço",
    "after earnings": "depois do balanço",
    "before earnings": "antes do balanço",
    "earnings": "lucros",
    # ele chama a resistência de "wall"; "green/red" = no lucro/no prejuízo
    "running into a wall": "batendo numa parede",
    "ran into a wall": "bateu numa parede",
    "run into a wall": "bater numa parede",
    "into a wall": "numa parede",
    "running into resistance": "batendo na resistência",
    "ran into resistance": "bateu na resistência",
    "a green trade": "uma operação no lucro",
    "green trade": "operação no lucro",
    "a red trade": "uma operação no prejuízo",
    "red trade": "operação no prejuízo",
    "when I'm green": "quando estou no lucro",
    "I'm green": "estou no lucro",
    "I'm red": "estou no prejuízo",
    "the wick low": "o fundo do pavio",
    "Wicklow": "fundo do pavio",
    "wick high": "topo do pavio",
    "futures market open": "abertura dos futuros",
    "features market open": "abertura dos futuros",
    "the futures market": "o mercado futuro",
    "futures market": "mercado futuro",
    "features market": "mercado futuro",
    "post market trading": "pós-mercado",
    "post-market trading": "pós-mercado",
    "pre-market trading": "pré-mercado",
    "premarket trading": "pré-mercado",
    "the trading session": "o pregão",
    "trading session": "pregão",
    "a good play": "uma boa jogada",
    "good play": "boa jogada",
    "bad play": "jogada ruim",
    "out of the gate": "logo de cara",
    "the smartest movie": "o movimento mais inteligente",
    "smartest movie": "movimento mais inteligente",
    "the smartest move": "o movimento mais inteligente",
    "smartest move": "movimento mais inteligente",
    "high quality play": "jogada de alta qualidade",
    "quality play": "jogada de qualidade",
    "break above": "rompimento acima",
    "break below": "quebra abaixo",
    "intraday low": "fundo intradiário",
    "intraday high": "topo intradiário",
    "market open": "abertura do mercado",
    "the high and the low": "o topo e o fundo",
    "the high and low": "o topo e o fundo",
    "highs and lows": "topos e fundos",
    "the high is": "o topo é",
    "the high was": "o topo foi",
    "the low is": "o fundo é",
    "the low was": "o fundo foi",
    "long calls": "calls de compra",
    "highs": "topos",
    "lows": "fundos",
    # trade = operação; frases macro ("trade war") continuam "comercial"
    "trade war": "guerra comercial",
    "trade deficit": "déficit comercial",
    "trade deal": "acordo comercial",
    "trade agreement": "acordo comercial",
    "trade balance": "balança comercial",
    "trade talks": "negociações comerciais",
    "trade discipline": "disciplina de trade",
    # rompimentos e movimentos rápidos
    "breakdowns": "quebras",
    "breakdown": "quebra",
    "breakouts": "rompimentos",
    "breakout": "rompimento",
    "breaking out": "rompendo",
    "breaking down": "quebrando",
    "break out": "romper",
    "break down": "quebrar",
    "vomit back down": "despencar de volta",
    "vomit down": "despencar",
    "vomits": "despenca",
    "vomit": "despencar",
    "pop up to": "repique até",
    "the pop": "o repique",
    "a pop": "um repique",
    # fill = execução da ordem
    "my fill": "minha ordem executada",
    "good fill": "boa execução",
    "bad fill": "execução ruim",
    "partial fill": "execução parcial",
    "got filled": "fui executado",
    "getting filled": "sendo executado",
    "get filled": "ser executado",
    "get the fill": "ter a ordem executada",
    "got the fill": "tive a ordem executada",
    # opções: comprar/vender + call/put (instrumento fica em inglês)
    "selling calls": "vendendo calls",
    "selling puts": "vendendo puts",
    "buying calls": "comprando calls",
    "buying puts": "comprando puts",
    "sell a call": "vender uma call",
    "sell a put": "vender uma put",
    "buy a call": "comprar uma call",
    "buy a put": "comprar uma put",
    "sell the call": "vender a call",
    "sell the put": "vender a put",
    "buy the call": "comprar a call",
    "buy the put": "comprar a put",
    "sell calls": "vender calls",
    "sell puts": "vender puts",
    "buy calls": "comprar calls",
    "buy puts": "comprar puts",
    "I sold calls": "vendi calls",
    "I sold puts": "vendi puts",
    "I bought calls": "comprei calls",
    "I bought puts": "comprei puts",
    "call options": "opções de call",
    "put options": "opções de put",
    # stop (ordem) fica "stop"
    "stop loss": "stop loss",
    "hitting the stop": "batendo no stop",
    "hit the stop": "bater no stop",
    "my stop": "meu stop",
    "the stop": "o stop",
    # stopped/knocked out = stopado
    "I got stopped out": "fui stopado",
    "got stopped out": "fui stopado",
    "get stopped out": "ser stopado",
    "stopped me out": "me stopou",
    "stopped out": "stopado",
    "knocked out": "stopado",
    # bullish/bearish = comprador/vendedor
    "I'm bullish": "estou comprador",
    "I am bullish": "estou comprador",
    "I'm bearish": "estou vendedor",
    "I am bearish": "estou vendedor",
    "bullish": "comprador",
    "bearish": "vendedor",
    # up/down = lucro/prejuízo na operação
    "I'd be down": "eu estaria no negativo",
    "I would be down": "eu estaria no negativo",
    "I'm down": "estou no negativo",
    "I am down": "estou no negativo",
    "we're down": "estamos no negativo",
    "I'd be up": "eu estaria no lucro",
    "I would be up": "eu estaria no lucro",
    "I'm up": "estou no lucro",
    "I am up": "estou no lucro",
    "we're up": "estamos no lucro",
    # order = ordem (de negociação)
    "buy orders": "ordens de compra",
    "buy order": "ordem de compra",
    "sell orders": "ordens de venda",
    "sell order": "ordem de venda",
    "limit orders": "ordens limite",
    "limit order": "ordem limite",
    "market order": "ordem a mercado",
    "stop orders": "ordens stop",
    "stop order": "ordem stop",
    "order flow": "fluxo de ordens",
    "order book": "livro de ordens",
    "orders": "ordens",
    "shorts": "vendidos",
    "longs": "comprados",
    # ---- variantes frequentes em lives de trading ----
    # earnings: variantes de ASR / concordância
    "earning season": "temporada de balanços",
    "earning call": "teleconferência de resultados",
    "good earnings": "bons resultados",
    "bad earnings": "resultados ruins",
    "great earnings": "ótimos resultados",
    "strong earnings": "resultados fortes",
    "weak earnings": "resultados fracos",
    "earnings tonight": "balanço hoje à noite",
    "earnings tomorrow": "balanço amanhã",
    "earnings today": "balanço hoje",
    "earnings play": "jogada de balanço",
    # high/low singulares e compostos (topo/fundo)
    "higher low": "fundo ascendente",
    "lower high": "topo descendente",
    "equal low": "fundo igual",
    "equal high": "topo igual",
    "equal lows": "fundos iguais",
    "equal highs": "topos iguais",
    "double top": "topo duplo",
    "double bottom": "fundo duplo",
    "triple bottom": "fundo triplo",
    "triple top": "topo triplo",
    "recent low": "fundo recente",
    "recent high": "topo recente",
    "recent lows": "fundos recentes",
    "recent highs": "topos recentes",
    "previous low": "fundo anterior",
    "previous high": "topo anterior",
    "prior low": "fundo anterior",
    "prior high": "topo anterior",
    "current low": "fundo atual",
    "current high": "topo atual",
    "post market high": "topo do pós-mercado",
    "post market low": "fundo do pós-mercado",
    "pre-market high": "topo do pré-mercado",
    "pre-market low": "fundo do pré-mercado",
    "premarket high": "topo do pré-mercado",
    "premarket low": "fundo do pré-mercado",
    "took out the low": "rompeu o fundo",
    "took out the high": "rompeu o topo",
    "take out the low": "romper o fundo",
    "take out the high": "romper o topo",
    "10 year yield": "juros de 10 anos",
    "10-year yield": "juros de 10 anos",
    "2 year yield": "juros de 2 anos",
    "2-year yield": "juros de 2 anos",
    "30 year yield": "juros de 30 anos",
    "30-year yield": "juros de 30 anos",
    # lock in = travar o lucro
    "lock in": "travar",
    "locking in": "travando",
    "locked in": "travado",
    "locks in": "trava",
    "lock it in": "travar",
    "lock this in": "travar isso",
    "lock that in": "travar isso",
    "lock some in": "travar uma parte",
    # play = jogada; move = movimento
    "this play": "essa jogada",
    "that play": "essa jogada",
    "the play": "a jogada",
    "a play": "uma jogada",
    "new plays": "novas jogadas",
    "riskless play": "jogada sem risco",
    "interesting play": "jogada interessante",
    "large play": "jogada grande",
    "big play": "jogada grande",
    "easy play": "jogada fácil",
    "easiest play": "jogada mais fácil",
    "playing calls": "operando calls",
    "playing puts": "operando puts",
    "play this": "operar isso",
    "play it": "operar isso",
    "play that": "operar isso",
    "in a play": "numa jogada",
    "in the play": "na jogada",
    "the move": "o movimento",
    "this move": "esse movimento",
    "that move": "esse movimento",
    "a move": "um movimento",
    "a big move": "um movimento grande",
    "big move": "movimento grande",
    "the next move": "o próximo movimento",
    "next move": "próximo movimento",
    # pop = repique
    "every pop": "cada repique",
    "nice pop": "belo repique",
    "this pop": "esse repique",
    "that pop": "esse repique",
    "big pop": "repique grande",
    "weird pop": "repique estranho",
    "quick pop": "repique rápido",
    "little pop": "repiquezinho",
    "popped above": "repicou acima",
    "popped up": "repicou",
    "popped": "repicou",
    "pops": "repica",
    "popping": "repicando",
    "pop it above": "repicar acima de",
    "pop above": "repicar acima",
    "if it pops": "se repicar",
    "the pops": "os repiques",
    # calls/puts (nunca "chamadas/coloca")
    "the calls": "as calls",
    "the puts": "as puts",
    "my calls": "minhas calls",
    "my puts": "minhas puts",
    "call selling": "venda de calls",
    "put selling": "venda de puts",
    "call buying": "compra de calls",
    "put buying": "compra de puts",
    "sold puts": "vendeu puts",
    "sold calls": "vendeu calls",
    "bought puts": "comprou puts",
    "bought calls": "comprou calls",
    "leap calls": "calls LEAP",
    "leaps": "LEAPs",
    "call open interest": "open interest de calls",
    "put open interest": "open interest de puts",
    "calls or puts": "calls ou puts",
    "puts or calls": "puts ou calls",
    "calls and puts": "calls e puts",
    "call credits": "call credit spread",
    "put credits": "put credit spread",
    "call current spread": "call credit spread",
    # short/long soltos (ganham lookahead/lookbehind via _compile_entry)
    "a short": "uma venda",
    "a long": "uma compra",
    "the short": "a venda",
    "the long": "a compra",
    "this short": "essa venda",
    "this long": "essa compra",
    "that short": "essa venda",
    "that long": "essa compra",
    "decent short": "venda decente",
    "decent long": "compra decente",
    "nice short": "bela venda",
    "nice long": "bela compra",
    "was long": "estava comprado",
    "was short": "estava vendido",
    "is long": "está comprado",
    "is short": "está vendido",
    "are long": "estão comprados",
    "are short": "estão vendidos",
    "currently short": "vendido no momento",
    "currently long": "comprado no momento",
    "who is long": "quem está comprado",
    "who is short": "quem está vendido",
    "who's long": "quem está comprado",
    "who's short": "quem está vendido",
    "still long": "ainda comprado",
    "still short": "ainda vendido",
    "shorting": "vendendo",
    "shorted": "vendeu",
    "short it": "vender",
    "short the": "vender o",
    "short this": "vender isso",
    "short that": "vender isso",
    "long it": "comprar",
    "long the": "comprar o",
    "long this": "comprar isso",
    "long that": "comprar isso",
    "took a long": "comprei",
    "took a short": "vendi",
    "take a long": "comprar",
    "take a short": "vender",
    "I'm flat": "estou zerado",
    "I am flat": "estou zerado",
    "went flat": "zerou",
    "go flat": "zerar",
    "flat on the day": "zerado no dia",
    "kind of flat": "meio de lado",
    "is flat": "está de lado",
    "trading flat": "andando de lado",
    "stays flat": "fica de lado",
    # stop = stop
    "the stops": "os stops",
    "my stops": "meus stops",
    "this stop": "esse stop",
    "that stop": "esse stop",
    "your stop": "seu stop",
    "a stop": "um stop",
    "stops are": "stops estão",
    "stops at": "stops em",
    "stop is": "stop está",
    "stop was": "stop estava",
    "move my stop": "mover meu stop",
    "move this stop": "mover esse stop",
    "move the stop": "mover o stop",
    "trailing stop": "stop móvel",
    "trail the stop": "mover o stop",
    "knocking me out": "me stopando",
    "knocks me out": "me stopa",
    "knock me out": "me stopar",
    "I got knocked out": "fui stopado",
    "got knocked out": "fui stopado",
    "get knocked out": "ser stopado",
    # stock = ação
    "the stock": "a ação",
    "in the stock": "na ação",
    "this stock": "essa ação",
    "that stock": "essa ação",
    "a stock": "uma ação",
    "best stock": "melhor ação",
    "the stocks": "as ações",
    # gap (NÃO usar "preenchimento" - a regra "preenchimento"->"execução" mutila)
    "gap fill": "fechamento do gap",
    "gap fills": "fechamentos do gap",
    "fill the gap": "fechar o gap",
    "filled the gap": "fechou o gap",
    "fills the gap": "fecha o gap",
    "gap up": "gap de alta",
    "gap down": "gap de baixa",
    "upside gap": "gap de alta",
    "downside gap": "gap de baixa",
    "gapped up": "abriu em gap de alta",
    "gapped down": "abriu em gap de baixa",
    # trade verbo (conjugado pelo sujeito - o MT não conjuga token)
    "I trade": "eu opero",
    "you trade": "você opera",
    "we trade": "nós operamos",
    "they trade": "eles operam",
    "to trade": "operar",
    "I traded": "eu operei",
    "you traded": "você operou",
    "we traded": "operamos",
    "they traded": "operaram",
    "I'm trading": "estou operando",
    "you're trading": "você está operando",
    "we're trading": "estamos operando",
    "trading it": "operando isso",
    "trade it": "operar isso",
    "trade them": "operá-los",
    "trade with": "operar com",
    "trade small": "operar pequeno",
    "trade big": "operar grande",
    "trade both": "operar os dois",
    "don't trade": "não opere",
    "never trade": "nunca opere",
    "trade again": "operar de novo",
    "trade today": "operar hoje",
    "trade tomorrow": "operar amanhã",
    "trade futures": "operar futuros",
    "trade options": "operar opções",
    "trade stocks": "operar ações",
    "free trade": "trade sem risco",
    "free trades": "trades sem risco",
    "free trade agreement": "acordo de livre comércio",
    "a free trade agreement": "um acordo de livre comércio",
    # hit = pancada (prejuízo), nunca "sucesso"
    "take a hit": "levar uma pancada",
    "took a hit": "levou uma pancada",
    "taking a hit": "levando uma pancada",
    "takes a hit": "leva uma pancada",
    "a big hit": "uma pancada grande",
    "a hit on the account": "uma pancada na conta",
    "bit of a hit": "uma pancadinha",
    "a little bit of a hit": "uma pancadinha",
    # diversos
    "shout out to": "agradecimento ao",
    "shoutout to": "agradecimento ao",
    "shout out": "agradecimento",
    "shoutout": "agradecimento",
    "call it a day": "encerrar o dia",
    "called it a day": "encerrei o dia",
    "calling it a day": "encerrando o dia",
    "trap sellers": "vendedores presos",
    "trap buyers": "compradores presos",
    "trapped sellers": "vendedores presos",
    "trapped buyers": "compradores presos",
    "wick outs": "pavios",
    "wick out": "pavio",
    "wicked below": "deixou pavio abaixo",
    "wicked above": "deixou pavio acima",
    "wicked me out": "me stopou no pavio",
    "big prints": "ordens grandes",
    "big print": "ordem grande",
    "pump": "disparada",
    "pumping": "disparando",
    "pumped": "disparou",
    "a huge miss": "um número bem abaixo do esperado",
    "big miss": "número bem abaixo do esperado",
    "huge miss": "número bem abaixo do esperado",
    "rug everyone": "dão golpe em todo mundo",
    "rugs you": "te dá o golpe",
    "rug pull": "golpe",
    "rug pulled": "levou golpe",
    "on net": "no líquido",
    "scaling out": "realizando parcial",
    "scale out": "realizar parcial",
    "scaled out": "realizei parcial",
    "scaling in": "entrando aos poucos",
    "scale in": "entrar aos poucos",
    "scaled in": "entrei aos poucos",
    "chopping sideways": "andando de lado",
    "chopping around": "andando de lado",
    "grinding higher": "subindo devagar",
    "grinding up": "subindo devagar",
    "grinding lower": "caindo devagar",
    "grinding down": "caindo devagar",
    "on deck": "na fila",
    "the goat": "o maior de todos",
    "Piper fired": "o Piper disparou",
    "Piper is going to fire": "o Piper vai disparar",
    "Piper fires": "o Piper dispara",
    "the strike": "o strike",
    "strike price": "preço de strike",
    "at the strike": "no strike",
    "we are up big": "estamos bem no lucro",
    "we're up big": "estamos bem no lucro",
    "I'm up big": "estou bem no lucro",
    "spies": "SPY",
    "the spies": "o SPY",
    "subs": "inscritos",
    "DMing": "mandando DM",
    "options flows": "fluxo de opções",
    "the break of": "o rompimento de",
    "a break of": "um rompimento de",
    "this break at": "esse rompimento em",
    "the break at": "o rompimento em",
    "on this break": "nesse rompimento",
    "break and hold": "rompimento e sustentação",
    "hitting the bid": "batendo na oferta",
    "hit the bid": "bater na oferta",
    "in the money": "no lucro",
    "the flow": "o fluxo",
    "rip higher": "disparar",
    "ripping higher": "disparando",
    "ripped higher": "disparou",
    "rip through": "atravessar com força",
    "ripping through": "atravessando com força",
    "let's rip": "vamos mandar ver",
    "rip it": "mandar ver",
    "is ripping": "está disparando",
    "Max 7": "Mag 7",
    "Mag seven": "Mag 7",
    "tradifies": "Tradeify",
    "trade-of-eight": "Tradeify",
    "trade of eight": "Tradeify",
    "partially filled": "parcialmente executada",
    "fully filled": "totalmente executada",
    "gets filled": "é executada",
    "whatever gets filled": "o que for executado",
    "a fill": "uma execução",
    "the fill": "a execução",
    "average fill": "execução média",
    "my other fill": "minha outra execução",
    # variantes de ASR de nomes (o Whisper base erra estes com frequência)
    "Nasek": "Nasdaq",
    "Nasak": "Nasdaq",
    "Nazik": "Nasdaq",
    "Nasick": "Nasdaq",
    "Nazic": "Nasdaq",
    "Nazics": "Nasdaq",
    "NASEC": "Nasdaq",
    "Nasik": "Nasdaq",
    "Nazx": "Nasdaq",
    "Benazek": "Nasdaq",
    "Nazdeckistan": "Nasdaq",
    "VWOP": "VWAP",
    "V-Wop": "VWAP",
    "BWOP": "VWAP",
    "VWops": "VWAP",
    "V-WAP": "VWAP",
    "VWAPs": "VWAP",
    "Crowd Shrek": "CrowdStrike",
    "CrowdTrack": "CrowdStrike",
}

DEFAULT_FIX: Dict[str, str] = {
    # ORDEM = glossario.json: o mapa
    # "corrigir" é aplicado por ordem de inserção em fix() - uma chave
    # curta antes de uma longa que a contém como palavra inteira nunca
    # dispara. Editar SEMPRE os dois arquivos juntos e na MESMA ordem
    # (rodar o script de verificação de entradas mortas antes de commitar).
    'quebras falso': 'quebras falsas',
    'rompimentos falsificado': 'rompimentos falsos',
    'rompimento falsificado': 'rompimento falso',
    'rompimentos falso': 'rompimentos falsos',
    'um bela': 'uma bela',
    'um boa': 'uma boa',
    'um jogada': 'uma jogada',
    'ao minha': 'à minha',
    'é a peça': 'é a jogada',
    'trocam': 'operam',
    'couro cabeludo': 'scalp',
    'no aberto': 'na abertura',
    'o aberto': 'a abertura',
    'uma rompimento': 'um rompimento',
    'as tigelas': 'os compradores',
    'tigelas': 'compradores',
    'surto falso': 'rompimento falso',
    'surtos falsos': 'rompimentos falsos',
    'trocando': 'operando',
    'as fugas falsas': 'os rompimentos falsos',
    'a fuga falsa': 'o rompimento falso',
    'as fugas': 'os rompimentos',
    'a fuga': 'o rompimento',
    'fugas falsas': 'rompimentos falsos',
    'fuga falsa': 'rompimento falso',
    'fugas': 'rompimentos',
    'fuga': 'rompimento',
    'um mergulho': 'uma queda',
    'o mergulho': 'a queda',
    'esse mergulho': 'essa queda',
    'este mergulho': 'esta queda',
    'mergulhos': 'quedas',
    'mergulho': 'queda',
    'as baixas noturnas': 'os fundos da noite',
    'a baixa noturna': 'o fundo da noite',
    'as altas noturnas': 'os topos da noite',
    'a alta noturna': 'o topo da noite',
    'a noite baixa': 'o fundo da noite',
    'baixas noturnas': 'fundos da noite',
    'baixa noturna': 'fundo da noite',
    'altas noturnas': 'topos da noite',
    'alta noturna': 'topo da noite',
    'a alta de ontem': 'o topo de ontem',
    'a baixa de ontem': 'o fundo de ontem',
    'jogar, jogue': 'operar',
    'operar, opere': 'operar',
    'ordens preenchidas': 'ordens executadas',
    'ordem preenchida': 'ordem executada',
    'preenchimento': 'execução',
    'perda de parada': 'stop loss',
    'parado fora': 'stopado',
    'novos recordes': 'novos topos',
    'novo recorde': 'novo topo',
    'foi para os lados': 'andou de lado',
    'indo para os lados': 'andando de lado',
    'no alto': 'no topo',
    'alto e baixo': 'topo e fundo',
    'altos e baixos': 'topos e fundos',
    'o alto é': 'o topo é',
    'o baixo é': 'o fundo é',
    'o alto foi': 'o topo foi',
    'o baixo foi': 'o fundo foi',
    'no baixo': 'no fundo',
    "queda d'água": 'queda em cascata',
    'médias móveis está': 'médias móveis estão',
    'na resistência resistir': 'na resistência',
    'resistir a resistir': 'resistir',
    'comprando à resistência': 'comprando na resistência',
    'geometria da geometria': 'geometria do gráfico',
    'noivado': 'engajamento',
    'noivados': 'engajamentos',
    'seu comércio': 'seu trade',
    'peça de teatro': 'jogada',
    'tocar isso': 'jogar isso',
    'nasic': 'Nasdaq',
    'ish': 'e pouco',
    'ex-cadeira': 'ex-presidente',
    'uma cavalgada de convidados': 'um desfile de convidados',
    'um comércio verde': 'uma operação no lucro',
    'comércio verde': 'operação no lucro',
    'comércio vermelho': 'operação no prejuízo',
    'estou verde': 'estou no lucro',
    'correr em uma parede': 'bater numa parede',
    'correndo para dentro de uma parede': 'batendo numa parede',
    'correndo para dentro': 'batendo',
    'correu em uma parede': 'bateu numa parede',
    'cavalgada de convidados': 'desfile de convidados',
    'rasgou': 'mandou ver',
    'rasgaram': 'mandaram ver',
    'a resistência à resistência': 'a resistência',
    'resistência à resistência': 'resistência',
    'ir junto em': 'comprar em',
    'ir junto a': 'comprar a',
    'em o': 'no',
    'em a': 'na',
    'antes de o': 'antes do',
    'depois de o': 'depois do',
    'de o': 'do',
    'antes de a': 'antes da',
    'depois de a': 'depois da',
    'de a': 'da',
    'não o movimento mais inteligente': 'não é o movimento mais inteligente',
    'comércio pós-mercado': 'pós-mercado',
    'sessão de operação': 'pregão',
    'boa peça': 'boa jogada',
    'quero rasgar': 'quero mandar ver',
    'rasgar': 'mandar ver',
    'rasgando': 'mandando ver',
    'fora do portão': 'logo de cara',
    'todo o caminho para baixo': 'lá embaixo',
    'todo o caminho para cima': 'lá em cima',
    'todo o caminho até': 'até',
    'jogo de qualidade': 'jogada de qualidade',
    'movimento mais inteligente do cinema': 'movimento mais inteligente',
    'apoio': 'suporte',
    'apoios': 'suportes',
    'intervalo acima de': 'rompimento acima de',
    'para comprando': 'para comprar',
    'para vendendo': 'para vender',
    'filme mais inteligente': 'movimento mais inteligente',
    'peça de qualidade': 'jogada de qualidade',
    'peça de alta qualidade': 'jogada de alta qualidade',
    'baixa intradiária': 'fundo intradiário',
    'alta intradiária': 'topo intradiário',
    'cachoeiras': 'quedas em cascata',
    'cachoeira': 'queda em cascata',
    'muitas negociações': 'muitas operações',
    'muita negociação': 'muita operação',
    'pré-comercialização': 'pré-mercado',
    'execução médio': 'execução média',
    'o execução': 'a execução',
    'no execução': 'na execução',
    '% fora': '% de desconto',
    'de negociação': 'de operação',
    'a negociação': 'a operação',
    'chamadas longas': 'calls de compra',
    'estourou': 'repicou',
    'estourar': 'repicar',
    'estourado acima': 'repicou acima',
    'estourado': 'repicado',
    'otimista': 'comprador',
    'pessimista': 'vendedor',
    'negociando': 'operando',
    'negociam': 'operam',
    'negocia': 'opera',
    'negociar': 'operar',
    'negociei': 'operei',
    'negociou': 'operou',
    'Reserva Federal': 'Federal Reserve',
    'a Fed': 'o Fed',
    'Fed chair': 'presidente do Fed',
    'Fed cadeira': 'presidente do Fed',
    'cadeira do Fed': 'presidente do Fed',
    'taxas de juro': 'taxas de juros',
    'taxa de juro': 'taxa de juros',
    'milhares de milhões': 'bilhões',
    'mil milhões': 'bilhões',
    'milhar de milhões': 'bilhão',
    'as obrigações do tesouro': 'os títulos do Tesouro',
    'obrigações do tesouro': 'títulos do Tesouro',
    'as obrigações soberanas': 'os títulos soberanos',
    'obrigações soberanas': 'títulos soberanos',
    'registou': 'registrou',
    'registaram': 'registraram',
    'atingiu um máximo': 'bateu um topo',
    'atingiu novos topos': 'bateu novos topos',
    'atingiu novos fundos': 'bateu novos fundos',
    'atingiu um topo': 'bateu um topo',
    'atingiu o topo': 'bateu o topo',
    'atingiu um fundo': 'bateu um fundo',
    'atingiu o fundo': 'bateu o fundo',
    'atingiu o alvo': 'bateu o alvo',
    'atingiu o stop': 'bateu o stop',
    'mínimos históricos': 'fundos históricos',
    'máximos históricos': 'topos históricos',
    'máximas históricas': 'topos históricos',
    'mínimas históricas': 'fundos históricos',
    'máxima histórica': 'topo histórico',
    'mínima histórica': 'fundo histórico',
    'novas máximas': 'novos topos',
    'nova máxima': 'novo topo',
    'novas mínimas': 'novos fundos',
    'nova mínima': 'novo fundo',
    'máxima do dia': 'topo do dia',
    'mínima do dia': 'fundo do dia',
    'este comércio': 'este trade',
    'esse comércio': 'esse trade',
    'um bom comércio': 'um bom trade',
    'um comércio ruim': 'um trade ruim',
    'o meu comércio': 'o meu trade',
    'meu comércio': 'meu trade',
    'entrar no comércio': 'entrar no trade',
    'sair do comércio': 'sair do trade',
    'fechar o comércio': 'fechar o trade',
    'o pior comércio': 'o pior trade',
    'o melhor comércio': 'o melhor trade',
    'comércios de qualidade': 'trades de qualidade',
    'comércios': 'trades',
    'esta troca': 'este trade',
    'essa troca': 'esse trade',
    'uma boa troca': 'um bom trade',
    'uma troca': 'um trade',
    'a troca': 'o trade',
    'série de negociações': 'série de trades',
    'série de negócios': 'série de trades',
    'decisão Fed': 'decisão do Fed',
    'a aranha': 'o SPY',
    'aranhas': 'SPY',
    'aranha': 'SPY',
    'dois operações': 'duas operações',
    'neste operação': 'nesta operação',
    'nesse operação': 'nessa operação',
    'este operação': 'esta operação',
    'esse operação': 'essa operação',
    'se eu ter': 'se eu tiver',
    'o guerra comercial': 'a guerra comercial',
    'números déficit': 'números do déficit',
    'ordem de venda sentado': 'ordem de venda parada',
    'ordem de compra sentado': 'ordem de compra parada',
    'tomei a operação': 'entrei na operação',
    'tomar a operação': 'entrar na operação',
    'um quebra limpo': 'uma quebra limpa',
    'um quebra': 'uma quebra',
    'o quebra': 'a quebra',
    'vender chamadas': 'vender calls',
    'vender chamada': 'vender call',
    'comprar chamadas': 'comprar calls',
    'comprar chamada': 'comprar call',
    'antes batendo': 'antes de bater',
    'I fui': 'fui',
    'bater na parada': 'bater no stop',
    'bateu na parada': 'bateu no stop',
    'minha parada': 'meu stop',
    'nocauteados': 'stopados',
    'nocauteadas': 'stopadas',
    'nocauteado': 'stopado',
    'nocauteada': 'stopada',
    'vomitar': 'despencar',
    'vomitou': 'despencou',
    'vomitando': 'despencando',
    'assistir para o': 'ficar de olho no',
    'assistir para a': 'ficar de olho na',
    'um grande ordem': 'uma grande ordem',
    'um ordem': 'uma ordem',
    'o ordem': 'a ordem',
    'um operação': 'uma operação',
    'o operação': 'a operação',
    'um grande operação': 'uma grande operação',
    'pedidos de compra': 'ordens de compra',
    'pedido de compra': 'ordem de compra',
    'pedidos de venda': 'ordens de venda',
    'pedido de venda': 'ordem de venda',
    'fiz um pedido': 'coloquei uma ordem',
    'colocar um pedido': 'colocar uma ordem',
    'criadores de mercado': 'market makers',
    'criador de mercado': 'market maker',
    'fabricantes de mercado': 'market makers',
    'fabricante de mercado': 'market maker',
    'formadores de mercado': 'market makers',
    'SPY curto': 'venda no SPY',
    'trocar SPY': 'operar SPY',
    'spy': 'SPY',
    'espião': 'SPY',
    'crude oil': 'petróleo bruto',
    'oil': 'petróleo',
    'gold': 'ouro',
    'óleo bruto': 'petróleo bruto',
    'preços do óleo': 'preços do petróleo',
    'preço do óleo': 'preço do petróleo',
    'barril de óleo': 'barril de petróleo',
    'barris de óleo': 'barris de petróleo',
    'mercado de óleo': 'mercado de petróleo',
    'estoques de óleo': 'estoques de petróleo',
    'o óleo subiu': 'o petróleo subiu',
    'o óleo caiu': 'o petróleo caiu',
    'os touros': 'os compradores',
    'os ursos': 'os vendedores',
    'touros': 'compradores',
    'ursos': 'vendedores',
    'touro': 'comprador',
    'urso': 'vendedor',
    'trancado em': 'travado em',
    'trancar isso': 'travar isso',
    'tranquei': 'travei',
    'trancei': 'travei',
    'trancando': 'travando',
    'trancado': 'travado',
    'trancada': 'travada',
    'trancar': 'travar',
    'esta peça': 'essa jogada',
    'essa peça': 'essa jogada',
    'uma peça interessante': 'uma jogada interessante',
    'uma peça tão grande': 'uma jogada tão grande',
    'peça sem risco': 'jogada sem risco',
    'a peça da': 'a jogada da',
    'em uma peça': 'numa jogada',
    'nesta peça': 'nessa jogada',
    'tocando chamadas': 'operando calls',
    'esse pop': 'esse repique',
    'este pop': 'este repique',
    'um bom pop': 'um bom repique',
    'belo pop': 'belo repique',
    'o pop': 'o repique',
    'um pop': 'um repique',
    'saltamos acima': 'repicamos acima',
    'pulamos acima': 'repicamos acima',
    'nós surramos': 'repicamos',
    'eclodimos': 'rompemos',
    'rasgá-lo': 'mandar ver',
    'um rasgo': 'um repique',
    'rasga': 'manda ver',
    'ripássemos': 'disparássemos',
    'as paradas': 'os stops',
    'mover essa parada': 'mover esse stop',
    'essa parada': 'esse stop',
    'esta parada': 'este stop',
    'minhas paradas': 'meus stops',
    'me nocauteando': 'me stopando',
    'vou ter stopado': 'vou ser stopado',
    'ter stopado': 'ser stopado',
    'tenho stopado': 'fui stopado',
    'consegui stopado': 'fui stopado',
    'quebrar até mesmo': 'break even',
    'ponto de equilíbrio': 'break even',
    'no estoque': 'na ação',
    'o melhor estoque': 'a melhor ação',
    'melhor estoque': 'melhor ação',
    'meu estoque': 'minha ação',
    'o estoque': 'a ação',
    'execução da lacuna': 'fechamento do gap',
    'a lacuna': 'o gap',
    'lacunas': 'gaps',
    'lacuna': 'gap',
    'spread de crédito de chamada': 'call credit spread',
    'crédito de chamada': 'crédito de call',
    'as ligações são': 'as calls estão',
    'meu calls': 'minhas calls',
    'um calls': 'calls',
    'quem é longo': 'quem está comprado',
    'quem é curto': 'quem está vendido',
    'é curto atualmente': 'está vendido atualmente',
    'um curto-circuito': 'uma venda',
    'com falta hoje': 'vendido hoje',
    'as bolas': 'os compradores',
    'as compradores': 'os compradores',
    'os compradoras': 'os compradores',
    'comprador rompimento': 'rompimento comprador',
    'vendedor rompimento': 'rompimento vendedor',
    'posição comprador': 'posição compradora',
    'oportunidade comprador': 'oportunidade compradora',
    'tendência vendedor': 'tendência vendedora',
    'tendência comprador': 'tendência compradora',
    'o lucros': 'os lucros',
    'seu lucros': 'seus lucros',
    'bom lucros': 'bons lucros',
    'antes de lucros': 'antes do balanço',
    'depois de lucros': 'depois do balanço',
    'acima de lucros': 'acima do balanço',
    'ganhos de chamada': 'teleconferência de resultados',
    'reação dos ganhos': 'reação ao balanço',
    'ganhadores de ganhos': 'destaques do balanço',
    'temporada ganhando muito dinheiro': 'temporada de balanços',
    'você troca': 'você opera',
    'eu trocar': 'eu operar',
    'trocar com': 'operar com',
    'trocar o ETF': 'operar o ETF',
    'trocar de novo': 'operar de novo',
    'trocar comigo': 'operar comigo',
    'não troco': 'não opero',
    'os trocou': 'os operou',
    'trocá-lo': 'operá-lo',
    'trocar isso': 'operar isso',
    'trocar os dois': 'operar os dois',
    'trocar pequeno': 'operar pequeno',
    'vou trocar': 'vou operar',
    'quiser trocar': 'quiser operar',
    'quer trocar': 'quer operar',
    'trocar hoje': 'operar hoje',
    'trocar futuros': 'operar futuros',
    'trocar opções': 'operar opções',
    'trocar ações': 'operar ações',
    'negociação ativa': 'operação ativa',
    'negociação real': 'operação real',
    'à negociação': 'à operação',
    'o próximo comércio': 'o próximo trade',
    'próximo comércio': 'próximo trade',
    'o maior negócio': 'o maior trade',
    'negócio final': 'trade final',
    'comércio A plus': 'trade A plus',
    'copiadora de comércio': 'trade copier',
    'toda o trade': 'todo o trade',
    'toda o': 'todo o',
    'todo a': 'toda a',
    'neste quebra': 'nesta quebra',
    'nesse quebra': 'nessa quebra',
    'no quebra': 'na quebra',
    'do uma': 'de uma',
    'a ordens': 'as ordens',
    'todos os ordens': 'todas as ordens',
    'os ordens': 'as ordens',
    'tudo ordens': 'todas as ordens',
    'Minha ordens': 'Minhas ordens',
    'um execução': 'uma execução',
    'no pior execução': 'na pior execução',
    'execução preenchido': 'execução',
    'meu outro execução': 'minha outra execução',
    'for preenchido': 'for executado',
    'é preenchido': 'é executado',
    'parcialmente preenchido': 'parcialmente executada',
    'Completamente preenchido': 'totalmente executada',
    'Este recente fundos': 'Estes fundos recentes',
    'este recente alta': 'este topo recente',
    'recente alta': 'topo recente',
    'a alta dos': 'o topo dos',
    'um anterior alto': 'um topo anterior',
    'alta anterior': 'topo anterior',
    'um ponto baixo': 'um fundo',
    'esse é o baixo': 'esse é o fundo',
    'este baixo': 'este fundo',
    'o baixo nope': 'o fundo',
    'uma baixa igual': 'um fundo igual',
    'baixa igual': 'fundo igual',
    'o Alto': 'o topo',
    'do alto nível': 'do topo',
    'nível mais baixo': 'fundo ascendente',
    'ponto mais baixo': 'fundo ascendente',
    'mais alto baixo': 'fundo ascendente',
    'baixo mais alto': 'fundo ascendente',
    'rejeição mais baixa': 'rejeição no topo descendente',
    'confirmação mais baixa': 'confirmação de fundo ascendente',
    'Alta rompimento': 'rompimento do topo',
    'marca de água alta': "marca d'água",
    'na discórdia': 'no Discord',
    'da discórdia': 'do Discord',
    'discórdia': 'Discord',
    'mapa do livro': 'Bookmap',
    'piscinas escuras': 'dark pools',
    'conta de suporte': 'conta prop',
    'adereços': 'props',
    'Tradificar': 'Tradeify',
    'tradificação': 'Tradeify',
    'apenas fãs': 'OnlyFans',
    'os submarinos dele': 'os inscritos dele',
    'submarinos': 'inscritos',
    'no córrego': 'no stream',
    'esse fluxo': 'esse stream',
    'suas correntes': 'seus streams',
    'flâmula': 'streamer',
    'de retirada': 'de drawdown',
    'deslizamento': 'slippage',
    'essa configuração': 'esse setup',
    'da instalação': 'do setup',
    'os seus médicos': 'seus DMs',
    'desmaiando todos': 'mandando DM para todos',
    'na greve': 'no strike',
    'frente executando': 'front running',
    'a frente me executando': 'front running em mim',
    'eu chamo isso de um dia': 'encerro o dia',
    'chamá-lo um dia': 'encerrar o dia',
    'gritar com': 'agradecimento ao',
    'grite para': 'agradecimento ao',
    'a raça de qualquer um': 'a corrida está aberta',
    'pegou uma batida': 'levou uma pancada',
    'um pouco do sucesso': 'um pouco da pancada',
    'um pouco de sucesso': 'uma pequena pancada',
    'um grande sucesso na conta': 'uma pancada grande na conta',
    'enorme senhorita': 'número bem abaixo do esperado',
    'É o bode': 'É o maior de todos',
    'no convés': 'na fila',
    'triplicando o fundo do poço': 'fundo triplo',
    'cortando lateralmente': 'andando de lado',
    'vendedores de armadilhas': 'vendedores presos',
    'compradores de armadilhas': 'compradores presos',
    'vendedores armadilha': 'vendedores presos',
    'um monte de mechas': 'um monte de pavios',
    'grande impressão': 'ordem grande',
    'escalando e escalando': 'realizando parcial',
    'escalar em escala': 'entrar aos poucos',
    'Piper despediu-se': 'o Piper disparou',
    'Piper vai atirar': 'o Piper vai disparar',
    '10 anos de produção': 'juros de 10 anos',
    'de produção caíram': 'os juros caíram',
    'na net': 'no líquido',
    'na internet, mas': 'no líquido, mas',
    'Pequenos escalpos': 'Pequenos scalps',
    'escalpos': 'scalps',
    'gerenciamento de pulso': 'gerenciamento de risco',
    'uma pequena crosta': 'um pequeno scalp',
    'O fluxo de náuseas': 'O fluxo do Nasdaq',
    'em NQQ': 'no NQ',
    'no QN': 'no NQ',
    'QFPs': 'MFFs',
    'QFP': 'MFF',
    'conta lúcida': 'conta Lucid',
    'Esqueci-me': 'Esqueci',
    'a partilha': 'o compartilhamento',
    'da Fed': 'do Fed',
    'Chefe Fed': 'chefe do Fed',
    'presidente Fed': 'presidente do Fed',
    'mergulharmos': 'cairmos',
    'mergulhe': 'caia',
    'mergulham': 'caem',
    'que mergulha': 'que cai',
    'dar outro queda': 'dar outra queda',
    'outro queda': 'outra queda',
    'Estamos no topo hoje': 'Estamos bem no lucro hoje',
    'no dinheiro': 'no lucro',
    'o colapso': 'a quebra',
    'um colapso': 'uma quebra',
    'colapso de': 'quebra de',
    'colapsos de': 'quebras de',
}

_TOKEN = "XPROTECTED{}X"
# case-insensitive: o Opus-MT às vezes devolve o token em minúsculas
_TOKEN_RE = re.compile(r"XPROTECTED(\d+)X", re.IGNORECASE)

# Siglas que DEVEM ser traduzidas (não são tickers/empresas)
_CAPS_STOP = {"I", "A", "OK", "TV", "US", "USA", "UK", "EU", "GDP", "AI",
              "PM", "AM", "ET", "EST", "CET", "Q1", "Q2", "Q3", "Q4",
              "OIL", "GOLD", "NEWS", "CEO", "IPO"}

# Palavras capitalizadas comuns do inglês que não são nomes próprios.
# Palavra capitalizada fora desta lista é tratada como nome (empresa etc.)
# e mantida em inglês, inclusive no início da frase.
_TITLE_STOP = {
    # calendário
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "Today", "Tomorrow", "Yesterday", "Christmas",
    # pronomes/artigos/conjunções/preposições/auxiliares
    "The", "This", "That", "These", "Those", "It", "He", "She", "They",
    "We", "You", "My", "Your", "Our", "His", "Her", "Their", "But",
    "And", "Or", "So", "If", "When", "What", "Why", "How", "Who",
    "Where", "Which", "Because", "While", "After", "Before", "During",
    "With", "Without", "From", "Into", "About", "Against", "Between",
    "Over", "Under", "Above", "Below", "For", "Not", "No", "Yes",
    "In", "On", "At", "As", "By", "To", "Up", "Out", "Down", "Off",
    "All", "Any", "Some", "Most", "More", "Less", "Few", "Both",
    "Each", "Every", "One", "Two", "Three", "Four", "Five", "Ten",
    "First", "Second", "Third", "Last", "Next", "Other", "Another",
    "Same", "Such", "Many", "Much", "Once", "Twice", "Almost",
    "Already", "Soon", "Later", "Early", "Late", "Back", "Away",
    "Is", "Are", "Was", "Were", "Be", "Been", "Do", "Does", "Did",
    "Have", "Has", "Had", "Will", "Would", "Can", "Could", "Should",
    "Must", "May", "Might", "Let", "Lets",
    # advérbios/interjeições comuns de fala
    "Now", "Then", "There", "Here", "Just", "Really", "Maybe", "Again",
    "Also", "Still", "Even", "Very", "Too", "Only", "Never", "Always",
    "Guys", "Okay", "Ok", "Yeah", "Well", "Right", "Alright", "Look",
    "Listen", "Watch", "Remember", "People", "Everyone", "Everybody",
    "Something", "Nothing", "Anything", "Everything",
    "Sorry", "Thanks", "Thank", "Please", "Hello", "Hey", "Hi",
    "Welcome", "Good", "Great", "Nice", "Beautiful", "Perfect",
    "Exactly", "Absolutely", "Probably", "Actually", "Anyway",
    "Whatever", "Sure", "Fine", "Wow", "Oh", "Ah", "Um", "Uh",
    "Morning", "Afternoon", "Evening", "Tonight", "Careful", "Easy",
    "Wait", "Hold", "Boom", "Bang", "Damn", "Jesus", "Man", "Dude",
    "Folks", "Chat", "Guy", "Lady", "Ladies", "Gentlemen",
    "Like", "Basically", "Literally", "Honestly", "Obviously",
    "Seriously", "Anyways", "Kind", "Sort", "Kinda", "Sorta",
    "Gonna", "Wanna", "Gotta", "Lot", "Lots", "Bit", "Little",
    "Big", "Huge", "Crazy", "Insane", "Wild", "Sweet", "Cool",
    # verbos comuns em início de frase (imperativo/gerúndio)
    "Buy", "Sell", "Take", "Keep", "Come", "Go", "Going", "Getting",
    "Get", "Wait", "Stop", "Start", "Think", "See", "Say", "Make",
    "Trading", "Buying", "Selling", "Looking", "Coming", "Moving",
    # substantivos de mercado que DEVEM ser traduzidos
    "Gold", "Oil", "Crude", "Silver", "Copper", "Stocks", "Stock", "Bonds",
    "Bond", "Markets", "Market", "Traders", "Trader", "Futures",
    "Options", "Banks", "Bank", "Tech", "Energy", "Buyers", "Sellers",
    "Volume", "Price", "Prices", "Support", "Resistance", "Bulls",
    "Bears", "Rate", "Rates", "Inflation", "Earnings", "News",
    "American", "Americans", "America", "God",
}

# pt-PT "estar a + infinitivo" -> gerúndio pt-BR ("está a subir" -> "está subindo")
_GERUND_RE = re.compile(
    r"\b(está|estão|estava|estavam|estou|estamos|continua|continuam"
    r"|continuo|segue|seguem) a ([a-záéíóúâêôãõç]+[aei]r)(-se)?\b",
    re.IGNORECASE)

# "⁇" (U+2047) é o <unk> do CT2 no lugar de acentos que o tokenizer não
# reconheceu. Mapa ordenado de casos frequentes; o resto vira espaço.
_UNK_MAP = [
    (re.compile(r"(?i)⁇\s*s vezes"), "Às vezes"),
    (re.compile(r"(?i)⁇\s*s (?=\d)"), "às "),
    (re.compile(r"(?i)⁇\s*timo"), "Ótimo"),
    (re.compile(r"(?i)⁇\s*tima"), "Ótima"),
    (re.compile(r"(?i)⁇\s*ndice"), "Índice"),
    (re.compile(r"(?i)⁇\s*medida"), "À medida"),
    (re.compile(r"(?i)conseq\s*⁇\s*ências"), "consequências"),
    (re.compile(r"(?i)conseq\s*⁇\s*ência"), "consequência"),
    (re.compile(r"(?i)⁇\s*caro"), "Ícaro"),
    (re.compile(r"(?i)⁇\s*ltimo"), "Último"),
    (re.compile(r"(?i)⁇\s*ltima"), "Última"),
    (re.compile(r"(?i)⁇\s*nico"), "Único"),
    (re.compile(r"(?i)⁇\s*nica"), "Única"),
    (re.compile(r"(?i)⁇\s*rea"), "Área"),
    (re.compile(r"(?i)⁇\s*gua"), "Água"),
    (re.compile(r"(?i)⁇\s*s(?=[a-záéíóú])"), "às "),
    (re.compile(r"⁇\s*(?=\d)"), "às "),
]


def _fix_unk(text: str) -> str:
    """Limpa o "⁇" (U+2047, <unk> do CT2) no lugar de acentos não reconhecidos."""
    if "⁇" not in text:
        return text
    for regex, repl in _UNK_MAP:
        text = regex.sub(repl, text)
    # o que sobrar do token vira espaço (não pontuação, não corta a frase)
    text = re.sub(r"\s*⁇\s*", " ", text)
    return re.sub(r"  +", " ", text).strip()

_CAPS_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,5}\b")
_TITLE_SEQ_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_SENT_START_RE = re.compile(r"(?:^|[.!?…]\s+|[\"'“‘(]\s*)$")

# ------------------------------------------------- log de qualidade da tradução
# Arquivo separado (traducoes.log, na raiz) com as 4 etapas de cada frase:
# o inglês recebido, o texto mascarado que o Argos vê, a saída crua do Argos
# e o português final falado. Serve para auditar a qualidade da tradução.
_QLOG = logging.getLogger("tradutor.traducoes")
_QLOG.propagate = False  # arquivo próprio; não duplica no tradutor.log
_QLOG.setLevel(logging.INFO)  # independe da configuração do logger raiz


# Liga/desliga a gravação de traducoes.log (config.json: "gravar_log").
# Desligado, apply() nem monta a mensagem: custo zero.
_QLOG_ENABLED = True


def set_quality_log(enabled: bool) -> None:
    """Define se traducoes.log é gravado (chamado no boot, a partir do config)."""
    global _QLOG_ENABLED
    _QLOG_ENABLED = bool(enabled)


def _quality_logger() -> logging.Logger:
    if not _QLOG.handlers:
        try:
            from logging.handlers import RotatingFileHandler
            root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            handler = RotatingFileHandler(
                os.path.join(root, "traducoes.log"), maxBytes=2_000_000,
                backupCount=2, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            _QLOG.addHandler(handler)
        except Exception:
            _QLOG.addHandler(logging.NullHandler())
    return _QLOG


_MAX_AUTO_MASKS = 12


class Glossary:
    """Blindagem de termos na ida e correções determinísticas na volta."""

    def __init__(self, path: str = "glossario.json") -> None:
        protect, fix, translate = DEFAULT_PROTECT, DEFAULT_FIX, DEFAULT_TRANSLATE
        tickers = DEFAULT_TICKERS
        auto = True
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                protect = list(data.get("proteger", protect))
                fix = dict(data.get("corrigir", fix))
                translate = dict(data.get("traduzir", translate))
                tickers = dict(data.get("tickers", DEFAULT_TICKERS))
                auto = bool(data.get("auto_proteger_nomes", True))
                log.info("glossário: %d protegidos, %d traduções fixas, "
                         "%d tickers, %d correções, auto-nomes=%s (%s)",
                         len(protect), len(translate), len(tickers),
                         len(fix), auto, path)
            except Exception:
                log.exception("glossario.json inválido, usando padrão")
        # proteger => restaura o texto original; traduzir => restaura o pt fixo
        entries: List[Tuple[str, str]] = (
            [(t, "") for t in protect if t.strip()]
            + [(t, pt) for t, pt in translate.items() if t.strip() and pt])
        entries.sort(key=lambda e: len(e[0]), reverse=True)  # frase > palavra
        self._protect_re = [(_compile_entry(t), pt) for t, pt in entries]
        # tickers: case-SENSITIVE, expandidos para o nome da empresa
        self._ticker_re = [
            (re.compile(r"\b" + re.escape(t) + r"\b"), name)
            for t, name in sorted(tickers.items(),
                                  key=lambda e: len(e[0]), reverse=True)
            if t.strip() and name]
        self._fix = fix
        # pré-compilado (512+ entradas estouram o re._MAXCACHE de 512 e
        # `re.sub` com string passa a recompilar por chamada (~40ms/frase))
        self._fix_re = [
            (re.compile(r"\b" + re.escape(wrong) + r"\b", re.IGNORECASE), right)
            for wrong, right in fix.items()]
        self._auto = auto

    def mask(self, text: str) -> Tuple[str, List[str]]:
        """Substitui termos por tokens; devolve (texto, restaurações).

        Termos protegidos restauram o texto original; termos com tradução
        fixa restauram diretamente o português definido no glossário.
        """
        found: List[str] = []

        def _sub_time(m: re.Match) -> str:
            h, ap = int(m.group(1)), m.group(2).lower()
            if ap == "p":
                txt = ("meio-dia" if h == 12
                       else f"{h} da tarde" if h <= 5 else f"{h} da noite")
            else:
                txt = "meia-noite" if h == 12 else f"{h} da manhã"
            found.append(txt)
            return _TOKEN.format(len(found) - 1)

        text = _TIME_RE.sub(_sub_time, text)
        for regex, fixed_pt in self._protect_re:
            def _sub(match: re.Match, _pt: str = fixed_pt) -> str:
                found.append(_pt if _pt else match.group(0))
                return _TOKEN.format(len(found) - 1)

            text = regex.sub(_sub, text)

        def _sub_trade(m: re.Match) -> str:
            found.append(m.group(3).lower())
            return f"{m.group(1)}{m.group(2)}{_TOKEN.format(len(found) - 1)}"

        text = _TRADE_NOUN_RE.sub(_sub_trade, text)

        # high/low com determinante -> topo/fundo (mesmo esquema do trade: o
        # determinante fica FORA do token porque é o MT quem concorda o
        # artigo). O lookahead positivo em `_HILO_RE` evita "a high
        # probability setup"/"the high side" (high como adjetivo, não topo).
        def _sub_hilo(m: re.Match) -> str:
            pt = _HILO_PT[m.group(3).lower()]
            found.append(pt)
            return f"{m.group(1)}{m.group(2)}{_TOKEN.format(len(found) - 1)}"

        text = _HILO_RE.sub(_sub_hilo, text)
        for regex, name in self._ticker_re:
            def _sub_ticker(match: re.Match, _n: str = name) -> str:
                found.append(_n)
                return _TOKEN.format(len(found) - 1)

            text = regex.sub(_sub_ticker, text)
        if self._auto:
            text = self._auto_mask(text, found)
        return text, found

    def _auto_mask(self, text: str, found: List[str]) -> str:
        """Blindagem automática de nomes próprios prováveis (empresas, tickers).

        1. Siglas em MAIÚSCULAS de 2-6 letras (AAPL, TSLA) fora de `_CAPS_STOP`.
        2. Sequências Capitalizadas no meio da frase ("Nvidia", "Goldman
           Sachs"); palavra única no INÍCIO da frase não é mascarada (pode
           ser palavra comum), e os termos de `_TITLE_STOP` são ignorados.
        """

        def _caps(match: re.Match) -> str:
            w = match.group(0)
            if w in _CAPS_STOP or len(found) >= _MAX_AUTO_MASKS:
                return w
            found.append(w)
            return _TOKEN.format(len(found) - 1)

        text = _CAPS_RE.sub(_caps, text)

        out: list = []
        last = 0
        for m in _TITLE_SEQ_RE.finditer(text):
            if len(found) >= _MAX_AUTO_MASKS:
                break
            words = m.group(0).split()
            # descarta stopwords das pontas ("The Nvidia rally" -> "Nvidia")
            while words and words[0] in _TITLE_STOP:
                words.pop(0)
            while words and words[-1] in _TITLE_STOP:
                words.pop()
            if not words:
                continue
            span_txt = " ".join(words)
            start = m.start() + m.group(0).find(words[0])
            end = start + len(span_txt)
            found.append(span_txt)
            out.append(text[last:start])
            out.append(_TOKEN.format(len(found) - 1))
            last = end
        out.append(text[last:])
        return "".join(out)

    def unmask(self, text: str, found: List[str]) -> str:
        """Restaura os tokens com a grafia original capturada na ida."""

        def _sub(match: re.Match) -> str:
            i = int(match.group(1))
            return found[i] if i < len(found) else ""

        # token com "ED"/"D" grudado antes do número ("XPROTECTEDED0X")
        text = re.sub(r"(?i)XPROTECTED(?:ED|D)+(\d+)X", r"XPROTECTED\1X", text)
        # normaliza token com "X" duplicado pelo tradutor (XPROTECTED0XX)
        text = re.sub(r"(?i)(XPROTECTED\d+X)X+(?!PROTECTED)", r"\1", text)
        # dígito colado ao token ("XPROTECTED0XX2" -> "XPROTECTED0X"): eco do
        # MT, descarta o dígito colado
        text = re.sub(r"(?i)(XPROTECTED\d+X)\d+", r"\1", text)
        # tokens colados ("…0XXPROTECTED1X") ganham espaço; token ecoado
        # 2x pelo MT colapsa para um só
        text = re.sub(r"(?i)(XPROTECTED\d+X)(?=XPROTECTED)", r"\1 ", text)
        text = re.sub(r"(?i)(XPROTECTED(\d+)X)(\s*XPROTECTED\2X)+", r"\1", text)
        text = _TOKEN_RE.sub(_sub, text)
        # tolera token mutilado pelo tradutor (raro): remove restos
        text = re.sub(r"X?PROTECTED\d*X*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bX{2,}\b", "", text)
        # resíduos de token restantes: "ED0X"/"D0X" soltos (mas não quando
        # grudados a preço/dígito, ex.: "$0X" legítimo). Prefixo "ED"/"D"
        # OBRIGATÓRIO: sem ele apagaria alavancagem legítima ("3X ETF", "5X").
        text = re.sub(r"(?<![\d$.,])\b(?:ED|D)\d{1,2}X\b", "", text)
        # "900x" (eco de token) -> "900"; "10x" (alavancagem) é legítimo e fica
        text = re.sub(r"\b(\d{3,})x\b", r"\1", text)
        return re.sub(r"  +", " ", text).strip()

    def fix(self, text: str, source: str = "") -> str:
        """Correções pós-tradução (rede de segurança), por palavra inteira.

        `source` é o inglês de origem: as regras de "anos fantasma" (o
        Opus-MT lê número solto como idade) só rodam quando o EN não
        menciona idade de verdade ("years", "yrs", "year-old"...).
        """
        text = _fix_unk(text)
        for regex, right in self._fix_re:
            text = regex.sub(right, text)
        # gerúndio: "está a acumular(-se)" -> "está acumulando(-se)"
        text = _GERUND_RE.sub(
            lambda m: f"{m.group(1)} {m.group(2)[:-1]}ndo{m.group(3) or ''}",
            text)
        # "é subindo" / "são caindo" -> "está subindo" / "estão caindo"
        # (mas não "Sexta-feira é quando temos..." -> "está quando")
        text = re.sub(r"\bé (?!quando\b)([a-záéíóúâêôãõç]+ndo)\b", r"está \1", text)
        text = re.sub(r"\bsão (?!quando\b)([a-záéíóúâêôãõç]+ndo)\b", r"estão \1", text)
        # O Opus-MT lê número solto como idade ("at 57" -> "aos 57 anos",
        # "about 60" -> "de 60 anos"). No contexto de trade número é
        # preço/região, nunca idade. Só remove o "anos" fantasma quando o
        # EN de origem não fala de idade de verdade ("I've known him for
        # 25 years" tem que MANTER "25 anos").
        if not re.search(r"(?i)\b(?:years?|yrs?|yr|year-old|decade)\b|-year\b",
                          source):
            text = re.sub(r"\baos (\d+(?:[.,]\d+)?) anos\b", r"no \1", text)
            text = re.sub(r"\bde (\d+(?:[.,]\d+)?) anos\b", r"do \1", text)
            text = re.sub(r"\b(\d+(?:[.,]\d+)?) anos\b", r"\1", text)
            # o MT lê "seventies"/"eighties" etc. como idade
            text = re.sub(
                r"\b(?:um |1 )?(vinte|trinta|quarenta|cinquenta|sessenta"
                r"|setenta|oitenta|noventa) anos\b", r"\1", text,
                flags=re.IGNORECASE)
        # preço lido como idade mesmo com "aos" ("Gostei aos 50" -> "Gostei em 50")
        text = re.sub(r"\baos (\d+(?:[.,]\d+)?)\b(?! anos)", r"em \1", text)
        # posição ordinal lida como número: "na 3a posição" -> "em 3"
        text = re.sub(r"\bna (\d+)a posição\b", r"em \1", text)
        # ordinal solto mal formatado: "o 360o" -> "o 360"
        text = re.sub(r"\b([oa]|na|no) (\d+)[oa]\b", r"\1 \2", text)
        # preço lido como hora: "às 900" -> "em 900" (hora nunca tem 3+ dígitos)
        text = re.sub(r"\bàs (\d{3,})\b", r"em \1", text)
        # "160 level" -> "160 graus" (o MT lê nível de preço como temperatura)
        if re.search(r"(?i)\blevel\b", source):
            text = re.sub(r"\b(\d+) graus\b", r"\1", text)
        # "break above 280" mascarado vira "rompimento acima 280" (sem o "de")
        text = re.sub(r"\b(acima|abaixo) (\d)", r"\1 de \2", text)
        text = re.sub(r"\b[Nn]o (\d{1,2}) da (manhã|tarde|noite)", r"às \1 da \2", text)
        text = re.sub(r"resistência a (\d)", r"resistência em \1", text)
        # eco de número do MT: "em 260 a 260" -> "em 260"
        text = re.sub(r"\b(\d+(?:[.,]\d+)?) a \1\b", r"\1", text)
        # "a low at 440" -> "um mínimo de 440"; no gráfico é fundo EM tal preço
        text = re.sub(r"\b[Uu]m mínimo de (\d)", r"um fundo em \1", text)
        text = re.sub(r"\b[Uu]ma mínima de (\d)", r"um fundo em \1", text)
        text = re.sub(r"\b[Uu]m máximo de (\d)", r"um topo em \1", text)
        text = re.sub(r"\b[Uu]ma máxima de (\d)", r"um topo em \1", text)
        # concordância: as correções acima trocam a palavra, não o artigo
        # ("um mergulho" -> "um queda"). Aqui o artigo acompanha o feminino.
        text = re.sub(
            r"\b([Uu]m|[Oo]s?|[Ee]stes?|[Ee]sses?|[Aa]quele|[Mm]eus?|[Ss]eus?"
            r"|[Nn]ossos?|[Tt]odos)"
            r" (quedas?|jogadas?|operaç(?:ão|ões)|paredes?|ordens?|resistências?"
            r"|médias?|quebras?|vendas?|compras?|posiç(?:ão|ões)|aberturas?"
            r"|reaç(?:ão|ões)|teleconferências?|temporadas?|execuç(?:ão|ões)"
            r"|calls|puts|aç(?:ão|ões))\b",
            _fem_article, text)
        text = re.sub(
            r"\b([Uu]mas?|[Aa]s|[Ee]stas?|[Ee]ssas?|[Aa]quelas?|[Mm]inhas?|[Ss]uas?"
            r"|[Nn]ossas?|[Tt]odas)"
            r" (fundos?|topos?|trades?|rompimentos?|movimentos?|pregões?|pregão"
            r"|balanços?|repiques?|pivôs?|suportes?|lucros?|overnight"
            r"|stops?|gaps?|setups?|drawdowns?|pavios?|strikes?|streams?"
            r"|break even|breakeven)\b",
            _masc_article, text)
        # "todos" + artigo já corrigido pela regra acima ("todos os ordens"
        # -> "todos as ordens": só o "os" mais próximo do substantivo foi
        # trocado) - alinha o "todos"/"todas" externo ao artigo interno.
        text = re.sub(r"\b[Tt]odos( as )\b", lambda m: (
            "Todas" if m.group(0)[0].isupper() else "todas") + m.group(1), text)
        text = re.sub(r"\b[Tt]odas( os )\b", lambda m: (
            "Todos" if m.group(0)[0].isupper() else "todos") + m.group(1), text)
        # concordância de adjetivo com o substantivo feminino restaurado
        text = _ADJ_NOUN_RE.sub(_agree_adj_noun, text)
        text = _NOUN_ADJ_RE.sub(_agree_noun_adj, text)
        # "trade" que escapou da máscara e virou "comércio", mas o sentido
        # macro ("o comércio internacional") não pode ser tocado.
        text = re.sub(
            r"\b([Oo]|[Uu]m|[Ee]sse|[Ee]ste) comércio\b"
            r"(?! (?:internacional|exterior|externo|global|mundial|eletrônico|varejista|local))",
            r"\1 trade", text)
        return text

    def apply(self, translate_fn, text: str) -> str:
        """mask -> traduz -> unmask -> fix, num passo só."""
        masked, found = self.mask(text)
        out = translate_fn(masked)
        final = self.fix(self.unmask(out, found), source=text)
        # restauração no início da frase pode vir minúscula ("fui stopado…")
        if final and final[0].islower():
            final = final[0].upper() + final[1:]
        if not _QLOG_ENABLED:
            return final
        try:
            tokens = "; ".join(f"{i}={t!r}" for i, t in enumerate(found))
            _quality_logger().info(
                "\nEN      | %s\nMASCARA | %s\nTOKENS  | %s\n"
                "MT      | %s\nPT      | %s\n",
                text, masked, tokens or "(nenhum)", out, final)
        except Exception:
            log.debug("falha no log de qualidade", exc_info=True)
        return final
