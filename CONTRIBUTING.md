# Como contribuir

Obrigada pelo interesse em contribuir com o Tradutor Simultâneo. Há três
formas de ajudar, descritas abaixo. Veja também a seção "Contribuindo" do
[`README.md`](README.md#contribuindo).

## Novo tema de glossário

O app organiza o jargão por tema: cada arquivo em `glossarios/<nome>.json`
cobre um domínio (o padrão é `trading`, jargão de mercado financeiro). Para
propor um tema novo (medicina, games, futebol, o que for):

1. Copie `glossarios/geral.json` para `glossarios/<seu-tema>.json` (nome
   curto, minúsculo, sem espaços ou acentos). Ele já vem com
   `regras_de_mercado` em `false`; mantenha assim, a menos que o seu tema
   também seja sobre mercado financeiro.
2. Preencha os campos `nome` (rótulo mostrado no combo "Tema" da interface) e
   `descricao`.
3. Adicione as regras nas quatro listas: `proteger` (termos que ficam em
   inglês), `tickers` (se fizer sentido no seu domínio; senão deixe `{}`),
   `traduzir` (traduções fixas) e `corrigir` (correções aplicadas depois da
   tradução).
4. Rode `.venv\Scripts\python.exe tests\test_glossary.py`: ele valida todos
   os arquivos em `glossarios/`, incluindo o seu.
5. Abra um Pull Request só com esse arquivo.

Dicas: comece pela lista `proteger` (o que deve ficar em inglês, como nomes
próprios e siglas sem tradução natural) e pela lista `corrigir` (onde o
tradutor erra sistematicamente). Ligue `gravar_log` (`true`) no
`config.json`, deixe o app rodando por alguns minutos ouvindo conteúdo do seu
domínio e use o `traducoes.log` gerado para achar os erros. Não inclua dados
pessoais ou de terceiros no arquivo do tema.

## Correções no tema trading

Para propor uma correção nas regras já existentes de `glossarios/trading.json`,
abra um Pull Request com o trecho do `traducoes.log` que mostra o problema (o
texto em inglês ouvido e o texto final em português falado) e a regra
proposta para corrigir.

## Mudanças no código

Antes de abrir o Pull Request, rode os testes offline listados no
[`README.md`](README.md#testes):

```powershell
.venv\Scripts\python.exe tests\test_glossary.py
.venv\Scripts\python.exe tests\test_segmenter.py
.venv\Scripts\python.exe tests\test_audio_io.py --offline
.venv\Scripts\python.exe tests\test_gui.py
```

Mantenha docstrings e comentários em português do Brasil, no estilo do
projeto (explicando o porquê da decisão, não só o quê). Não use travessões
("--"); prefira vírgula, dois-pontos, parênteses ou ponto.

Sugestões e problemas: abra uma Issue no repositório.
