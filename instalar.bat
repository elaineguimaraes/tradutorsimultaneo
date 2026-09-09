@echo off
setlocal
rem ============================================================
rem  Tradutor Simultaneo - instalador para um computador novo
rem  Faz tudo sozinho: Python, dependencias, VB-CABLE e modelos.
rem ============================================================
cd /d "%~dp0"
echo.
echo === Tradutor Simultaneo - instalacao ===
echo.

rem --- 1) Python 3 -------------------------------------------------
py -3 --version >nul 2>&1
if errorlevel 1 (
    echo [1/5] Python nao encontrado. Instalando via winget...
    winget install -e --id Python.Python.3.13 --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo ERRO: instale o Python manualmente em https://www.python.org/downloads/
        echo       ^(marque "Add python.exe to PATH"^) e rode este instalador de novo.
        pause & exit /b 1
    )
    echo Feche esta janela e rode instalar.bat DE NOVO para continuar.
    pause & exit /b 0
)
echo [1/5] Python OK.

rem --- 2) Ambiente virtual + dependencias --------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [2/5] Criando ambiente virtual...
    py -3 -m venv .venv || (echo ERRO ao criar o venv & pause & exit /b 1)
)
echo [2/5] Instalando dependencias (pode demorar varios minutos)...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo ERRO no pip install & pause & exit /b 1)

rem --- 3) VB-CABLE -------------------------------------------------
echo [3/5] Verificando VB-CABLE...
powershell -NoProfile -Command "if (Get-CimInstance Win32_SoundDevice | Where-Object {$_.Name -like '*VB-Audio*'}) { exit 0 } else { exit 1 }"
if errorlevel 1 (
    echo       Baixando e instalando VB-CABLE - aceite a janela de permissao ^(UAC^)...
    powershell -NoProfile -Command "$d=\"$env:TEMP\vbcable\"; New-Item -ItemType Directory -Force $d | Out-Null; Invoke-WebRequest -Uri 'https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip' -OutFile \"$d\vb.zip\" -UseBasicParsing; Expand-Archive \"$d\vb.zip\" -DestinationPath $d -Force; Start-Process -FilePath \"$d\VBCABLE_Setup_x64.exe\" -ArgumentList '-i','-h' -Verb RunAs -Wait"
    echo       IMPORTANTE: o instalador do VB-CABLE costuma deixar o CABLE como
    echo       saida padrao do Windows. Depois da instalacao, va em
    echo       Configuracoes ^> Som ^> Saida e escolha seus alto-falantes.
) else (
    echo       VB-CABLE ja instalado.
)

rem --- 4) Modelos de IA -------------------------------------------
echo [4/5] Baixando e preparando os modelos (Whisper + tradutor Opus-MT, ~1,2 GB na primeira vez; a conversao do tradutor leva alguns minutos)...
".venv\Scripts\python.exe" scripts\preparar_modelos.py
if errorlevel 1 (
    echo.
    echo   AVISO: o tradutor Opus-MT nao ficou pronto. O app vai funcionar com o
    echo   tradutor reserva ^(Argos^), de qualidade menor e em portugues de Portugal.
    echo   Para tentar de novo depois: .venv\Scripts\python.exe scripts\preparar_modelos.py
    echo.
)

rem --- 5) Pronto ---------------------------------------------------
echo.
echo [5/5] Instalacao concluida!
echo.
echo   Para usar:  dois cliques em run.bat
echo   Leia o README.md para configurar o roteamento de audio (1 minuto).
echo.
pause
