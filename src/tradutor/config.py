"""Configuração persistida do app (config.json na raiz do projeto).

O arquivo é criado/atualizado automaticamente pelo app em tempo de execução
(`AppConfig.save`) e não deve ser versionado: guarda estado específico da
máquina onde o app roda (índice de dispositivo de áudio, posição da legenda
na tela, volumes). `AppConfig.load` devolve os padrões abaixo quando o
arquivo ainda não existe.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AppConfig:
    capture_device_hint: Optional[str] = None
    output_device: Optional[int] = None          # legado: índice específico de uma máquina
    output_device_name: Optional[str] = None     # fonte da verdade: resolvido p/ índice no boot
    gain_original: float = 0.30
    gain_tts: float = 1.00
    duck_level: float = 0.85   # fração de REDUÇÃO do original enquanto a voz fala
    # RTF ≈ 0,32 num Ryzen 7 5700U (8 núcleos); "small" é mais preciso, porém
    # RTF ≈ 1,0, não acompanha o ao vivo em CPU.
    whisper_model: str = "base"
    # idioma do áudio ("en", "es"…); "auto" = detecção do Whisper a cada segmento
    # (comportamento antigo; erra em trechos curtos, ex.: inglês lido como francês)
    idioma_origem: str = "en"
    tts_voice: str = "pt-BR-FranciscaNeural"
    tts_speed: float = 1.0   # velocidade base da voz (1.0/1.25/1.5); a escada de atraso soma em cima
    # Valores afinados em uso real. O Opus-MT traduz frase a frase; como o
    # SentenceBuffer (juntar_frases) re-junta as frases cortadas no meio, um
    # corte mais curto (5 s) reduz a latência com pouco custo para o MT.
    silence_ms: int = 450
    max_segment_s: float = 5.0
    # true = frase cortada por max_segment_s espera o resto antes de traduzir
    # (o MT é frase a frase e inventa final para meia-frase)
    juntar_frases: bool = True
    max_espera_frase_s: float = 10.0   # tempo máximo segurando a cauda sem continuação
    # maior cauda (em palavras) que espera a continuação; maiores são traduzidas na hora
    max_palavras_espera: int = 8
    # true = o Whisper recebe o vocabulário do tema e o final do trecho
    # anterior como contexto (acerta Nasdaq, Nvidia, FOMC); false = às cegas
    contexto_asr: bool = True
    rate_ladder: list = field(default_factory=lambda: [[2.0, 10], [5.0, 25], [9.0, 40]])
    max_backlog_s: float = 12.0
    gravar_log: bool = False  # true = grava traducoes.log (auditoria de tradução); desligado por padrão
    glossario: str = "trading"   # tema em glossarios/<nome>.json (combo "Tema" na interface)
    show_subtitles: bool = True
    subtitle_font_size: int = 18
    subtitle_pos: Optional[list] = None
    overlay_click_through: bool = False

    def save(self, path: str = "config.json") -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str = "config.json") -> "AppConfig":
        if not os.path.exists(path):
            return AppConfig()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            known = {f.name for f in dataclasses.fields(AppConfig)}
            return AppConfig(**{k: v for k, v in data.items() if k in known})
        except Exception:
            return AppConfig()
