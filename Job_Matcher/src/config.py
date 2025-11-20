"""
    Configurações e utilitários de ambiente/caminhos para o Job Matcher.

    Este módulo carrega variáveis de ambiente a partir de um arquivo .env, definindo diretórios raiz do projeto
    e fornece helpers seguros para leitura de variáveis, implementa heurística para encontrar automaticamente
    um arquivo de currículo.

    Observação: Este módulo executa load_dotenv() ao ser importado.
    Se for reutilizado como biblioteca, avalie mover o carregamento do .env para o entrypoint da aplicação.
"""

# IMPORTS (Recurso de Linguagem/Compatibilidade)
from __future__ import annotations

# IMPORTS (Stdlib)
import os
import unicodedata
from pathlib import Path
from typing import Iterable, Optional, Sequence

# IMPORTS (Terceiros)
from dotenv import load_dotenv

load_dotenv()

# Diretórios base usados pelo app CLI.
PROJECT_ROOT:Path = Path(__file__).resolve().parent.parent
DATA_DIR:Path = PROJECT_ROOT / "data"
CURRICULOS_DIR:Path = DATA_DIR / "curriculos"
PRIORITY_KEYWORDS: tuple[str, ...] = ("cv", "curriculo", "currículo", "resume", "curriculum")

# Raiz do projeto e diretórios padrão de dados/currículos.
__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "CURRICULOS_DIR",
    "env_path",
    "env_str",
    "env_int",
    "env_bool",
    "find_first_file",
]

# ======================================================================
# Helpers de leitura de .env
# ======================================================================

def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    """
        Lê uma variável de ambiente como string (retorna default se ausente/vazia).
    """
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return val

def env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    """
        Lê uma variável de ambiente como int, com fallback seguro.
    """
    val = env_str(name)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default

def env_bool(name: str, default: Optional[bool] = None) -> Optional[bool]:
    """
        Lê uma variável de ambiente como booleano (true/false/1/0/yes/no).
    """
    val = env_str(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}

def env_path(name: str, must_exist: bool = False) -> Optional[Path]:
    """
        Lê um caminho de arquivo/pasta do .env, expandindo ~ e variáveis.

        Se "must_exist=True" e o caminho não existir, retorna None.
    """
    raw = env_str(name)
    if not raw:
        return None
    p = Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
    if must_exist and not p.exists():
        return None
    return p

# ======================================================================
# Descoberta de arquivos
# ======================================================================

def _normalize_for_match(s: str) -> str:
    """
        Remove acentuação e normaliza para minúsculas para matching robusto.
    """
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).lower()

def _normalize_exts(exts: Iterable[str]) -> set[str]:
    """
        Garante que extensões começam com ponto e são minúsculas.
    """
    out:set[str] = set()
    for e in exts:
        e = e.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        out.add(e)
    return out

def find_first_file(
        dirs:Iterable[Path],
        exts:Iterable[str],
        *, recursive:bool = False,
) -> Optional[Path]:
    """
        Encontra um arquivo candidato numa lista de diretórios seguindo a heurística:
        1. Prioriza nomes contendo palavras-chave (cv, curriculo, currículo, resume...)
        2. Em seguida, o arquivo mais recente por mtime
        3. Como desempate final, ordem lexicográfica do caminho

        Parâmetros:
          - dirs: diretórios a procurar (serão ignorados se não existirem)
          - exts: extensões aceitas ('.pdf', '.docx', '.txt'...) — case-insensitive
          - recursive: se True, busca recursivamente (rglob)
        Retorna:
          - Path do arquivo selecionado ou None se nada for encontrado.
    """
    exts_set = _normalize_exts(exts)
    candidates: list[Path] = []

    for d in dirs:
        if not d.exists():
            continue
        iterator = d.rglob("*") if recursive else d.glob("*")
        for p in iterator:
            if not p.is_file():
                continue
            if p.suffix.lower() in exts_set:
                candidates.append(p)
    if not candidates:
        return None

    def score(p: Path) -> tuple[int, float, str]:
        keywords_norm = _normalize_for_match(" ".join(PRIORITY_KEYWORDS)).split()
        hit = any(key in keywords_norm for key in keywords_norm)
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (0 if hit else 1, -mtime, str(p).lower())

    candidates.sort(key=score)
    return candidates[0]