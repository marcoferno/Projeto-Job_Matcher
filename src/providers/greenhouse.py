"""
    'Integração com a API pública de vagas da Greenhouse (boards-api).

    Este módulo encapsula a lógica para buscar vagas em boards públicos da
    Greenhouse (ex.: 'nubank', 'quintoandar'), tratando detalhes como:

    Fornece funções de busca para vagas de um board específico ou
    de múltiplos boards configurados em variáveis de ambiente.'
"""

# IMPORTS (Recurso de Linguagem/Compatibilidade)
from __future__ import annotations

# IMPORTS (Stdlib)
import os
import time
import re
from bs4 import BeautifulSoup
from typing import Any, Dict, List, Optional, TypedDict
from urllib.parse import urlparse

# IMPORTS (Terceiros)
import requests
from requests import Session
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except Exception:
    Retry = None

# ======================================================================
# Tipos e Constantes
# ======================================================================

# Timeout padrão das requisições HTTP (connect, read), em segundos.
DEFAULT_TIMEOUT: tuple[int, int] = (5, 20)

# User-Agent padrão enviado nas chamadas para a API de boards da Greenhouse.
DEFAULT_UA = "job-matcher/1.0 (Greenhouse) +https://example.local"

# Endpoint base da boards-api da Greenhouse (acesso público a vagas por board).
BOARDS_API = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs"

# Dica: Adicionar? content=true faz a API retornar HTML na descrição da vaga.
DEFAULT_QUERY_PARAMS = {"content": "true"}

class GreenhouseJob(TypedDict, total=False):
    """
        Subset tipado dos campos mais relevantes retornados pela boards-api.

        "Total" = False indica que todos os campos são opcionais, pois a API pode
        omitir alguns deles dependendo da vaga e da configuração do board.
    """
    source:str
    id:str | int
    title:str
    company:str | None
    location:str | None
    absolute_url:str | None
    internal_job_id:int | None
    updated_at:str | None
    created_at:str | None
    description:str | None

class GreenhouseError(RuntimeError):
    """Erro específico do provider Greenhouse."""

# ======================================================================
# Cliente com sessão/retry
# ======================================================================

def _build_session(user_agent: str = DEFAULT_UA) -> Session:
    """
        Cria e retorna uma sessão HTTP pré-configurada para uso com a Greenhouse boards-api.

        Define cabeçalhos padrão e, quando "urllib3.Retry" estiver disponível,
        habilita tentativas automáticas (retry) para requisições GET que receberem respostas 429 ou erros 5xx.
    """
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    )

    if Retry is not None:
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

def _parse_boards_from_env(env_value: Optional[str]) -> List[str]:
    """
        Aceita:
            GH_BOARDS="nubank,quintoandar,stoneco"
            GH_BOARDS="https://boards.greenhouse.io/nubank, https://boards.greenhouse.io/stoneco"
        Retorna apenas os slugs:
            ["nubank", "quintoandar", ...]
    """
    if not env_value:
        return []
    boards:List[str] = []

    for raw in env_value.split(","):
        s = (raw or "").strip()
        if not s:
            continue
        if s.startswith("http"):
            try:
                path = urlparse(s).path  # "/nubank" ou "/board/nubank"
                slug = path.strip("/").split("/")[-1]
            except Exception:
                continue
        else:
            slug = s
        if slug:
            boards.append(slug.lower())
    seen, out = set(), []
    for b in boards:
        if b and b not in seen:
            seen.add(b)
            out.append(b)
    return out

def _html_to_text(html:Optional[str]) -> str:
    """
        Converte HTML em texto simples.

        Tenta usar BeautifulSoup, se instalado. Caso contrário, aplica um
        fallback simples removendo tags com expressão regular.
    """
    if not html:
        return ""
    try:
        return BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)
    except Exception:
        return re.sub(r"<[^>]+>", " ", str(html))

# ======================================================================
# Facades compatíveis (API atual do projeto)
# ======================================================================

def search_greenhouse_board(
    board:str,
    *,
    session:Optional[Session] = None,
    timeout:tuple[int, int] = DEFAULT_TIMEOUT,
    include_html:bool = True,
) -> List[Dict[str, Any]]:
    """
        Busca vagas públicas de um board da Greenhouse via boards-api.

        Args:
            "Board": Slug do board (ex.: "nubank", "quintoandar").
            "Session": Sessão HTTP a reutilizar; se None, cria uma.
            "Timeout": Timeout (connect, read) em segundos.
            "Include_html": Quando True, pede 'content=true' e converte o HTML da descrição para texto simples.
        Returns:
            Lista de dicts compatíveis com o shape `GreenhouseJob`.
        Levanta:
            GreenhouseError: para board inválido, falha de rede ou JSON inesperado.
    """
    if not board or "/" in board:
        raise GreenhouseError(f"Board inválido: {board!r}")
    s = session or _build_session()
    params = dict(DEFAULT_QUERY_PARAMS) if include_html else {}
    url = BOARDS_API.format(board=board)

    try:
        resp = s.get(url, params=params, timeout=timeout)

        if resp.status_code == 429:
            time.sleep(1.0)
            resp = s.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise GreenhouseError(f"Falha de rede Greenhouse({board}): {e}")
    except ValueError as e:
        raise GreenhouseError(f"JSON inválido Greenhouse({board}): {e}")
    jobs = data.get("jobs", []) or []
    if not isinstance(jobs, list):
        raise GreenhouseError(f"Payload inesperado em {board}: 'jobs' não é lista.")

    out:List[GreenhouseJob] = []
    for j in jobs:

        # "company": O board já é por empresa; tentamos extrair do host se quiser diferente.
        company = board

        # descrição: Usar 'content' (HTML) quando pedido, senão 'content' pode vir None.
        desc_html = j.get("content") if include_html else None
        description = _html_to_text(desc_html) if include_html else None

        out.append(
            GreenhouseJob(
                source="greenhouse",
                id=j.get("id"),
                internal_job_id=j.get("internal_job_id"),
                title=j.get("title"),
                company=company,
                location=(j.get("location") or {}).get("name") if isinstance(j.get("location"), dict) else None,
                absolute_url=j.get("absolute_url"),
                updated_at=j.get("updated_at"),
                created_at=j.get("updated_at") or j.get("created_at"),
                description=description,
            )
        )
    return out


def search_greenhouse_from_env(
    *,
    env_var: str = "GH_BOARDS",
    include_html: bool = True,
    timeout: tuple[int, int] = DEFAULT_TIMEOUT,
) -> List[Dict[str, Any]]:
    """
        Busca vagas em todos os boards configurados numa variável de ambiente.

        Lê a variável "env_var", interpretando os seus valores como slugs ou URLs de boards da Greenhouse,
         e agrega os resultados de todos os boards válidos numa única lista.

        Formatos aceitos:
            GH_BOARDS="nubank, quintoandar"
            GH_BOARDS="https://boards.greenhouse.io/nubank, https://boards.greenhouse.io/stoneco"

        Boards individuais que falharem geram um aviso, mas o processo segue para os demais.
        Se nenhum board válido for encontrado ou todos falharem, é levantado um GreenhouseError.
    """
    raw = os.getenv(env_var, "")
    boards = _parse_boards_from_env(raw)
    if not boards:
        raise GreenhouseError(f"{env_var} está vazio ou inválido.")
    s = _build_session()
    all_jobs:List[Dict[str, Any]] = []

    for b in boards:
        try:
            all_jobs.extend(
                search_greenhouse_board(b, session=s, timeout=timeout, include_html=include_html)
            )
        except GreenhouseError as e:
            print(f"[Greenhouse] aviso: {e}")
            continue
    return all_jobs