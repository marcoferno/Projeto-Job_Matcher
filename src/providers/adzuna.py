"""
    Módulo de integração com a API de vagas da Adzuna (Brasil).

    Este arquivo encapsula toda a lógica necessária para consultar vagas na Adzuna,
    incluindo construção de requisições HTTP, aplicação de timeout e política de
    re tentativas, leitura das credenciais a partir de variáveis de ambiente e
    normalização dos dados retornados pela API.
"""

# IMPORTS (Recurso de Linguagem/Compatibilidade)
from __future__ import annotations

# IMPORTS (Stdlib)
import os
from typing import Any, Dict, Iterable, List, Optional, TypedDict

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

# Endpoint base da API da Adzuna Brasil.
# {page} será substituído pelo número da página na chamada.
ADZUNA_BASE = "https://api.adzuna.com/v1/api/jobs/br/search/{page}"

# Timeout padrão (connect, read), em segundos.
DEFAULT_TIMEOUT = (5, 20)

# User-Agent padrão enviado nas requisições HTTP.
DEFAULT_UA = "job-matcher/1.0 (+https://example.local; contact: you@example.com)"

class AdzunaJob(TypedDict, total=False):
    """
        Subset de campos relevantes retornados pela API da Adzuna.

        Usamos TypedDict para ter checagem de tipo estática, mas todos
        os campos são opcionais (total=False) porque a API pode omitir
        alguns dependendo da vaga.
    """
    source:str
    id:str | int
    title:str
    company:str | None
    location:str | None
    created:str | None
    salary_min:float | None
    salary_max:float | None
    contract_type:str | None
    contract_time:str | None
    category:str | None
    redirect_url:str | None
    description:str | None

class AdzunaError(RuntimeError):
    """Erro específico do provider Adzuna."""

# ======================================================================
# Cliente com sessão/retry
# ======================================================================

class AdzunaClient:
    """
        Cliente de alto nível para a API de vagas da Adzuna (Brasil).
    """
    def __init__(
            self,
            app_id:str,
            app_key:str,
            *,
            base_url:str = ADZUNA_BASE,
            session:Optional[Session] = None,
            user_agent:str = DEFAULT_UA,
    ) -> None:
        """
            Inicializa o cliente com credenciais e sessão HTTP.

            Args:
                "App_id": APP ID fornecido pela Adzuna.
                "App_key": APP KEY fornecida pela Adzuna.
                "Base_url": URL base da API (permite override para testes).
                "Session": sessão HTTP opcional (para injetar em testes).
                "User_agent": User-Agent a ser enviado nas requisições.
        """
        if not app_id or not app_key:
            raise AdzunaError("Credenciais da Adzuna ausentes (app_id/app_key).")
        self.app_id = app_id
        self.app_key = app_key
        self.base_url = base_url
        self.session = session or self._build_session(user_agent)

    @staticmethod
    def _build_session(user_agent: str) -> Session:
        """
            Cria uma sessão HTTP com User-Agent e política de retry.

            Se urllib3.Retry estiver disponível, configura retries para
            respostas 429/5xx em requisições GET.
        """
        s = requests.Session()
        s.headers.update({"User-Agent": user_agent})
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

    @staticmethod
    def _clamp_per_page(n: int) -> int:
        # Adzuna limita a 50 resultados por página
        return max(1, min(50, n))

    def search_page(
            self,
            *,
            query:str,
            where:Optional[str] = None,
            page:int = 1,
            results_per_page:int = 30,
            timeout:tuple[int, int] = DEFAULT_TIMEOUT,
            extra_params:Optional[Dict[str, Any]] = None,
    ) -> List[AdzunaJob]:
        """
            Busca uma página de vagas na Adzuna (Brasil).

            Args:
                "Query": Texto de busca (palavra-chave principal).
                "Where": Localização (cidade/estado, ex.: "Rio de Janeiro").
                "Page": Número da página (1-based).
                "Results_per_page": Quantidade de resultados por página (1..50).
                "Timeout": Timeout (connect, read), em segundos.
                "Extra_params": Parâmetros adicionais passados direto para a API.
        """
        if page < 1:
            raise AdzunaError("page deve ser >= 1")

        params: Dict[str, Any] = {
            "app_id":self.app_id,
            "app_key":self.app_key,
            "results_per_page":self._clamp_per_page(results_per_page),
            "what": query.strip(),
            "content-type":"application/json",
        }

        # OPCIONAL: Você pode incluir filtros como 'sort_by' ('date'/'relevance'), 'max_days_old', etc.
        # Ex.: params["sort_by"] = "date"

        if where:
            params["where"] = where

        if extra_params:
            params.update(extra_params)

        url = self.base_url.format(page=page)

        try:
            resp = self.session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise AdzunaError(f"Falha de rede Adzuna: {e}")
        except ValueError as e:
            raise AdzunaError(f"Resposta JSON inválida da Adzuna: {e}")

        results = data.get("results", []) or []
        jobs: List[AdzunaJob] = []
        for item in results:
            # Mapeia o payload bruto da Adzuna para nosso TypedDict simplificado
            jobs.append(AdzunaJob(
                source="adzuna",
                id=item.get("id"),
                title=item.get("title"),
                company=(item.get("company") or {}).get("display_name"),
                location=(item.get("location") or {}).get("display_name"),
                created=item.get("created"),
                salary_min=item.get("salary_min"),
                salary_max=item.get("salary_max"),
                contract_type=item.get("contract_type"),
                contract_time=item.get("contract_time"),
                category=(item.get("category") or {}).get("label"),
                redirect_url=item.get("redirect_url"),
                description=item.get("description"),
                )
            )
        return jobs

    def search_paged(
            self,
            *,
            query:str,
            where:Optional[str] = None,
            pages:int = 2,
            per_page:int = 50,
            timeout:tuple[int, int] = DEFAULT_TIMEOUT,
            extra_params:Optional[Dict[str, Any]] = None,
    ) -> List[AdzunaJob]:
        """
            Busca múltiplas páginas de vagas na Adzuna e concatena os resultados.

            Percorre as páginas de 1 até `pages` (no máximo), respeitando o limite
            de 50 resultados por página imposto pela API. A iteração é interrompida
            antecipadamente se alguma página vier vazia.
        """
        all_jobs: List[AdzunaJob] = []
        for p in range(1, max(1, pages) + 1):
            batch = self.search_page(
                query=query,
                where=where,
                page=p,
                results_per_page=per_page,
                timeout=timeout,
                extra_params=extra_params,
            )
            if not batch:
                break
            all_jobs.extend(batch)
        return all_jobs

# ======================================================================
# Facades compatíveis (API atual do projeto)
# ======================================================================

def _get_env_creds() -> tuple[str, str]:
    """
        Lê as credenciais da Adzuna das variáveis de ambiente.
    """
    app_id = os.getenv("ADZUNA_APP_ID", "")
    app_key = os.getenv("ADZUNA_APP_KEY", "")

    if not app_id or not app_key:
        raise AdzunaError("ADZUNA_APP_ID/ADZUNA_APP_KEY não configurados")
    return app_id, app_key

def search_adzuna(
        query:str,
        where:Optional[str] = None,
        page:int = 1,
        results_per_page: int = 30,
) -> List[Dict[str, Any]]:
    """
        Wrapper simples para buscar uma página na Adzuna.

        Mantém a assinatura compatível com o código legado:
        retorna uma lista de dicts (AdzunaJob é compatível com Dict[str, Any]).
    """
    app_id, app_key = _get_env_creds()
    client = AdzunaClient(app_id, app_key)
    return list(client.search_page(query=query, where=where, page=page, results_per_page=results_per_page))  # type: ignore[return-value]

def search_adzuna_paged(
        query:str,
        where:Optional[str] = None,
        pages:int = 2,
        per_page:int = 50,
) -> List[Dict[str, Any]]:
    """
        Wrapper para buscar múltiplas páginas na Adzuna.

       Agrupa os resultados de 1..pages numa única lista, expondo uma
       interface simples em termos de List[Dict[str, Any>].
    """
    app_id, app_key = _get_env_creds()
    client = AdzunaClient(app_id, app_key)
    return list(client.search_paged(query=query, where=where, pages=pages, per_page=per_page))  # type: ignore[return-value]