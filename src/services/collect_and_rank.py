"""
    Módulo responsável por coletar vagas de emprego em múltiplos providers.

    Ele encapsula toda a lógica de busca, transformação e ranqueamento,
    permitindo comparar vagas vindas de fontes diferentes de forma uniforme.

    Além disso, o módulo oferece funções auxiliares para simplificar queries
    booleanas para a API da Adzuna, limpar HTML presente nas descrições de
    vaga e ajustar os scores conforme preferências simples de localização
    do candidato.
"""

# IMPORTS (Recurso de Linguagem/Compatibilidade)
from __future__ import annotations

# IMPORTS (Stdlib)
import re
from bs4 import BeautifulSoup
from typing import Any, Dict, List, Optional, Sequence, Tuple

# IMPORTS (Local)
from ..providers.adzuna import search_adzuna_paged, AdzunaError
from ..providers.greenhouse import search_greenhouse_from_env, GreenhouseError
from ..core.models import Vaga
from ..core.ranking import rank_por_tfidf

try:
    from ..core.semantic import rank_por_embeddings as _rank_semantic
except Exception:
    _rank_semantic = None

# Logger opcional (Não quebra se loguru não estiver instalado.)
try:from loguru import logger
except Exception:
    class _Dummy:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
        def debug(self, *a, **k): pass
    logger = _Dummy()

# ======================================================================
# Helpers
# ======================================================================

_BOOL_TOKENS = re.compile(r"\b(AND|OR|E|OU)\b|[()&|]", flags=re.IGNORECASE)

def _normalize_query_for_adzuna(q: str) -> str:
    """
        Simplifica booleanos para a API da Adzuna.

        Exemplos:
          "python AND (backend OR dados)" -> "python backend dados"
          "python && dados"               -> "python dados"
          "python E (dados OU backend)"   -> "python dados backend"
    """
    # Remove conectores booleanos/operadores e parênteses, mantendo apenas termos
    if not q:
        return ""
    return _BOOL_TOKENS.sub(" ", q).strip()

def _html_to_text(html: Optional[str]) -> str:
    """
        Converte HTML em texto para não poluir o ranking.
        Usa BeautifulSoup se disponível; caso contrário, fallback regex.
    """
    if not html:
        return ""
    try:
        return BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)
    except Exception:
        return re.sub(r"<[^>]+>", " ", str(html))

def _job_text_for_skills(job: Dict[str, Any]) -> str:
    """
        Texto consolidado para o ranking por skills.

        Inclui título, empresa, descrição (texto+HTML), tags e local.
    """
    title = (job.get("title") or job.get("titulo") or "") or ""
    company = (job.get("company") or job.get("empresa") or "") or ""
    desc = (job.get("description") or "") + " " + _html_to_text(job.get("description_html"))
    tags = job.get("tags") or job.get("skills") or job.get("keywords") or []
    loc = (job.get("location") or job.get("local") or "") or ""
    parts = [title, company, desc, " ".join(map(str, tags)), loc]
    # Tudo em minúsculas para facilitar matching case-insensitive
    return "\n".join(p for p in parts if p).lower()

# ======================================================================
# Coleta
# ======================================================================

def collect_jobs(query: str, where: Optional[str] = None, limit: int = 50, verbose: bool = True,) -> List[Dict[str, Any]]:
    """
        Coleta vagas no Adzuna e Greenhouse.

        Parâmetros:
            "Query": String de busca (booleanos são simplificados automaticamente)
            "Where": Cidade/estado (ou None)
            "Limit": Limite "direcional" (Adzuna usa até 50 por página; aqui mantém 1 página para compat)
            "Verbose": imprime logs informativos
        Retorna:
            lista de dicionários heterogêneos (cada fonte tem campos próprios).
    """
    jobs:List[Dict[str, Any]] = []

    # Adzuna
    try:
        q_norm = _normalize_query_for_adzuna(query)
        adz = search_adzuna_paged(query=q_norm, where=where, pages=1, per_page=min(limit, 50))
        if verbose:
            logger.info(f"Adzuna retornou {len(adz)} vagas para query='{q_norm}' where='{where or '*'}'")
        jobs += adz
    except AdzunaError as e:
        logger.warning(f"Adzuna falhou: {e}")
    except Exception as e:
        logger.warning(f"Adzuna erro inesperado: {e}")

    # Greenhouse
    try:
        gh = search_greenhouse_from_env()
        if verbose:
            logger.info(f"Greenhouse retornou {len(gh)} vagas (somando todos boards do .env)")
        jobs += gh
    except GreenhouseError as e:
        logger.warning(f"Greenhouse falhou: {e}")
    except Exception as e:
        logger.warning(f"Greenhouse erro inesperado: {e}")
    return jobs

def collect_jobs_paged(query: str, where: Optional[str] = None, *, adzuna_pages: int = 2, adzuna_per_page: int = 50, include_greenhouse: bool = True,) -> List[Dict[str, Any]]:
    """
        Variante paginada para buscar MAIS resultados.
    """
    jobs: List[Dict[str, Any]] = []

    try:
        q_norm = _normalize_query_for_adzuna(query)
        jobs += search_adzuna_paged(query=q_norm, where=where, pages=adzuna_pages, per_page=adzuna_per_page)
        logger.info(f"Adzuna (paginado) total: {len(jobs)}")
    except AdzunaError as e:
        logger.warning(f"Adzuna falhou (paginado): {e}")
    except Exception as e:
        logger.warning(f"Adzuna erro inesperado (paginado): {e}")

    if include_greenhouse:
        try:
            gh = search_greenhouse_from_env()
            logger.info(f"Greenhouse total: {len(gh)}")
            jobs += gh
        except GreenhouseError as e:logger.warning(f"Greenhouse falhou: {e}")
        except Exception as e:logger.warning(f"Greenhouse erro inesperado: {e}")
    return jobs

# ======================================================================
# Ranking
# ======================================================================

def _compile_skill_patterns(skills: Sequence[str]) -> List[re.Pattern]:
    """
        Compila padrões para cada skill com prioridade por palavra inteira.

        Para termos alfanuméricos (python, fastapi, sql): usa word-boundary \b
        Para termos com símbolos (C++, .NET, C#): cai para substring case-insensitive
    """
    patterns: List[re.Pattern] = []

    for raw in skills:
        s = (raw or "").strip()
        if not s:
            continue
        if re.fullmatch(r"[A-Za-z0-9_]+", s):
            patterns.append(re.compile(rf"\b{s}\b", flags=re.IGNORECASE))
        else:
            patterns.append(re.compile(re.escape(s), flags=re.IGNORECASE))
    return patterns

def rank_jobs_by_skill_hits(jobs: List[Dict[str, Any]], skills: List[str], top_k: int = 5, *, max_hits_per_skill: int = 5,) -> List[Dict[str, Any]]:
    """
        Ranking simples por contagem de skills (com regex e word-boundaries).

        Sanitiza HTML antes de comparar e limita a contagem de cada skill para evitar textos repetitivos
        inflarem o score. Empates estáveis por (score desc, title asc, company asc, id asc).
    """

    if not jobs:
        return []
    if top_k <= 0:
        return []

    pats = _compile_skill_patterns(skills)

    def score(job: Dict[str, Any]) -> int:
        text = _job_text_for_skills(job)
        total = 0
        for pat in pats:
            hits = len(pat.findall(text))
            total += min(hits, max_hits_per_skill)
        return total

    # Obs.: hoje recalculamos score(job) em cada comparação do sort.
    # Para otimizar, devo pré-computar os scores e reutilizar aqui.
    def tie_key(job: Dict[str, Any]) -> Tuple[int, str, str, str]:
        title = (job.get("title") or job.get("titulo") or "") or ""
        company = (job.get("company") or job.get("empresa") or "") or ""
        jid = str(job.get("id") or "")
        return score(job), title.lower(), company.lower(), jid
    ranked = sorted(jobs, key=tie_key, reverse=True)
    return ranked[:min(top_k, len(ranked))]

def apply_preferences(pairs: list[tuple[dict, float]], *, prefer: list[str] | None = None, ban: list[str] | None = None, boost_remote: float = 0.0, boost_match: float = 0.05, penalty_ban: float = 0.10,) -> list[tuple[dict, float]]:
    """
        Ajusta o score final por preferências simples.

          +boost_match para vagas cujo 'location' contém algum termo de 'prefer'
          +boost_remote se 'location' indicar remoto
          -penalty_ban se 'location' contém algum termo de 'ban'
    """
    # Nota: boosts/penalidades são cumulativos.
    # Uma vaga pode, por exemplo, ganhar boost_remote e boost_match
    # Sofrer penalty_ban se cair em mais de uma categoria.

    prefer = [p.lower() for p in (prefer or []) if p.strip()]
    ban = [b.lower() for b in (ban or []) if b.strip()]

    def tweak(job: dict, score: float) -> float:
        loc = (job.get("location") or job.get("local") or "").lower()
        s = score
        if loc:
            if boost_remote and ("remoto" in loc or "remote" in loc):
                s += boost_remote
            if prefer and any(p in loc for p in prefer):
                s += boost_match
            if ban and any(b in loc for b in ban):
                s -= penalty_ban
        return s
    return [(j, tweak(j, s)) for j, s in pairs]

# ======================================================================
# Normalização: Jobs (dict) -> Vaga
# ======================================================================

def _pick_first(*vals):
    """
        Retorna o primeiro valor "verdadeiro" (não vazio / não None).

        Útil para conciliar campos diferentes entre providers.
    """
    for v in vals:
        if v:
            return v
    return None

def _normalize_location(raw_loc):
    # Greenhouse às vezes manda location como dict {"name": "..."}.
    # Adzuna já manda string. Este helper trata os dois casos.

    if not raw_loc:
        return None
    if isinstance(raw_loc, dict):
        return raw_loc.get("name") or raw_loc.get("display_name")
    return str(raw_loc)

def _strip_html(text: str) -> str:
    """
        Remove tags HTML de forma simples.
    """
    if not text:
        return ""
    return re.sub(r"<[^>]+>", " ", str(text))

def jobs_to_vagas(jobs: List[Dict[str, Any]]) -> List[Vaga]:
    """
        Converte a lista heterogênea de jobs para uma lista de objetos Vaga normalizados.

        - titulo:  titulo | title
        - empresa: empresa | company | board (ex.: 'quintoandar')
        - descricao: descricao | description | content (limpo de HTML)
        - url: url | redirect_url | absolute_url
        - local: local | location (string ou dict)
        - publicado_em: publicado_em | created | created_at | updated_at
        - tags: tags | skills | keywords
        - source: source (adzuna / greenhouse / outro)
        - senioridade: senioridade | level
    """
    vagas: List[Vaga] = []

    # Campos válidos do modelo Vaga.

    try:
        field_names = set(Vaga.__fields__.keys())
    except Exception:
        field_names = {
            "id",
            "titulo",
            "empresa",
            "descricao",
            "url",
            "local",
            "senioridade",
            "tags",
            "source",
            "publicado_em",
        }
    for j in jobs:
        source = j.get("source") or "desconhecido"
        vid = (
            j.get("id")
            or j.get("internal_job_id")
            or j.get("job_id")
            or None
        )

        titulo = _pick_first(j.get("titulo"), j.get("title"), "(Sem título)")
        empresa = _pick_first(j.get("empresa"), j.get("company"), source)

        desc_raw = _pick_first(
            j.get("descricao"),
            j.get("description"),
            j.get("content"),
        )

        descricao = _strip_html(desc_raw) if desc_raw else None

        url = _pick_first(
            j.get("url"),
            j.get("redirect_url"),
            j.get("absolute_url"),
        )

        local = _pick_first(
            j.get("local"),
            _normalize_location(j.get("location")),
        )

        tags = (
            j.get("tags")
            or j.get("skills")
            or j.get("keywords")
            or None
        )

        publicado_em = _pick_first(
            j.get("publicado_em"),
            j.get("created"),
            j.get("created_at"),
            j.get("updated_at"),
        )

        senioridade = j.get("senioridade") or j.get("level")

        base = {
            "id": vid,
            "titulo": titulo,
            "empresa": empresa,
            "descricao": descricao,
            "url": url,
            "local": local,
            "tags": tags,
            "source": source,
            "publicado_em": publicado_em,
            "senioridade": senioridade,
        }

        data_for_model = {k: v for k, v in base.items() if k in field_names and v is not None}

        try:
            vaga = Vaga(**data_for_model)
            vagas.append(vaga)
        except Exception as e:
            logger.warning(f"Falha ao normalizar vaga {j.get('id')}: {e}")
            continue

    return vagas

# # ======================================================================
# Ranking combinado: TF-IDF + Embeddings
# # ======================================================================

def rank_multi(
    perfil_texto: str,
    vagas: List[Vaga],
    top_k: int = 5,
    *,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> List[Tuple[Vaga, float]]:
    """
        Ranking "auto": combina TF-IDF + Embeddings em um único score.

        Estratégia:
          - sempre calcula TF-IDF para todas as vagas;
          - tenta calcular embeddings; se falhar, usa só TF-IDF;
          - combina os scores:
                score_final = w_sem * score_sem + w_tfidf * score_tfidf
            onde, por padrão:
                w_sem    = 0.6 (se embeddings disponíveis)
                w_tfidf  = 0.4
            se embeddings não estiverem disponíveis, w_sem = 0 e w_tfidf = 1.
        Retorna
            Top-K (Vaga, score_final).
    """
    if not vagas:
        return []

    perfil = (perfil_texto or "").strip()

    if not perfil:
        return []

    # Usamos id(v) como chave para garantir que TF-IDF e embeddings
    # Sejam combinados na vaga correta, mesmo que a ordem mude.
    tfidf_pairs = rank_por_tfidf(perfil, vagas, top_k=len(vagas))
    tfidf_scores:Dict[int, float] = {id(v): s for v, s in tfidf_pairs}
    sem_scores:Dict[int, float] = {}

    if _rank_semantic is not None:
        try:
            sem_pairs = _rank_semantic(
                perfil,
                vagas,
                top_k=len(vagas),
                model_name=model_name,
            )
            sem_scores = {id(v): s for v, s in sem_pairs}
        except Exception as e:
            logger.warning(f"rank_multi: embeddings falharam ({e}); usando só TF-IDF.")

    if sem_scores:
        w_sem = 0.6
        w_tfidf = 0.4
    else:
        w_sem = 0.0
        w_tfidf = 1.0

    combined:List[Tuple[Vaga, float]] = []

    for v in vagas:
        key = id(v)
        s_tfidf = tfidf_scores.get(key, 0.0)
        s_sem = sem_scores.get(key, 0.0)
        score = w_sem * s_sem + w_tfidf * s_tfidf
        combined.append((v, float(score)))

    def tie_key(item: Tuple[Vaga, float]):
        v, s = item
        title = getattr(v, "titulo", None) or getattr(v, "title", "") or ""
        company = getattr(v, "empresa", None) or getattr(v, "company", "") or ""
        vid = getattr(v, "id", "") or ""
        return -s, title.lower(), company.lower(), str(vid)
    combined.sort(key=tie_key)

    k = max(0, min(top_k, len(combined)))
    return combined[:k]