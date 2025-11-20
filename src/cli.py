"""
    Interface de linha de comando (CLI) para o job-matcher.

    Este módulo expõe comandos Typer para executar um ranking de demonstração usando TF-IDF com vagas de exemplo.
    Coletando vagas em tempo real de providers externos e ranquear por skills, fazerndo um match completo entre
    currículo e vagas (match_live), combinando TF-IDF e/ou embeddings semânticos, com filtros por fonte,
    remota e recência.

    A CLI se integra com os módulos core (parsing, ranking, modelos) e com os serviços
    de coleta e normalização de vagas, oferecendo um fluxo ponta-a-ponta via terminal.
"""

# IMPORTS (Stdlib)
import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

# IMPORTS (Terceiros)
import typer
from loguru import logger

# IMPORTS (LOCAL)
from .config import DATA_DIR, CURRICULOS_DIR, env_path, find_first_file, PROJECT_ROOT
from .services.collect_and_rank import (collect_jobs,rank_jobs_by_skill_hits,collect_jobs_paged,apply_preferences,jobs_to_vagas,rank_multi)
from .providers.adzuna import search_adzuna
from .providers.greenhouse import search_greenhouse_from_env
from .core.models import Vaga
from .core.parsing import extrair_texto
from .core.ranking import rank_por_tfidf

try:
    from .core.semantic import rank_por_embeddings
except Exception:
    rank_por_embeddings = None

app = typer.Typer(add_completion=False, no_args_is_help=True)

# ======================================================================
# Utilitários para caminhos
# ======================================================================

_JOBS_EXAMPLE = [
    {
        "id": "1",
        "titulo": "Desenvolvedor Python Backend",
        "empresa": "TechX",
        "descricao": "Desenvolvimento de APIs REST com FastAPI, SQL, testes automatizados, Docker. Experiência com nuvem é um diferencial.",
        "url": "https://exemplo.com/vaga/1",
        "local": "Remoto",
        "senioridade": "Pleno",
        "tags": ["python", "fastapi", "sql", "docker", "testes"],
    },
    {
        "id": "2",
        "titulo": "Engenheiro de Dados",
        "empresa": "DataCo",
        "descricao": "Pipelines em Python, Spark, Airflow, modelagem de dados, SQL e integração com lakes na AWS.",
        "url": "https://exemplo.com/vaga/2",
        "local": "São Paulo",
        "senioridade": "Pleno",
        "tags": ["python", "spark", "airflow", "aws", "sql"],
    },
    {
        "id": "3",
        "titulo": "Cientista de Dados",
        "empresa": "AI Labs",
        "descricao": "Modelagem estatística, ML supervisionado, scikit-learn, experimentação e comunicação com stakeholders.",
        "url": "https://exemplo.com/vaga/3",
        "local": "Remoto",
        "senioridade": "Sênior",
        "tags": ["machine learning", "scikit-learn", "experimentos", "estatística"],
    },
    {
        "id": "4",
        "titulo": "Desenvolvedor Full Stack",
        "empresa": "WebWorks",
        "descricao": "Back-end em Python/Django, front-end em React, integração com APIs e testes.",
        "url": "https://exemplo.com/vaga/4",
        "local": "Híbrido - Rio de Janeiro",
        "senioridade": "Pleno",
        "tags": ["python", "django", "react", "api", "tests"],
    },
    {
        "id": "5",
        "titulo": "MLOps Engineer",
        "empresa": "ModelOps",
        "descricao": "CI/CD de modelos, Docker, Kubernetes, monitoração, orquestração e boas práticas de MLOps.",
        "url": "https://exemplo.com/vaga/5",
        "local": "Remoto",
        "senioridade": "Sênior",
        "tags": ["mlops", "docker", "kubernetes", "monitoramento", "cicd"],
    },
    {
        "id": "6",
        "titulo": "Analista de Dados",
        "empresa": "Insights Inc.",
        "descricao": "SQL avançado, dashboards, Python para ETL leve, comunicação e análise exploratória.",
        "url": "https://exemplo.com/vaga/6",
        "local": "Remoto",
        "senioridade": "Júnior",
        "tags": ["sql", "dashboards", "python", "etl"],
    },
]

def _resolve_cv_path(cli_cv: Optional[str]) -> Path:
    """
        Resolve o caminho do currículo a partir da linha de comando, .env ou pastas padrão.

        Prioridade:
        1. Caminho explícito via --cv.
        2. Variável de ambiente CV_PATH (usando env_path).
        3. Primeiro arquivo .pdf/.docx/.txt encontrado em data/curriculos/ ou data/.

        Levanta:
           FileNotFoundError: se nenhum currículo válido for encontrado.
    """
    if cli_cv:
        p = Path(cli_cv).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--cv informado, mas não existe: {p}")
        return p
    env_cv = env_path("CV_PATH")

    if env_cv and env_cv.exists():
        return env_cv
    exts = {".pdf", ".docx", ".txt"}

    found = find_first_file([CURRICULOS_DIR, DATA_DIR], exts)

    if found:
        logger.info(f"Currículo detectado automaticamente: {found.relative_to(PROJECT_ROOT)}")
        return found
    raise FileNotFoundError(
        "Não encontrei currículo. Passe --cv CAMINHO/arquivo.pdf, "
        "ou coloque um arquivo .pdf/.docx/.txt em data/curriculos/, "
        "ou defina CV_PATH no .env."
    )

def _ensure_jobs_file(cli_jobs: Optional[str]) -> Path:
    """
        Garante um arquivo JSON de vagas para o comando de demo.

        Fluxo:
        1. Se --jobs foi informado, valida se o arquivo existe.
        2. Se a variável JOBS_FILE estiver definida no .env, utiliza esse caminho.
        3. Caso contrário, garante a existência de data/jobs_demo.json:
          - se não existir, cria um arquivo com vagas de exemplo (_JOBS_EXAMPLE).
        Levanta:
            FileNotFoundError: se o caminho passado em --jobs não existir.
    """
    if cli_jobs:
        p = Path(cli_jobs).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--jobs informado, mas não existe: {p}")
        return p
    env_jobs = env_path("JOBS_FILE")
    if env_jobs and env_jobs.exists():
        return env_jobs

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    default_path = DATA_DIR / "jobs_demo.json"
    if not default_path.exists():
        logger.warning("jobs_demo.json não encontrado — criando exemplo automaticamente.")
        default_path.write_text(json.dumps(_JOBS_EXAMPLE, ensure_ascii=False, indent=2), encoding="utf-8")
    return default_path

# ======================================================================
# COMANDO 1 — DEMO (TF-IDF) -> Sem exigir Query.
# ======================================================================

@app.command("rank_demo")
def rank_demo(
    cv:Optional[str] = typer.Option(None, help = "(Opcional) caminho do seu currículo (PDF/DOCX/TXT)"),
    jobs:Optional[str] = typer.Option(None, help = "(Opcional) caminho do JSON com vagas"),
    top:int = typer.Option(5, help = "Quantas vagas retornar"),
):
    """
        Roda um ranking de demonstração usando TF-IDF com vagas de exemplo.

        Usando um currículo real que lê as vagas de um JSON e aplica "rank_por_tfidf"
        mostrando as vagas relevantes no terminal.
    """
    cv_path = _resolve_cv_path(cv)
    jobs_path = _ensure_jobs_file(jobs)

    logger.info(f"Lendo currículo: {cv_path}")
    perfil_texto = extrair_texto(str(cv_path))

    logger.info(f"Lendo vagas: {jobs_path}")
    data = json.loads(Path(jobs_path).read_text(encoding="utf-8"))

    vagas = [Vaga(**v) for v in data]
    logger.info("Calculando ranking (TF-IDF)…")

    topk = rank_por_tfidf(perfil_texto, vagas, top_k=top)
    typer.echo("\n=== TOP VAGAS (DEMO TF-IDF) ===")
    for i, (v, score) in enumerate(topk, 1):
        typer.echo(f"{i}. {v.titulo} — {v.empresa}  | score={score:.4f}")
        if v.url:
            typer.echo(f"   {v.url}")
    typer.echo("================================\n")

# ======================================================================
# COMANDO 2 — COLETA REAL (Adzuna/boards) -> Exige Query
# ======================================================================

@app.command("collect_live")
def collect_live(
    query: str = typer.Option(..., help="Texto simples, ex.: 'desenvolvedor python backend'"),
    where: str = typer.Option("", help="Localidade (ex.: 'São Paulo') ou vazio p/ qualquer lugar"),
    skills: str = typer.Option("python,fastapi,sql", help="Skills separadas por vírgula p/ ranquear"),
    top: int = typer.Option(5, help="Quantas vagas retornar"),
):
    """
        Coleta vagas em tempo real (Adzuna/Greenhouse) e ranqueia por skills.

        Usa "collect_jobs" para buscar até 50 vagas e as converte a string de skills em lista,
        aplicando "rank_jobs_by_skill_hits" para ranquear por ocorrência de skills.

        Exibindo as vagas relevantes com título, empresa, local e URL.
    """
    skills_list = [s.strip() for s in skills.split(",") if s.strip()]
    where_arg = where or None

    jobs = collect_jobs(query=query, where=where_arg, limit=50)
    if not jobs:
        typer.echo("Nenhuma vaga retornada. Confira suas chaves do .env ou ajuste a query.")
        raise typer.Exit(code=1)

    topk = rank_jobs_by_skill_hits(jobs, skills_list, top_k=top)

    typer.echo("\n=== TOP VAGAS (COLETA REAL) ===")
    for i, j in enumerate(topk, 1):
        title = j.get("title") or j.get("titulo") or "Sem título"
        company = j.get("company") or j.get("empresa") or "—"
        loc = j.get("location") or j.get("local") or "—"
        url = j.get("redirect_url") or j.get("url") or ""
        typer.echo(f"{i}. {title} — {company} — {loc}")
        if url:
            typer.echo(f"   {url}")
    typer.echo("================================\n")

# ======================================================================
# COMANDO 3 — MATCH LIVE (TF-IDF com o Currículo paginado.)
# ======================================================================

@app.command("match_live")
def match_live(
    query: str = typer.Option(..., help="Termos da vaga (ex: 'desenvolvedor python')."),
    where: str = typer.Option("", help="Localidade para busca (ex: 'São Paulo')."),
    pages: int = typer.Option(1, help="Nº de páginas na Adzuna."),
    top: int = typer.Option(5, help="Nº de vagas no ranking final."),
    cv:Optional[str] = typer.Option(
        None,
        help="Caminho do currículo (DOCX/PDF). Se vazio, detecta automaticamente em data/curriculos.",
    ),
    engine:str = typer.Option(
        "auto",
        help="Motor de ranking: 'tfidf', 'semantic' ou 'auto' (combina ambos).",
    ),
    model:str = typer.Option(
        "paraphrase-multilingual-MiniLM-L12-v2",
        help="Nome do modelo de embeddings (se engine='semantic' ou 'auto').",
    ),
    prefer:str = typer.Option(
        "",
        help="Preferências separadas por vírgula (ex: 'remoto,São Paulo').",
    ),
    ban:str = typer.Option(
        "",
        help="Palavras para penalizar/remover (ex: 'estágio,temporário').",
    ),
    boost_remote:float = typer.Option(
        0.0,
        help="Bônus adicional para vagas remotas (ex: 0.05 = +5%).",
    ),
    only_remote:bool = typer.Option(
        False,
        help="Se ligado, mantém apenas vagas remotas.",
    ),
    max_days:int = typer.Option(
        0,
        help="Descarta vagas com mais de N dias (0 = não filtrar por data).",
    ),
    sources:str = typer.Option(
        "adzuna,greenhouse",
        help="Provedores de vaga a considerar (ex: 'adzuna', 'greenhouse' ou ambos, separados por vírgula).",
    ),
):
    """
        Faz o match em tempo real entre o seu CV e vagas coletadas.

        Fluxo:
         1. Lê e extrai texto do currículo (DOCX/PDF/TXT).
         2. Coleta vagas paginadas (Adzuna + Greenhouse).
         3. Normaliza vagas em objetos Vaga.
         4. Aplica filtros de fonte, remoto e recência (max_days).
         5. Roda ranking TF-IDF, semântico ou combinado (auto).
         6. (Opcional) ajusta scores com preferências simples (apply_preferences).
    """
    # 1. Leitura do currículo.
    cv_path = _resolve_cv_path(cv)
    logger.info(f"Lendo currículo: {cv_path}")
    perfil_texto = extrair_texto(str(cv_path))

    # 2. Coleta vagas paginadas.
    where_arg = where or None
    logger.info(f"Coletando vagas: query='{query}', where='{where}', pages={pages}")
    jobs = collect_jobs_paged(
        query=query,
        where=where_arg,
        adzuna_pages=pages,
        adzuna_per_page=50,
        include_greenhouse=True,
    )

    # 3. Normalização centralizada: dicts -> Vaga.
    vagas = jobs_to_vagas(jobs)
    if not vagas:
        typer.echo("Nenhuma vaga encontrada após normalização (jobs vazios ou inválidos).")
        raise typer.Exit(code=0)

    # 4. Filtrar por fonte (adzuna/greenhouse) se o usuário informar.
    source_tokens = [s.strip().lower() for s in sources.split(",") if s.strip()]
    allowed_sources = set(source_tokens)
    if allowed_sources:
        vagas = [
            v for v in vagas
            if (getattr(v, "source", "") or "").lower() in allowed_sources
        ]

    # 4.1 Filtrar apenas vagas remotas, se solicitado.
    if only_remote:
        remote_terms = ("remoto", "remote", "home office")
        def is_remote(v):
            parts = [
                getattr(v, "local", None) or getattr(v, "location", None) or "",
                getattr(v, "titulo", None) or getattr(v, "title", None) or "",
                getattr(v, "descricao", None) or getattr(v, "description", None) or "",
            ]
            text = " ".join(parts).lower()
            return any(term in text for term in remote_terms)
        vagas = [v for v in vagas if is_remote(v)]

    # 4.2 Filtrar por recência (max_days)
    if max_days > 0:
        now = datetime.now(timezone.utc)
        fresh_vagas = []

        for v in vagas:
            pub = getattr(v, "publicado_em", None)
            pub_dt = None

            if isinstance(pub, str):
                s = pub.strip()
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                try:
                    pub_dt = datetime.fromisoformat(s)
                except Exception:
                    pub_dt = None
            elif isinstance(pub, datetime):
                pub_dt = pub
            if pub_dt is not None:
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                age_days = (now - pub_dt).days
                if age_days <= max_days:
                    fresh_vagas.append(v)
            else:
                fresh_vagas.append(v)
        vagas = fresh_vagas
    if not vagas:
        typer.echo("Nenhuma vaga encontrada após normalização.")
        raise typer.Exit(code=0)

    # 5. Ranking: TF-IDF ou Embeddings ou Combinado
    engine = engine.lower().strip()
    if engine == "semantic":
        if rank_por_embeddings is None:
            typer.echo("Engine 'semantic' requer 'sentence-transformers'. Instale e rode novamente.")
            raise typer.Exit(code=1)
        try:
            topk = rank_por_embeddings(perfil_texto, vagas, top_k=top, model_name=model)
            header = f"TOP VAGAS (MATCH LIVE, SEMANTIC: {model})"
        except Exception as e:
            logger.warning(f"rank_por_embeddings falhou ({e}); aplicando fallback para TF-IDF")
            topk = rank_por_tfidf(perfil_texto, vagas, top_k=top)
            header = "TOP VAGAS (MATCH LIVE, TF-IDF) [fallback]"
    elif engine == "auto":
        try:
            topk = rank_multi(perfil_texto, vagas, top_k=top, model_name=model)
            header = f"TOP VAGAS (MATCH LIVE, AUTO: sem+tfidf, model={model})"
        except Exception as e:
            logger.warning(f"rank_multi falhou ({e}); aplicando fallback para TF-IDF")
            topk = rank_por_tfidf(perfil_texto, vagas, top_k=top)
            header = "TOP VAGAS (MATCH LIVE, TF-IDF) [fallback]"
    else:
        topk = rank_por_tfidf(perfil_texto, vagas, top_k=top)
        header = "TOP VAGAS (MATCH LIVE, TF-IDF)"

    # 6. Aplica preferências (Pós-processamento de Score)
    topk = list(topk or [])

    pref_list = [x.strip() for x in prefer.split(",") if x.strip()]
    ban_list = [x.strip() for x in ban.split(",") if x.strip()]

    if not pref_list and not ban_list and boost_remote == 0:
        typer.echo(f"\n=== {header} ===")
        for i, (v, score) in enumerate(topk, 1):
            title = getattr(v, "titulo", None) or getattr(v, "title", "Sem título")
            company = getattr(v, "empresa", None) or getattr(v, "company", "—")
            loc = getattr(v, "local", None) or getattr(v, "location", None) or "—"
            url = getattr(v, "url", None) or getattr(v, "absolute_url", None)
            typer.echo(f"{i}. {title} — {company} — {loc}  | Match(%)={score:.4f}")
            if url:
                typer.echo(f"   {url}")
        typer.echo("======================================\n")
        return

    vaga_by_key = {}
    pairs_for_prefs = []

    for v, score in topk:
        key = (str(v.id), v.titulo or "", v.empresa or "")
        vaga_by_key[key] = v
        job_dict = {
            "id": v.id,
            "title": v.titulo,
            "company": v.empresa,
            "location": v.local,
            "url": str(v.url) if v.url else "",
            "source": getattr(v, "source", None),
        }
        pairs_for_prefs.append((job_dict, score))
    pairs_for_prefs = apply_preferences(
        pairs_for_prefs,
        prefer=pref_list,
        ban=ban_list,
        boost_remote=boost_remote,
    )

    pairs_for_prefs.sort(key=lambda x: x[1], reverse=True)

    topk_adjusted = []
    for job_dict, score in pairs_for_prefs:
        key = (
            str(job_dict.get("id", "")),
            job_dict.get("title") or "",
            job_dict.get("company") or "",
        )
        v = vaga_by_key.get(key)
        if v is not None:
            topk_adjusted.append((v, score))
    typer.echo(f"\n=== {header} (após preferências) ===")

    for i, (v, score) in enumerate(topk_adjusted, 1):
        title = getattr(v, "titulo", None) or getattr(v, "title", "Sem título")
        company = getattr(v, "empresa", None) or getattr(v, "company", "—")
        loc = getattr(v, "local", None) or getattr(v, "location", None) or "—"
        url = getattr(v, "url", None) or getattr(v, "absolute_url", None)

        typer.echo(f"{i}. {title} — {company} — {loc}  | Match(%)={score:.4f}")
        if url:
            typer.echo(f"   {url}")
    typer.echo("======================================\n")

# ======================================================================
# COMANDO 4 — Diagnóstico
# ======================================================================

@app.command("diag")
def diag():
    """
        Diagnóstico rápido que verifica .env e testa as fontes (Adzuna/Greenhouse).
    """
    typer.echo(f"ADZUNA_APP_ID set?  {bool(os.getenv('ADZUNA_APP_ID'))}")
    typer.echo(f"ADZUNA_APP_KEY set? {bool(os.getenv('ADZUNA_APP_KEY'))}")
    typer.echo(f"GH_BOARDS: {os.getenv('GH_BOARDS', '(vazio)')}")

    try:
        r = search_adzuna(query="python", where=None, results_per_page=5)
        typer.echo(f"Adzuna OK: {len(r)} resultados")
        if r:
            first = r[0]
            typer.echo(f"  -> {first.get('title')} — {first.get('company')}")
    except Exception as e:
        typer.echo(f"Adzuna ERRO: {e}")

    try:
        gh = search_greenhouse_from_env()
        typer.echo(f"Greenhouse OK: {len(gh)} resultados (boards do GH_BOARDS)")
    except Exception as e:
        typer.echo(f"Greenhouse ERRO: {e}")

if __name__ == "__main__":
    app()