"""
    Interface web em Streamlit para o Job Matcher.

    Esta aplicação:

    Lê currículos salvos na pasta "data/curriculos" e usa o backend de coleta e ranking para buscar vagas em
    providers externos. Aplicando diferentes estratégias de ranking para calcular o "match" entre o currículo
    e as vagas, exibindo os resultados numa tabela interativa com links para as vagas.
"""

# IMPORTS (Recurso de Linguagem/Compatibilidade)
from __future__ import annotations

# IMPORTS (Stdlib)
from pathlib import Path
from datetime import datetime, timezone
from typing import List

# IMPORTS (Terceiros)
import streamlit as st

# IMPORTS (Local)
from src.core.parsing import extrair_texto
from src.core.ranking import rank_por_tfidf
from src.services.collect_and_rank import collect_jobs_paged, jobs_to_vagas, rank_multi

try:
    from src.core.semantic import rank_por_embeddings
except Exception:
    rank_por_embeddings = None  # type: ignore[assignment]

# ======================================================================
# Configuração e Helpers
# ======================================================================

PROJECT_ROOT:Path = Path(__file__).resolve().parent.parent
DATA_DIR:Path = PROJECT_ROOT / "data"
CURRICULOS_DIR:Path = DATA_DIR / "curriculos"
PRIORITY_KEYWORDS: tuple[str, ...] = ("cv", "curriculo", "currículo", "resume", "curriculum")

def detectar_curriculos() -> List[Path]:
    """
        Lista arquivos de currículo dentro de data/curriculos.
        Aceita .pdf, .docx e .doc.
    """
    if not CURRICULOS_DIR.exists():
        return []
    pdfs = list(CURRICULOS_DIR.glob("*.pdf"))
    docs = list(CURRICULOS_DIR.glob("*.docx"))
    docs += list(CURRICULOS_DIR.glob("*.doc"))
    arquivos = sorted(pdfs + docs)
    return arquivos

def carregar_texto_cv(cv_path: Path) -> str:
    """
        Extrai o texto do currículo usando o parser do backend.
    """
    return extrair_texto(str(cv_path))

def filtrar_vagas(vagas, only_remote: bool, max_days: int):
    """
        Aplica filtros simples nas vagas.

        "only_remote": Mantém só vagas com indício de remoto
        "max_days": Descarta vagas muito antigas (com base em publicado_em)
    """
    # 1. Filtro remoto
    if only_remote:
        remote_terms = ("remoto", "remote", "home office", "100% remoto")
        def is_remote(v):
            parts = [
                getattr(v, "local", None) or getattr(v, "location", None) or "",
                getattr(v, "titulo", None) or getattr(v, "title", None) or "",
                getattr(v, "descricao", None) or getattr(v, "description", None) or "",
            ]
            text = " ".join(parts).lower()
            return any(term in text for term in remote_terms)
        vagas = [v for v in vagas if is_remote(v)]

    # 2. Filtro por recência
    if max_days > 0:
        now = datetime.now(timezone.utc)
        dt_recentes = []
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
                    dt_recentes.append(v)
            else:
                dt_recentes.append(v)
        vagas = dt_recentes
    return vagas

def executar_busca(
    cv_path: Path,
    query: str,
    where: str,
    pages: int,
    top: int,
    engine: str,
    model_name: str,
    only_remote: bool,
    max_days: int,
):
    """
        Pipeline principal de matching para a interface web.

        1. Lê o currículo.
        2. Coleta vagas (Adzuna + Greenhouse, se configurado).
        3. Normaliza os dados para objetos Vaga.
        4. Aplica filtros (remoto, recência).
        5. Roda o ranking (tfidf / semantic / auto).

        Retorna:
            (topk, log_msgs), onde topk é uma lista de pares (Vaga, score).
    """
    log_msgs = []

    # 1. Lê o Currículo
    log_msgs.append(f"Lendo currículo: {cv_path}")
    perfil_texto = carregar_texto_cv(cv_path)

    # 2. Coleta de vagas (Adzuna + Greenhouse se configurado)
    log_msgs.append(f"Coletando vagas: query='{query}', where='{where}', pages={pages}")
    jobs = collect_jobs_paged(
        query=query,
        where=where or None,
        adzuna_pages=pages,
        adzuna_per_page=50,
        include_greenhouse=True,
    )

    # 3. Normalizar os dados para Vaga
    vagas = jobs_to_vagas(jobs)
    if not vagas:
        return [], log_msgs + ["Nenhuma vaga encontrada após normalização."]

    # 4. Aplica os filtros
    vagas = filtrar_vagas(vagas, only_remote=only_remote, max_days=max_days)
    if not vagas:
        return [], log_msgs + ["Nenhuma vaga resta após aplicar filtros."]

    # 5. Roda a lógica de ranking
    engine = engine.lower().strip()
    if engine == "semantic":
        if rank_por_embeddings is None:
            log_msgs.append("Engine 'semantic' requer 'sentence-transformers'. Usando TF-IDF.")
            topk = rank_por_tfidf(perfil_texto, vagas, top_k=top)
            header = "TOP VAGAS (TF-IDF) [fallback]"
        else:
            try:
                topk = rank_por_embeddings(perfil_texto, vagas, top_k=top, model_name=model_name)
                header = f"TOP VAGAS (SEMANTIC: {model_name})"
            except Exception as e:
                log_msgs.append(f"Erro em embeddings: {e}. Usando TF-IDF.")
                topk = rank_por_tfidf(perfil_texto, vagas, top_k=top)
                header = "TOP VAGAS (TF-IDF) [fallback]"
    elif engine == "auto":
        try:
            topk = rank_multi(perfil_texto, vagas, top_k=top, model_name=model_name)
            header = f"TOP VAGAS (AUTO: sem+tfidf, model={model_name})"
        except Exception as e:
            log_msgs.append(f"Erro em rank_multi: {e}. Usando TF-IDF.")
            topk = rank_por_tfidf(perfil_texto, vagas, top_k=top)
            header = "TOP VAGAS (TF-IDF) [fallback]"
    else:
        topk = rank_por_tfidf(perfil_texto, vagas, top_k=top)
        header = "TOP VAGAS (TF-IDF)"
    return list(topk or []), log_msgs + [header]

# ======================================================================
# Interface Streamlit
# ======================================================================

def main():
    st.set_page_config(
        page_title = "Job Matcher",
        page_icon = "🧠",
        layout = "wide",
    )

    st.title("🧠 Job Matcher – Encontre vagas que combinam com você")

    st.markdown(
        """
       Compare o seu currículo com vagas abertas em poucos cliques:

        1. Faça upload ou selecione seu currículo na barra lateral.
        2. Informe o tipo de vaga e a localidade desejada.
        3. Clique em **Buscar vagas** para ver o ranking de oportunidades
           que mais se parecem com o seu perfil.
        """
    )

    # ---------------- Sidebar (parâmetros) ----------------
    st.sidebar.header("Configurações da busca")

    # 1. Upload opcional de currículo
    uploaded_file = st.sidebar.file_uploader(
        "Envie um currículo em PDF ou DOCX",
        type=["pdf", "docx", "doc"],
        help="Se preferir, faça upload direto do seu currículo.",
    )

    if uploaded_file is not None:
        CURRICULOS_DIR.mkdir(parents=True, exist_ok=True)

        dest_path = CURRICULOS_DIR / uploaded_file.name
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # 2. Detectar currículos disponíveis.
    curriculos = detectar_curriculos()
    if not curriculos:
        st.error(
            "Nenhum currículo encontrado, envie um currículo em PDF ou DOCX pela barra lateral."
        )
        return
    cv_opcoes = {cv.name: cv for cv in curriculos}

    default_index = 0
    if uploaded_file is not None and uploaded_file.name in cv_opcoes:
        default_index = list(cv_opcoes.keys()).index(uploaded_file.name)

    cv_nome = st.sidebar.selectbox(
        "Selecione o currículo",
        options=list(cv_opcoes.keys()),
        index=default_index,
    )

    cv_path = cv_opcoes[cv_nome]

    query = st.sidebar.text_input(
        "Título ou Área da vaga",
        value = "Desenvolvedor",
        help = "Ex.: 'desenvolvedor python', 'estágio em TI', 'engenheiro de dados'.",
    )

    where = st.sidebar.text_input(
        "Cidade / Região da vaga",
        value = "Rio de Janeiro",
        help = "Ex: 'Rio de Janeiro', 'São Paulo' ou deixe vazio para qualquer lugar.",
    )

    engine = st.sidebar.selectbox(
        "Como calcular o match",
        options = ["Auto", "Semantic", "TF-IDF"],
        index = 0,
        help=
            "Auto (recomendado): Combina significado do texto + palavras-chave.\n\n"
            "Semantic: Foca mais no significado do texto.\n\n"
            "TF-IDF: Foca em palavras-chave.",
    )

    model_name = st.sidebar.text_input(
        "Modelo semântico (avançado)",
        value = "paraphrase-multilingual-MiniLM-L12-v2",
        help = "Usado quando o modo de match é 'Auto' ou 'Semantic'.\n\n"
    )

    pages = st.sidebar.number_input(
        "Quantas páginas de vagas buscar?",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
    )

    top = st.sidebar.number_input(
        "Quantas vagas mostrar no ranking?",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Número de vagas que serão exibidas na tabela de resultados.",
    )

    only_remote = st.sidebar.checkbox(
        "Mostrar apenas vagas remotas",
        value=False,
        help="Filtra resultados para vagas que mencionam trabalho remoto/home office.",
    )

    max_days = st.sidebar.number_input(
        "Somente vagas publicadas nos últimos (dias)",
        min_value=0,
        max_value=365,
        value=0,
        step=1,
        help="Use 0 para não filtrar por data de publicação.",
    )

    st.sidebar.markdown("---")
    buscar = st.sidebar.button("🔍 Buscar vagas")

    # ---------------- Corpo principal ----------------
    if not buscar:
        st.info("Ajuste os parâmetros na barra lateral e clique em **'Buscar vagas'** para ver as oportunidades.")
        return

    with st.spinner("Rodando matching..."):
        topk, logs = executar_busca(
            cv_path=cv_path,
            query=query,
            where=where,
            pages=int(pages),
            top=int(top),
            engine=engine,
            model_name=model_name,
            only_remote=only_remote,
            max_days=int(max_days),
        )

    # Mostrar logs simples
    with st.expander("Ver detalhes técnicos da busca (logs)", expanded=False):
        for msg in logs:
            st.write("- ", msg)

    if not topk:
        st.warning("Nenhuma vaga encontrada com esses critérios.")
        return

    # Montar tabela de resultados (Rever)
    rows = []
    for i, (v, score) in enumerate(topk, start=1):
        title = getattr(v, "titulo", None) or getattr(v, "title", "job_title")
        company = getattr(v, "empresa", None) or getattr(v, "company", "—")
        loc = getattr(v, "location", None) or getattr(v, "local", None) or "—"
        url = getattr(v, "url", None) or getattr(v, "redirect_url", None)
        source = getattr(v, "source", None) or getattr(v, "fonte", None) or "?"

        rows.append(
            {
                "Rank": i,
                "Título": title,
                "Empresa": company,
                "Local": loc,
                "Fonte": source,
                "Match": f"{float(score) * 100:.2f} %",
                "Link": url or None,
            }
        )
    st.subheader(f"Resultados (Ranking Top {len(rows)})")

    st.dataframe(rows, use_container_width=True)

    st.markdown(
        """
            Dica: clique no link da vaga (coluna **Link**) com botão direito
            para abrir em uma nova aba.
        """
    )

    # ---------------- Detalhes das vagas ----------------
    st.subheader("Detalhes das vagas")

    for i, (v, score) in enumerate(topk, start=1):
        title = getattr(v, "titulo", None) or getattr(v, "title", "Sem título")
        company = getattr(v, "empresa", None) or getattr(v, "company", "—")
        loc = getattr(v, "location", None) or getattr(v, "local", None) or "—"
        url = getattr(v, "url", None) or getattr(v, "redirect_url", None)

        label_expander = f"{i}. {title} — {company} ({loc})"
        with st.expander(label_expander):
            st.markdown(f"**Match:** {score * 100:.2f} %")

            if url:
                st.markdown(f"**Link da vaga:** [{url}]({url})")

            st.markdown("**Descrição completa:**")
            desc = getattr(v, "descricao", None) or getattr(v, "description", None)
            st.write(desc or "Sem descrição disponível.")

            tags = getattr(v, "tags", None) or getattr(v, "skills", None)
            if tags:
                if not isinstance(tags, (list, tuple)):
                    tags = [str(tags)]
                st.markdown("**Skills / tags:**")
                st.write(", ".join(map(str, tags)))

if __name__ == "__main__":
    main()
