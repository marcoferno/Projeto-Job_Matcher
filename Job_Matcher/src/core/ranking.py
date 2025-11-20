"""
    Implementa ranqueamento usando TF-IDF combinado com similaridade do cosseno,
    centralizando a montagem do texto de comparação da vaga, permitindo configurar hiperparâmetros
    e sendo robusto a listas vazias e textos nulos.
"""

# IMPORTS (Recurso de Linguagem/Compatibilidade)
from __future__ import annotations

# IMPORTS (Stdlib)
from typing import List, Sequence, Tuple, Optional

# IMPORTS (Terceiros)
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# IMPORTS (Local)
from .models import Vaga

DEFAULT_TFIDF_DTYPE = np.float32

def _vaga_text(v: Vaga) -> str:
    """
        Monta um texto consolidado da vaga para uso no ranking.

        Prioriza o formato "Vaga.text_for_match()" quando disponível.
        Caso contrário, faz um fallback usando campos comuns em PT/EN.
    """
    if hasattr(v, "text_for_match") and callable(getattr(v, "text_for_match")):
        return v.text_for_match()

    title = getattr(v, "titulo", None) or getattr(v, "title", "") or ""
    company = getattr(v, "empresa", None) or getattr(v, "company", "") or ""
    desc = getattr(v, "descricao", None) or getattr(v, "description", "") or ""
    tags = getattr(v, "tags", None) or getattr(v, "skills", None) or []
    loc = getattr(v, "local", None) or getattr(v, "location", None)
    parts = [title, company, desc, " ".join(tags or [])]

    if loc:
        parts.append(loc)
    return "\n".join(p for p in parts if p)

def _prepare_docs(perfil_texto:Optional[str], vagas: Sequence[Vaga]) -> List[str]:
    """
        Prepara a lista de documentos de texto para o TF-IDF.

        O primeiro elemento é sempre o texto do perfil/currículo;
        os demais são os textos consolidados de cada vaga.
    """
    perfil = (perfil_texto or "").strip()
    return [perfil] + [_vaga_text(v) for v in vagas]

def rank_por_tfidf(
        perfil_texto: str,
        vagas: List[Vaga],
        top_k: int = 5,
        *,
        max_features: int = 8000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        strip_accents: Optional[str] = "unicode",
        dtype = DEFAULT_TFIDF_DTYPE,
) -> List[Tuple[Vaga, float]]:

    """
    Retorna os Top-K pares (Vaga, score) ordenados por similaridade decrescente.

    Hiperparâmetros ajustáveis para controlar vocabulário, escala de TF, tratamento de acentos e uso de memória.
    Caso a lista de vagas ou o texto do perfil esteja vazio, devolve uma lista vazia.
    Empates de score são resolvidos de forma estável usando título, empresa ou id.
    """
    if not vagas:
        return []
    if not perfil_texto or not perfil_texto.strip():
        return []

    docs = _prepare_docs(perfil_texto, vagas)  # [perfil, vaga_1, vaga_2, ...]

    vect = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        strip_accents=strip_accents,
        lowercase=True,
        dtype=dtype,
    )

    X = vect.fit_transform(docs)
    if X.shape[0] <= 1 or X.shape[1] == 0:
        return []

    # Similaridade currículo (linha 0) vs. todas as vagas (linhas 1..N)
    sim = cosine_similarity(X[0], X[1:]).ravel()

    # Empates estáveis: ordena por (Match, titulo/title, empresa/company, id)
    def tie_key(item: Tuple[Vaga, float]):
        v, s = item
        title = getattr(v, "titulo", None) or getattr(v, "title", "") or ""
        company = getattr(v, "empresa", None) or getattr(v, "company", "") or ""
        vid = getattr(v, "id", "") or ""
        return (-s, title.lower(), company.lower(), str(vid))

    pares = list(zip(vagas, map(float, sim)))
    pares.sort(key = tie_key)

    k = max(0, min(top_k, len(pares)))
    return pares[:k]