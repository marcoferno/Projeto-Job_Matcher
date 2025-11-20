"""
    Implementa similaridade semântica baseada em embeddings, fornecendo funções para gerar embeddings de textos
    usando sentence-transformers, com memória em disco e em memória, além de ranquear vagas por similaridade
    de cosseno entre esses embeddings.
"""

# IMPORTS (Recurso de Linguagem/Compatibilidade)
from __future__ import annotations

# IMPORTS (Stdlib)
from typing import List, Sequence, Tuple, Dict, Optional

# IMPORTS (Local)
from .models import Vaga
from .cache import load_cached_embedding, save_cached_embedding

# IMPORTS (Terceiros)
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Cache de instâncias de SentenceTransformer em memória.
# Evita baixar/carregar o mesmo modelo a cada chamada.
_MODEL_CACHE: Dict[str, "SentenceTransformer"] = {}

def _get_model(name: str) -> "SentenceTransformer":
    """
        Obtém (ou carrega) um modelo de embeddings por nome.

        Usa memória (_MODEL_CACHE) para evitar carregar o mesmo modelo várias vezes.
        Se a biblioteca sentence-transformers não estiver instalada, levanta um RuntimeError com mensagem amigável.
    """
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers não está instalada. "
            "Instale com: pip install sentence-transformers"
        )
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = SentenceTransformer(name)  # baixa/carrega o modelo
    return _MODEL_CACHE[name]

def _vaga_text(v: Vaga) -> str:
    """
        Monta um texto consolidado da vaga para uso com embeddings.

        Prioriza o formato "Vaga.text_for_match()" quando disponível.
        Caso contrário, faz fallback pegando variações em PT/EN de título, empresa, descrição, tags/skills
        e local, e concatena tudo num único texto com quebras de linha.
    """
    if hasattr(v, "text_for_match") and callable(getattr(v, "text_for_match")):
        return v.text_for_match()  # type: ignore[no-any-return]

    title = getattr(v, "titulo", None) or getattr(v, "title", "") or ""
    company = getattr(v, "empresa", None) or getattr(v, "company", "") or ""
    desc = getattr(v, "descricao", None) or getattr(v, "description", "") or ""
    tags = getattr(v, "tags", None) or getattr(v, "skills", None) or []
    loc = getattr(v, "local", None) or getattr(v, "location", None)
    parts = [title, company, desc, " ".join(tags or [])]

    if loc:
        parts.append(loc)
    return "\n".join(p for p in parts if p)

def _embed_texts(
    texts: Sequence[str],
    *,
    model_name: str,
    batch_size: int = 32,
    normalize: bool = True,
    dtype = np.float32,
) -> np.ndarray:
    """
        Gera embeddings para uma lista de textos usando o modelo indicado, aproveitando cache em disco,
        codificando em batch o que faltar, persistindo novos embeddings e garantindo que os textos retornem vetor;
        devolve uma matriz NumPy [N, dim] na mesma ordem dos textos de entrada.
    """
    model = _get_model(model_name)
    texts = [t or "" for t in texts]

    # Dimensão esperada (usada para invalidar memória antiga)
    try:
        expected_dim = int(model.get_sentence_embedding_dimension())
    except Exception:
        expected_dim = None

    # 1) Tenta carregar da memória
    cached: List[Optional[np.ndarray]] = []
    for t in texts:
        emb = load_cached_embedding(model_name, t)
        if emb is not None and expected_dim is not None and emb.shape != (expected_dim,):
            emb = None # Invalida entradas de memórias com dimensão antiga/diferente.
        cached.append(emb)

    # 2) Identifica quais textos ainda precisam de embedding.
    to_compute_idx = [i for i, v in enumerate(cached) if v is None]
    if to_compute_idx:
        batch_texts = [texts[i] for i in to_compute_idx]
        new_embs = model.encode(
            batch_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        new_embs = np.asarray(new_embs, dtype=dtype, copy=False)

        if new_embs.ndim == 1:
            new_embs = new_embs.reshape(1, -1)

        for idx, emb in zip(to_compute_idx, new_embs):
            cached[idx] = emb
            try:
                save_cached_embedding(model_name, texts[idx], emb)
            except Exception:
                pass # Ignora falhas de escrita na memória

    # 3) Garante que nenhum ficou "None".
    finalized: List[np.ndarray] = []
    for i, v in enumerate(cached):
        if v is None: # Fallback defensivo: re-encode individualmente.
            single = model.encode(
                [texts[i]],
                batch_size=1,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            single = np.asarray(single, dtype=dtype, copy=False)

            if single.ndim == 1:
                single = single.reshape(1, -1)

            v = single[0]

            try:
                save_cached_embedding(model_name, texts[i], emb)
            except Exception:
                pass # Memória falhou → Segue sem salvar, mas não quebra o fluxo
        finalized.append(np.asarray(v, dtype=dtype, copy=False))
    mat = np.vstack(finalized)
    return mat

def rank_por_embeddings(
    perfil_texto: str,
    vagas: List[Vaga],
    top_k: int = 5,
    *,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 32,
) -> List[Tuple[Vaga, float]]:

    """
        Retorna os Top-K pares por similaridade semântica, usando sentence-transformers para gerar embeddings
        normalizados, ordenando do mais similar para o menos similar com desempate estável por título/empresa/id
        e garantindo que o resultado seja sempre uma lista.
    """
    if not vagas:
        return []

    perfil = (perfil_texto or "").strip()

    if not perfil:
        return []

    # docs[0] = perfil, docs[1..N] = vagas
    docs = [perfil] + [_vaga_text(v) for v in vagas]

    # Gera embeddings normalizados (norma=1)
    embs = _embed_texts(docs, model_name=model_name, batch_size=batch_size, normalize=True)
    q = embs[0:1]   # [1, dim] → embedding do perfil
    vs = embs[1:]   # [N, dim] → embeddings das vagas

    scores = (vs @ q.T).ravel()

    pares:List[Tuple[Vaga, float]] = list(zip(vagas, map(float, scores)))

    # Empates estáveis: (-score, titulo/title, empresa/company, id)
    def tie_key(item: Tuple[Vaga, float]):
        v, s = item
        title = getattr(v, "titulo", None) or getattr(v, "title", "") or ""
        company = getattr(v, "empresa", None) or getattr(v, "company", "") or ""
        vid = getattr(v, "id", "") or ""
        return (-s, title.lower(), company.lower(), str(vid))

    pares.sort(key=tie_key)

    k = max(0, min(top_k, len(pares)))
    return pares[:k]