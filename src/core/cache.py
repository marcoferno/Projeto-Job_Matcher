"""
    Os embeddings são armazenados como arquivos.npy, organizados por nome de modelo e por um hash estável do texto
    normalizado de entrada. A memória é usada para evitar o recálculo de embeddings para as mesmas entradas.
"""

# IMPORTS (Stdlib)
from pathlib import Path
from typing import Optional
import hashlib
import os

# IMPORTS (Terceiros)
import numpy as np

# IMPORTS (Local)
try:from ..config import DATA_DIR
# FALLBACK: <repo>/data/cache/embeddings
except ImportError:Cache_Dir = Path(__file__).resolve().parents[2] / "data" / "cache" / "embeddings"

Cache_Dir = DATA_DIR / "cache" / "embeddings"
Cache_Dir.mkdir(parents=True, exist_ok=True)

def _canonical(text: Optional[str]) -> str:
    """
        Normaliza o texto antes de calcular o hash e usar na memória.

        O objetivo é garantir que textos semanticamente idênticos gerem
        a mesma chave de memória, mesmo que diferenciem em espaços ou
        formatos de quebra de linha.
    """
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()

def _hash_text(text: str) -> str:
    """
        Retorna um hash estável para o texto informado.

        O texto é primeiro normalizado por "_canonical", e depois codificado
        em UTF-8 (ignorando caracteres inválidos), em seguida, é gerado
        um hash SHA1. O valor hexadecimal resultante é usado como parte
        do nome do arquivo de memória.
    """
    return hashlib.sha1(_canonical(text).encode("utf-8", errors="ignore")).hexdigest()

def _embedding_path(model_name:Optional[str], h: str) -> Path:
    """
        Monta o caminho do arquivo para um embedding em cache.

        É criado um subdiretório dentro de "Cache_Dir" para cada modelo,
        usando o nome do modelo com '/' substituído por '_'. O arquivo
        é nomeado usando o hash informado, com extensão '.npy'.
    """
    safe = (model_name or "model").replace("/", "_")
    directory = Cache_Dir / safe
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{h}.npy"

def load_cached_embedding(model_name:str, text:str) -> Optional[np.ndarray]:
    """
       Carrega um embedding em cache para o modelo e texto informados, se existir, caso contrário retorna None.

       O texto é normalizado e hasheado para determinar o caminho esperado
       do arquivo de memória. Se o arquivo existir e contiver um array NumPy
       válido e unidimensional, esse array é retornado.
    """
    path = _embedding_path(model_name, _hash_text(text))
    if not path.exists():
        return None
    try:
        arr = np.load(path, allow_pickle=False)
        if not isinstance(arr, np.ndarray) or arr.ndim != 1:
            return None
        return arr
    except ImportError:return None

def save_cached_embedding(model_name: str, text: str, emb: np.ndarray) -> None:
    """
       Salva um embedding em disco para reutilização futura.

       O texto é normalizado e hasheado para determinar o caminho do
       arquivo de memória. O embedding é convertido para um array NumPy
       com dtype e float32 e gravado num arquivo temporário, que depois
       é substituído de forma atômica pelo arquivo final, evitando
       arquivo parcialmente escritos em caso de falha.
    """
    path = _embedding_path(model_name, _hash_text(text))
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        arr = np.asarray(emb, dtype=np.float32)
        np.save(tmp, arr)
        os.replace(tmp, path)
    except Warning:
        try:
            if tmp.exists():
                tmp.unlink()
        except Warning:
            pass