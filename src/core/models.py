"""
    Modelos de dados em Pydantic v2 definindo contratos claros e robustos para as entidades de Vaga e Perfil,
    modelos aceitam dados vindos de múltiplos providers usando aliases de campos, aplicam um saneamento sendo
    tolerantes a campos em HTML ou timestamps em epoch.
"""

# IMPORTS (Recurso de Linguagem/Compatibilidade)
from __future__ import annotations


# IMPORTS (Stdlib)
from typing import Optional, List
from enum import Enum
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import re  # Usado como fallback para limpar HTML quando bs4 não estiver disponível.

# IMPORTS (Terceiros)
from pydantic import BaseModel, HttpUrl, Field, AliasChoices, ConfigDict, field_validator

# ======================================================================
# Enums
# ======================================================================

class Senioridade(str, Enum):
    """
        Faixas de senioridade suportadas para vagas e perfis.

        Os valores são strings em português, pensados para exibição direta
        na interface (Júnior / Pleno / Sênior / Outro).
    """
    junior = "Júnior"
    pleno = "Pleno"
    senior = "Sênior"
    outro = "Outro"

# ======================================================================
# Modelo Vaga
# ======================================================================

class Vaga(BaseModel):
    """
        Representa uma vaga de emprego normalizada para o app.

        Este modelo concentra campos vindos de múltiplos providers,
        usando aliases para mapear variações de nomes de campos e aplicando um saneamento leve nos dados.
    """

# -------- Config do modelo --------
    model_config = ConfigDict(
        str_strip_whitespace = True,
        extra = "ignore",
        use_enum_values = True,
        populate_by_name = True,
    )

# -------- Campos principais --------
    id:str = Field(
        default = "",
        validation_alias = AliasChoices("id", "job_id", "adref"),
        description = "Identificador da vaga."
    )

    titulo:str = Field(
        ...,
        validation_alias = AliasChoices("titulo", "title", "job_title"),
        description = "Título da vaga.",
    )

    empresa:Optional[str] = Field(
        default = None,
        validation_alias = AliasChoices("empresa", "company", "company_name", "company.display_name"),
        description = "Nome da empresa anunciante.",
    )

    descricao:Optional[str] = Field(
        default = None,
        validation_alias = AliasChoices("descricao", "description", "description_html", "job_description"),
        description = "Descrição textual da vaga.",
    )

    url:Optional[HttpUrl] = Field(
        default = None,
        validation_alias = AliasChoices("url", "redirect_url", "job_url"),
        description = "URL pública da vaga.",
    )

    local:Optional[str] = Field(
        default = None,
        validation_alias = AliasChoices("location", "local", "display_location"),
        description = "Localidade (cidade/estado/país ou 'Remoto').",
    )
    senioridade:Optional[Senioridade] = Field(
        default = None,
        validation_alias = AliasChoices("senioridade", "seniority"),
        description = "Faixa de senioridade (Júnior/Pleno/Sênior/Outro).",
    )

    tags:List[str] = Field(
        default_factory = list,
        validation_alias = AliasChoices("tags", "skills", "keywords"),
        description = "Palavras-chave ou habilidades relacionadas à vaga.",
    )

# -------- Campos opcionais úteis --------
    source:Optional[str] = Field(
        default = None,
        validation_alias=AliasChoices("source", "fonte", "orign"),
        description = "Provedor origem (ex.: 'adzuna', 'greenhouse')."
    )

    publicado_em:Optional[datetime] = Field(
        default = None,
        validation_alias = AliasChoices("publicado_em", "created", "created_at","posted_at"),
        description = "Data/hora de publicação.",
    )

    salario_min:Optional[float] = Field(
        default = None,
        validation_alias = AliasChoices("salario_min", "salary_min", "salary_minimum"),
        description = "Salário mínimo estimado (quando informado pelo provider).",
    )

    salario_max:Optional[float] = Field(
        default = None,
        validation_alias = AliasChoices("salario_max", "salary_max", "salary_maximum"),
        description = "Salário máximo estimado (quando informado pelo provider).",
    )

# -------- Validadores/Saneamento --------
    @field_validator("id", mode = "before")
    @classmethod
    def _coerce_id(cls, v):
        """
            Garante que o ID da vaga seja sempre uma string.

            Alguns providers enviam o identificador como inteiro;
            aqui fazemos o cast para string e tratamos "None" como string vazia.
        """
        if v is None:
            return ""
        return str(v)

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, v: List[str]) -> List[str]:
        """
            Normaliza a lista de "tags/skills", convertendo tudo para minúsculas, removendo espaços extras,
            descartando valores vazios e garantindo que não haja itens duplicados.
        """
        norm:list[str] = []
        seen = set()
        for item in v or []:
            s = (item or "").strip().lower()
            if s and s not in seen:
                seen.add(s)
                norm.append(s)
        return norm

    @field_validator("url", mode = "before")
    @classmethod
    def _empty_string_to_none(cls, v):
        """
            Converte strings vazias de "URL" em "None".

            Alguns providers enviam "" quando não possuem link público para a vaga;
            aqui normalizamos isso para "None".
        """
        if v in ("", None):
            return None
        return v

    @field_validator("descricao", mode = "before")
    @classmethod
    def _html_to_text(cls, v):
        """
            Converte descrições em HTML para texto simples.

            Se o provider enviar HTML, tenta usar BeautifulSoup para extrair o texto.
            Caso a biblioteca não esteja disponível, faz um fallback simples,
            removendo "tags" com expressão regular.
        """
        if not v:
            return v

        if "<" in v and ">" in v:
            try:
                soup = BeautifulSoup(v, "html.parser")
                text = soup.get_text("\n")
                return text
            except Exception:
                return re.sub(r"<[^>]+>", " ", v)
        return v

    @field_validator("publicado_em", mode = "before")
    @classmethod
    def _parse_datetime(cls, v):
        """
        Normaliza datas de publicação recebidas em formatos distintos, aceitando datetime pronto,
        valores epoch em segundos ou milissegundos e strings ISO8601. Para entradas numéricas (epoch),
        sempre retorna um datetime com timezone UTC.
        """
        if v is None or isinstance(v, datetime):
            return v

        if isinstance(v, (int, float)):
            if v > 1e12:
                v /= 1000.0
            return datetime.fromtimestamp(v, tz=timezone.utc)
        return v

    @field_validator("senioridade", mode = "before")
    @classmethod
    def _normalize_senioridade(cls, v):
        """
            Normaliza diferentes formas de indicar senioridade.

            Aceita variações comuns em português e inglês e converte para o enum Senioridade apropriado.
        """

        if v is None or isinstance(v, Senioridade):
            return v

        s = str(v).strip().lower()

        mapping = {
            # júnior
            "jr": Senioridade.junior,
            "jr.": Senioridade.junior,
            "junior": Senioridade.junior,
            "júnior": Senioridade.junior,
            "estagiario": Senioridade.junior,
            "estagiário": Senioridade.junior,
            "estagio": Senioridade.junior,
            "estágio": Senioridade.junior,
            "intern": Senioridade.junior,

            # pleno
            "pleno": Senioridade.pleno,
            "mid": Senioridade.pleno,
            "middle": Senioridade.pleno,
            "semi senior": Senioridade.pleno,

            # sênior
            "sr": Senioridade.senior,
            "sr.": Senioridade.senior,
            "senior": Senioridade.senior,
            "sênior": Senioridade.senior,
            "lead": Senioridade.senior,
            "staff": Senioridade.senior,
        }
        return mapping.get(s, Senioridade.outro)

# -------- Métodos utilitários --------
    def text_for_match(self) -> str:
        """
            Texto consolidado da vaga para uso em ranking/busca.

            Concatena título, empresa, descrição, "tags" e local num único string com quebras de linha,
            para facilitar o uso em TF-IDF, embeddings ou outros mecanismos de similaridade.
        """
        parts = [self.titulo, self.empresa, self.descricao, " ".join(self.tags or [])]
        return "\n".join(p for p in parts if p)

# ======================================================================
# Modelo Perfil
# ======================================================================

class Perfil(BaseModel):
    """
        Perfil do candidato (currículo consolidado).

        Representa um resumo normalizado das informações principais do currículo,
        pensado para ser comparado contra as vagas.
    """

    model_config = ConfigDict(
        str_strip_whitespace = True,
        extra = "ignore",
        use_enum_values = True
    )

    resumo:str = Field(
        ...,
        description = "Texto integral/essencial do currículo."
    )

    habilidades:List[str] = Field(
        default_factory = list,
        description = "Lista de habilidades autodetectadas ou informadas."
    )

    nome:Optional[str] = None
    local:Optional[str] = None
    senioridade_alvo:Optional[Senioridade] = None

    @field_validator("habilidades")
    @classmethod
    def _normalize_habilidades(cls, v: List[str]) -> List[str]:
        """
            Normaliza a lista de habilidades do perfil.

            Aplica as mesmas regras de normalização das "tags" da vaga:
            lower-case, trim, remoção de vazios e deduplicação.
        """
        norm: List[str] = []
        seen = set()
        for item in v or []:
            s = (item or "").strip().lower()
            if s and s not in seen:
                seen.add(s)
                norm.append(s)
        return norm