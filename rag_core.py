import os
import re
import json
import hashlib
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from llm_provider import get_chat_llm, get_embeddings

load_dotenv()

VECTOR_DB = os.getenv("VECTOR_DB", "vector_db")
CORPUS_PATH = os.getenv("CORPUS_PATH", "data/corpus.jsonl")

VEC_FETCH_K = int(os.getenv("VEC_FETCH_K", "20"))
VEC_K = int(os.getenv("VEC_K", "6"))
BM25_K = int(os.getenv("BM25_K", "6"))
FINAL_K = int(os.getenv("FINAL_K", "8"))


DISEASE_FILE_WHITELIST: Dict[str, List[str]] = {
    "hypertension": ["irq_d1_hypertension-moh.pdf"],
    "diabetes": ["diabetes.pdf"],
    "asthma": ["irq_d1_asthma guideline.pdf"],
    "ckd": ["chronic kidney disease (ckd) everything you need to know.pdf"],
}
STRICT_WHITELIST_MODE = True
# Query normalization 
def normalize_query(q: str) -> str:
    q = (q or "").strip()

    replacements = {
        # joined words
        "treatmentof": "treatment of",
        "symptomsof": "symptoms of",
        "causesof": "causes of",
        "preventionof": "prevention of",

        # prevention typos
        "preventation": "prevention",
        "preventations": "prevention",
        "prevetation": "prevention",
        "prevetations": "prevention",

        # hypertension typos
        "hypertenshions": "hypertension",
        "hypertenshion": "hypertension",
        "hypertention": "hypertension",
        "hpertension": "hypertension",

        # symptoms typos
        "symtom": "symptom",
        "symtoms": "symptoms",
        "symtomw": "symptoms",
        "symptons": "symptoms",

        # treatment typos
        "treament": "treatment",
        "tretment": "treatment",
    }

    q_lower = q.lower()
    for wrong, right in replacements.items():
        q_lower = q_lower.replace(wrong, right)

    q_lower = re.sub(r"\s+", " ", q_lower).strip()
    return q_lower
# Intent routing
def classify_intent(q: str) -> str:
    ql = (q or "").lower().strip()

    if re.fullmatch(r"(hi|hello|hey|salam|aoa|assalam[- ]?o[- ]?alaikum)[!. ]*", ql):
        return "greeting"

    feedback_patterns = [
        r"\bwhy .*mistake",
        r"\bwrong answer",
        r"\bincorrect",
        r"\bnot correct",
        r"\bsource.*wrong",
        r"\bwrong source",
        r"\bincurrect\b",
    ]
    for p in feedback_patterns:
        if re.search(p, ql):
            return "feedback"

    smalltalk_patterns = [
        r"\bhow are you\b",
        r"\bcan you help me\b",
        r"\bhelp me\b",
        r"\bwhat can you do\b",
        r"\bhow can you help\b",
        r"\bwho are you\b",
        r"\byour name\b",
        r"\bthanks\b",
        r"\bthank you\b",
    ]
    for p in smalltalk_patterns:
        if re.search(p, ql):
            return "smalltalk"

    policy_markers = [
        "medicare", "medicaid", "cms",
        "claim", "claims", "billing",
        "cms-1500", "1500", "837p",
        "timely filing", "filing deadline",
        "prior authorization", "preauthorization",
        "reimbursement", "payment",
        "payer", "provider", "coverage",
        "coding", "procedure code", "diagnosis code",
        "appeal", "appeals", "appeal deadline",
        "denial", "denials", "denied",
    ]
    if any(marker in ql for marker in policy_markers):
        return "policy"

    medical_markers = [
        "symptom", "symptoms", "treatment", "treat", "therapy", "management",
        "cause", "causes", "risk factor", "risk factors",
        "diagnosis", "diagnose", "diagnostic",
        "prevention", "prevent", "preventive", "prophylaxis",
        "medicine", "drug", "dose", "guideline",
        "bp", "blood pressure",
        "hypertension", "diabetes", "asthma", "stroke", "infection", "fever",
        "disease", "ckd", "kidney",
    ]
    if any(m in ql for m in medical_markers):
        return "medical"

    if re.search(r"\b(what|why|how|when|where|can|could|should|is|are)\b", ql):
        return "non_medical"

    return "non_medical"

# Disease detection 

def detect_target_disease(query: str) -> Optional[str]:
    q = (query or "").lower()

    disease_aliases = {
        "hypertension": ["hypertension", "high blood pressure", "blood pressure"],
        "diabetes": ["diabetes", "blood sugar", "high sugar"],
        "asthma": ["asthma"],
        "ckd": ["ckd", "chronic kidney disease", "kidney disease"],
    }

    for canonical, aliases in disease_aliases.items():
        for alias in aliases:
            if alias in q:
                return canonical

    # very loose 'bp' only if also has medical terms
    if " bp " in f" {q} " and any(k in q for k in ["symptom", "treat", "cause", "hypertension", "blood pressure"]):
        return "hypertension"

    return None


def detect_requested_sections(query: str) -> Set[str]:
    q = (query or "").lower()
    sections: Set[str] = set()

    if any(k in q for k in ["symptom", "symptoms", "sign", "signs"]):
        sections.add("symptoms")
    if any(k in q for k in ["cause", "causes", "risk factor", "risk factors", "etiology", "aetiology"]):
        sections.add("causes")
    if any(k in q for k in ["treatment", "treat", "management", "therapy", "medication", "medicine"]):
        sections.add("treatment")
    if any(k in q for k in ["diagnosis", "diagnose", "diagnostic"]):
        sections.add("diagnosis")
    if any(k in q for k in ["prevention", "prevent", "preventive", "prophylaxis"]):
        sections.add("prevention")

    if not sections:
        sections = {"general"}

    return sections
# BM25 helpers
_word_re = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
STOPWORDS = {
    "the", "of", "and", "or", "in", "on", "for", "to", "a", "an", "is", "are",
    "what", "give", "me", "please", "about", "tell", "do", "does"
}


def tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in _word_re.findall(text)]
    return [t for t in toks if t not in STOPWORDS]


def load_corpus_jsonl(path: str) -> List[Document]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"BM25 corpus not found: {path}. Run ingest.py first.")

    docs: List[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            docs.append(Document(page_content=obj["text"], metadata=obj.get("metadata", {})))
    return docs
# Fusion 
def doc_key(d: Document) -> str:
    cid = d.metadata.get("chunk_id")
    if cid:
        return str(cid)
    sf = d.metadata.get("source_file", "unknown.pdf")
    pg = d.metadata.get("page", "?")
    content_hash = hashlib.sha1(
        (d.page_content or "").encode("utf-8")
    ).hexdigest()
    return f"{sf}|{pg}|{content_hash}"


def rrf_fuse(
    vec_docs: List[Document],
    bm_docs: List[Document],
    k: int,
    rrf_k: int = 60,
    *,
    include_scores: bool = False,
) -> List[Document]:
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for rank, d in enumerate(vec_docs, start=1):
        key = doc_key(d)
        doc_map[key] = d
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    for rank, d in enumerate(bm_docs, start=1):
        key = doc_key(d)
        doc_map[key] = d
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not include_scores:
        return [doc_map[key] for key, _ in ranked[:k]]

    results: List[Document] = []
    for key, score in ranked[:k]:
        doc = doc_map[key]
        metadata = deepcopy(doc.metadata)
        metadata.setdefault("chunk_id", key)
        metadata["retrieval_score"] = score
        results.append(Document(page_content=doc.page_content, metadata=metadata))
    return results
# Strict disease filtering helpers
def _normalize_filename(name: str) -> str:
    return (name or "").strip().lower()


SECTION_KEYWORDS: Dict[str, List[str]] = {
    "symptoms": ["symptom", "symptoms", "sign", "signs", "presentation"],
    "causes": ["cause", "causes", "risk factor", "risk factors", "etiology"],
    "treatment": ["treatment", "management", "therapy", "medication", "medicine"],
    "diagnosis": ["diagnosis", "diagnose", "diagnostic", "screening", "test"],
    "prevention": ["prevention", "prevent", "prophylaxis"],
}


def _clean_chunk_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    text = re.sub(r"^\d+\s+", "", text)
    return text.strip()


def _extract_bullet_lines(text: str, max_items: int = 5) -> List[str]:
    text = _clean_chunk_text(text)
    items = re.findall(r"(?:[•\-−])\s*([^•\-−]+?)(?=\s*[•\-−]|$)", text)
    cleaned = [
        item.strip(" .;")
        for item in items
        if 12 <= len(item.strip(" .;")) <= 220
    ]
    if cleaned:
        return cleaned[:max_items]

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) >= 25
    ]
    return sentences[:max_items]


def _infer_chunk_label(text: str, requested_sections: Set[str]) -> str:
    text_lower = text.lower()
    for section in requested_sections:
        if section == "general":
            continue
        for keyword in SECTION_KEYWORDS.get(section, []):
            if keyword in text_lower:
                return section.replace("_", " ").title()
    return "Guideline excerpt"


def _rerank_docs_by_section(docs: List[Document], requested_sections: Set[str]) -> List[Document]:
    keywords: List[str] = []
    for section in requested_sections:
        keywords.extend(SECTION_KEYWORDS.get(section, []))
    if not keywords:
        return docs

    noise_markers = [
        "annex",
        "checklist",
        "monitoring checklist",
        "place a check",
        "comments column",
    ]

    def score(doc: Document) -> int:
        text = (doc.page_content or "").lower()
        value = sum(2 for keyword in keywords if keyword in text)

        if "symptoms" in requested_sections:
            if "symptoms of" in text or "clinical features" in text:
                value += 6
            if any(marker in text for marker in noise_markers):
                value -= 10

        return value

    return sorted(docs, key=score, reverse=True)


def filter_docs_by_whitelist(docs: List[Document], target_disease: str) -> List[Document]:
    allowed_files = [f.lower() for f in DISEASE_FILE_WHITELIST.get(target_disease, [])]
    if not allowed_files:
        return []
    filtered = []
    for d in docs:
        sf = _normalize_filename(d.metadata.get("source_file", ""))
        if sf in allowed_files:
            filtered.append(d)
    return filtered


def filter_docs_by_metadata_or_filename(docs: List[Document], target_disease: str) -> List[Document]:
    target = target_disease.lower()
    filtered: List[Document] = []
    for d in docs:
        md_disease = _normalize_filename(str(d.metadata.get("disease", "")))
        sf = _normalize_filename(str(d.metadata.get("source_file", "")))
        if target in md_disease or target in sf:
            filtered.append(d)
    return filtered
# Output schema

@dataclass
class AnswerResult:
    answer: str
    sources: List[str]
    confidence: float
    normalized_query: str


@dataclass
class AnswerTrace:
    answer: str
    sources: List[str]
    confidence: float
    normalized_query: str
    retrieved_docs: List[Document]


def _copy_documents(docs: List[Document]) -> List[Document]:
    """Return detached trace documents without mutating retriever-owned objects."""
    copies: List[Document] = []
    for doc in docs:
        metadata = deepcopy(doc.metadata)
        metadata.setdefault("chunk_id", doc_key(doc))
        copies.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
        )
    return copies


# Main service
class MedicalRAG:
    def __init__(self):
        self.embeddings = get_embeddings()

        self.vectorstore = FAISS.load_local(
            VECTOR_DB,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        self.corpus_docs = load_corpus_jsonl(CORPUS_PATH)
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.corpus_docs])

        self.llm = get_chat_llm(temperature=0)

    # Retrieval
    def _vector_mmr_search(self, query: str) -> List[Document]:
        return self.vectorstore.max_marginal_relevance_search(
            query,
            k=VEC_K,
            fetch_k=VEC_FETCH_K,
            lambda_mult=0.7,
        )

    def _bm25_search(self, query: str) -> List[Document]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        top_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:BM25_K]
        return [self.corpus_docs[i] for i in top_ids]

    def hybrid_retrieve(self, query: str, target_disease: Optional[str] = None) -> Tuple[List[Document], float]:
        vec_docs = self._vector_mmr_search(query)
        bm_docs = self._bm25_search(query)

        fused_docs = rrf_fuse(vec_docs, bm_docs, k=FINAL_K * 3)

        # Apply strict whitelist filtering if disease detected
        docs = fused_docs
        mixed_penalty = 0.0

        if target_disease:
            wl = filter_docs_by_whitelist(docs, target_disease)
            if wl:
                docs = wl
            else:
                # if strict whitelist mode but no whitelist matches, fallback to metadata/filename filter
                if STRICT_WHITELIST_MODE:
                    fallback = filter_docs_by_metadata_or_filename(docs, target_disease)
                    docs = fallback if fallback else []
                else:
                    docs = filter_docs_by_metadata_or_filename(docs, target_disease)

        docs = docs[:FINAL_K]

        # Confidence heuristic
        vec_ids = {doc_key(d) for d in vec_docs}
        bm_ids = {doc_key(d) for d in bm_docs}
        overlap = len(vec_ids & bm_ids)

        # penalty for mixed sources when disease is detected
        if target_disease and docs:
            srcs = [_normalize_filename(d.metadata.get("source_file", "")) for d in docs]
            allowed = [f.lower() for f in DISEASE_FILE_WHITELIST.get(target_disease, [])]
            if allowed:
                unrelated = sum(1 for s in srcs if s not in allowed)
                mixed_penalty = 0.10 * unrelated

        confidence = min(0.95, 0.50 + 0.08 * overlap + 0.03 * len(docs) - mixed_penalty)
        confidence = max(0.20, confidence)

        return docs, round(confidence, 2)

    #  Context formatting 
    def _build_context(self, docs: List[Document]) -> Tuple[str, List[str]]:
        blocks = []
        all_sources = []

        for i, d in enumerate(docs, start=1):
            source_file = d.metadata.get("source_file", "unknown.pdf")
            page = d.metadata.get("page", None)
            page_str = f"page {page + 1}" if isinstance(page, int) else "page ?"

            text = (d.page_content or "").strip()
            if len(text) > 1600:
                text = text[:1600] + "…"

            blocks.append(f"[S{i}] {source_file} ({page_str})\n{text}")
            all_sources.append(f"{source_file} ({page_str})")

        return "\n\n".join(blocks), sorted(set(all_sources))

    #Output spec 
    def _build_output_spec(self, requested_sections: Set[str]) -> str:
        if requested_sections == {"symptoms"}:
            return """
Return ONLY:
1) Explanation (1-2 lines)
2) Symptoms
3) Safety disclaimer (short)

Do NOT include causes, risk factors, treatment, diagnosis, or prevention unless the user asked for them.
"""
        if requested_sections == {"causes"}:
            return """
Return ONLY:
1) Explanation (1-2 lines)
2) Causes / Risk factors
3) Safety disclaimer (short)

Do NOT include symptoms, diagnosis, treatment, or prevention unless asked.
"""
        if requested_sections == {"treatment"}:
            return """
Return ONLY:
1) Explanation (1-2 lines)
2) Treatment / Management
3) Safety disclaimer (short)

Do NOT include symptoms, causes, diagnosis, or prevention unless asked.
"""
        if requested_sections == {"diagnosis"}:
            return """
Return ONLY:
1) Explanation (1-2 lines)
2) Diagnosis
3) Safety disclaimer (short)

Do NOT include symptoms, causes, treatment, or prevention unless asked.
"""
        if requested_sections == {"prevention"}:
            return """
Return ONLY:
1) Explanation (1-2 lines)
2) Prevention
3) Safety disclaimer (short)

Do NOT include symptoms, causes, diagnosis, or treatment unless asked.
"""

        return """
Return in this format (include ONLY sections relevant to the user's question):
1) Explanation
2) Symptoms (if asked)
3) Causes / Risk factors (if asked)
4) Treatment / Management (if asked)
5) Diagnosis (if asked)
6) Prevention (if asked)
7) Safety disclaimer (short)
"""

    #  Citations + sources control 
    def _normalize_citation_format(self, answer: str) -> str:
        # Convert accidental [7] -> [S7], keep [S7] unchanged
        return re.sub(r"(?<!\[S)\[(\d+)\]", r"[S\1]", answer)

    def _extract_cited_indices(self, answer: str, max_docs: int) -> List[int]:
        nums = [int(x) for x in re.findall(r"\[S(\d+)\]", answer)]
        seen = set()
        out = []
        for n in nums:
            if 1 <= n <= max_docs and n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def _sources_from_citations(self, docs: List[Document], cited_indices: List[int]) -> List[str]:
        sources = []
        for idx in cited_indices:
            d = docs[idx - 1]  # S1 -> docs[0]
            source_file = d.metadata.get("source_file", "unknown.pdf")
            page = d.metadata.get("page", None)
            page_str = f"page {page + 1}" if isinstance(page, int) else "page ?"
            sources.append(f"{source_file} ({page_str})")
        return sorted(set(sources))

    def _generate_answer(
        self,
        prompt: str,
        context: str,
        normalized: str,
        docs: List[Document],
        requested_sections: Optional[Set[str]] = None,
        allow_chunk_fallback: bool = True,
    ) -> str:
        response = self.llm.invoke(prompt)
        answer = self._normalize_citation_format((response.content or "").strip())

        # Retry any uncited response, including a premature abstention. Smaller
        # local models sometimes miss evidence on the first pass.
        if "[S" not in answer and docs:
            retry_prompt = f"""Using ONLY the numbered sources below, answer the question directly and concisely.

STRICT RULES:
- Every factual statement MUST end with at least one valid citation such as [S1] or [S2].
- Preserve material dates, quantities, exceptions, conditions, and scope.
- Do not cite a source unless it supports the statement.
- Do not add outside knowledge.
- If the sources do not clearly answer the question, respond exactly: Not found in documents.

Sources:
{context}

Question: {normalized}

Answer:"""
            response = self.llm.invoke(retry_prompt)
            answer = self._normalize_citation_format((response.content or "").strip())

        if (
            allow_chunk_fallback
            and answer != "Not found in documents."
            and "[S" not in answer
            and docs
        ):
            answer = self._fallback_cited_answer(docs, requested_sections)

        return answer

    def _fallback_policy_answer(
        self,
        query: str,
        docs: List[Document],
    ) -> str:
        """Return a directly quoted policy sentence when generation abstains.

        This fallback is deliberately extractive: it selects one source sentence
        with strong lexical overlap and policy-requirement language. It never
        assembles unrelated chunk fragments or adds outside facts.
        """
        query_text = query.lower().replace("-", " ")
        query_tokens = set(tokenize(query_text))
        if "timely filing" in query_text:
            query_tokens.update(
                {"file", "filed", "claims", "deadline", "limit", "months", "later"}
            )
        requirement_terms = {
            "must", "required", "requirement", "requirements", "deadline",
            "limit", "limits", "within", "later", "calendar", "months",
            "days", "effective", "deny", "file", "filed", "filing",
        }
        candidates: List[Tuple[float, int, str]] = []

        for source_index, doc in enumerate(docs, start=1):
            text = re.sub(r"\s+", " ", (doc.page_content or "").strip())
            sentences = re.split(r"(?<=[.!?])\s+", text)

            for sentence in sentences:
                sentence = sentence.strip()
                if not 40 <= len(sentence) <= 600:
                    continue

                sentence_tokens = set(tokenize(sentence))
                overlap = len(query_tokens & sentence_tokens)
                requirement_overlap = len(requirement_terms & sentence_tokens)
                has_specific_value = bool(re.search(r"\b\d+\b", sentence))

                score = (3.0 * overlap) + requirement_overlap
                if has_specific_value:
                    score += 2.0
                if "timely filing" in query_text and "timely filing" in sentence.lower():
                    score += 5.0
                if "timely filing" in query_text:
                    has_general_deadline = (
                        ("12 months" in sentence.lower() or "1 calendar year" in sentence.lower())
                        and ("filed" in sentence.lower() or "filing" in sentence.lower())
                    )
                    if has_general_deadline:
                        score += 20.0
                if "time limit for filing all" in sentence.lower():
                    score += 8.0
                if "no later than" in sentence.lower() or "no more than" in sentence.lower():
                    score += 3.0
                if not any(term in query_text for term in ("exception", "extend", "retroactive")):
                    if any(term in sentence.lower() for term in ("exception", "extend", "retroactive")):
                        score -= 10.0

                if overlap >= 2 and score >= 8:
                    candidates.append((score, source_index, sentence))

        if not candidates:
            return "Not found in documents."

        _, source_index, sentence = max(candidates, key=lambda item: item[0])
        return f"{sentence} [S{source_index}]"

    def _fallback_cited_answer(
        self,
        docs: List[Document],
        requested_sections: Optional[Set[str]] = None,
    ) -> str:
        """Build a readable cited answer from retrieved chunks when the LLM omits citations."""
        requested_sections = requested_sections or {"general"}
        blocks: List[str] = []
        seen_bullets: Set[str] = set()

        for i, d in enumerate(docs[:4], start=1):
            bullets = _extract_bullet_lines(d.page_content)
            if not bullets:
                continue

            page = d.metadata.get("page")
            page_str = f"page {page + 1}" if isinstance(page, int) else "unknown page"
            label = _infer_chunk_label(d.page_content, requested_sections)

            unique_bullets: List[str] = []
            for bullet in bullets:
                key = bullet.lower()[:80]
                if key not in seen_bullets:
                    seen_bullets.add(key)
                    unique_bullets.append(bullet)

            if not unique_bullets:
                continue

            lines = [f"#### {label} *({page_str})* [S{i}]"]
            for bullet in unique_bullets[:4]:
                lines.append(f"- {bullet}")
            blocks.append("\n".join(lines))

        if not blocks:
            return "Not found in documents."

        footer = (
            "\n\n---\n"
            "*This summary is drawn only from your uploaded guideline PDFs "
            "and is not a substitute for professional medical advice.*"
        )
        return "### Summary from retrieved guidelines\n\n" + "\n\n".join(blocks) + footer

    # Answer generation
    def answer(self, user_question: str) -> AnswerResult:
        """Return the original public result shape for backward compatibility."""
        trace = self.answer_with_trace(user_question)
        return AnswerResult(
            answer=trace.answer,
            sources=trace.sources,
            confidence=trace.confidence,
            normalized_query=trace.normalized_query,
        )

    def answer_with_trace(self, user_question: str) -> AnswerTrace:
        """Answer once and expose the exact documents supplied to generation."""
        normalized = normalize_query(user_question)
        intent = classify_intent(normalized)

        if intent == "greeting":
            return AnswerTrace(
                answer=(
                    "Hello! I’m a Medical RAG assistant. "
                    "Ask a disease-related question"
                    "(for example: symptoms, causes, treatment, or prevention)."
                ),
                sources=[],
                confidence=1.0,
                normalized_query=normalized,
                retrieved_docs=[],
            )

        if intent == "smalltalk":
            return AnswerTrace(
                answer=(
                    "Yes — I can answer medical guideline questions . "
                    "Ask about symptoms, causes, treatment, diagnosis, or prevention of a disease."
                ),
                sources=[],
                confidence=1.0,
                normalized_query=normalized,
                retrieved_docs=[],
            )

        if intent == "feedback":
            return AnswerTrace(
                answer=(
                    "You are right — mistakes usually happen when retrieval pulls mixed documents, "
                    "or when the answer is not clearly present in the PDFs. "
                    "This version uses strict disease-to-file filtering and shows only cited sources, "
                    "so it should be more accurate now."
                ),
                sources=[],
                confidence=1.0,
                normalized_query=normalized,
                retrieved_docs=[],
            )

        if intent == "non_medical":
            return AnswerTrace(
                answer=(
                    "I can only answer medical guideline questions from the provided PDFs. "
                    "Please ask a disease-related question (symptoms, causes, treatment, diagnosis, prevention)."
                ),
                sources=[],
                confidence=0.8,
                normalized_query=normalized,
                retrieved_docs=[],
            )

        # Healthcare document query
        target_disease = detect_target_disease(normalized)
        requested_sections = detect_requested_sections(normalized)

        docs, conf = self.hybrid_retrieve(normalized, target_disease=target_disease)
        if not docs:
            return AnswerTrace(
                answer="Not found in documents.",
                sources=[],
                confidence=0.25,
                normalized_query=normalized,
                retrieved_docs=[],
            )

        docs = _rerank_docs_by_section(docs, requested_sections)
        trace_docs = _copy_documents(docs)

        context, _all_sources = self._build_context(docs)
        output_spec = self._build_output_spec(requested_sections)

        # enforce whitelist if detected
        whitelist_files = DISEASE_FILE_WHITELIST.get(target_disease, []) if target_disease else []

        if intent == "policy":
            prompt = f"""
You are a HEALTHCARE PAYMENT POLICY RESEARCH ASSISTANT.

STRICT RULES:
- Use ONLY the information in the provided policy Sources.
- Do NOT add external knowledge or infer unstated requirements.
- If the requested detail is not clearly supported, write exactly: Not found in documents.
- Every factual statement MUST include citation(s) like [S1], [S2].
- Preserve material dates, quantities, exceptions, conditions, and scope.
- Do not cite sources that do not exist.

Sources:
{context}

Question:
{normalized}

Answer concisely and cite every factual statement.
"""
        else:
            prompt = f"""
You are a PROFESSIONAL MEDICAL GUIDELINE ASSISTANT.

STRICT RULES:
- Use ONLY the information in the provided Sources.
- Do NOT add external knowledge.
- If the requested detail is not clearly supported, write exactly: Not found in documents.
- Every factual claim MUST include citation(s) like [S1], [S2].
- If the user asks only one section (e.g., symptoms), answer ONLY that section.
- Do not mix diseases. If a target disease is identified, ONLY use sources that belong to that disease.

Target disease (if detected): {target_disease or "not detected"}
Allowed files (if target disease detected): {whitelist_files or "n/a"}
Requested sections: {sorted(list(requested_sections))}

Sources:
{context}

Question:
{normalized}

{output_spec}

Important:
- Add citations on each claim.
- Do not cite sources that do not exist.
"""
        answer = self._generate_answer(
            prompt,
            context,
            normalized,
            docs,
            requested_sections,
            allow_chunk_fallback=intent != "policy",
        )

        if intent == "policy" and (
            answer == "Not found in documents." or "[S" not in answer
        ):
            answer = self._fallback_policy_answer(normalized, docs)

        # Not found, no sources
        if answer == "Not found in documents.":
            return AnswerTrace(
                answer=answer,
                sources=[],
                confidence=min(conf, 0.4),
                normalized_query=normalized,
                retrieved_docs=trace_docs,
            )

        # Must have citations
        if "[S" not in answer:
            return AnswerTrace(
                answer="Not found in documents.",
                sources=[],
                confidence=min(conf, 0.35),
                normalized_query=normalized,
                retrieved_docs=trace_docs,
            )

        cited_indices = self._extract_cited_indices(answer, len(docs))
        if not cited_indices:
            return AnswerTrace(
                answer="Not found in documents.",
                sources=[],
                confidence=min(conf, 0.35),
                normalized_query=normalized,
                retrieved_docs=trace_docs,
            )

        # Only cited sources returned
        cited_sources = self._sources_from_citations(docs, cited_indices)

        # if target disease detected and whitelist exists, ensure cited sources obey whitelist
        if target_disease:
            allowed = [f.lower() for f in DISEASE_FILE_WHITELIST.get(target_disease, [])]
            if allowed:
                bad = [s for s in cited_sources if _normalize_filename(s.split(" (page")[0]) not in allowed]
                if bad:
                    # If citations point to wrong disease docs, reject answer
                    return AnswerTrace(
                        answer="Not found in documents.",
                        sources=[],
                        confidence=0.3,
                        normalized_query=normalized,
                        retrieved_docs=trace_docs,
                    )

        return AnswerTrace(
            answer=answer,
            sources=cited_sources,
            confidence=round(conf, 2),
            normalized_query=normalized,
            retrieved_docs=trace_docs,
        )
