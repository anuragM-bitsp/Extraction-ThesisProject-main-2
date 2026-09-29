"""
Streamlit UI for sci-extract.

Start with:

    streamlit run frontend/app.py

All PDF parsing and extraction happens in Steps 1–13 (`PipelineService`,
`get_extractor`, `DocumentIngestionPipeline`, the four Extractor classes).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from frontend.components import (
    render_batch_status,
    render_document_table,
    render_extraction_result,
    render_extractor_availability,
    render_extractor_config,
    render_upload_header,
)
from frontend.session_pipeline import (
    availability,
    available_models,
    extract_records,
    ingest_records,
    prompt_presets,
    register_upload,
)
from frontend.state import (
    MAX_FILES,
    SESSION_API_KEY,
    SESSION_AVAILABLE_MODELS,
    SESSION_DOCUMENTS,
    SESSION_KEY_APPLIED,
    SESSION_LLM_MODEL,
    SESSION_MAX_TOKENS,
    SESSION_PROMPT,
    SESSION_PROMPT_PRESET,
    init_session,
)
from orchestration.extractor_registry import configured_llm_model, configured_max_tokens

UPLOAD_FINGERPRINT = "upload_fingerprint"
from orchestration.db import get_object_store, get_session
from storage.repository import PaperRepository

STRATEGY_LABELS = {
    "rule": "Rule-based",
    "ner": "NER",
    "llm": "Prompt + LLM",
    "hybrid": "Hybrid",
}


def _repo() -> PaperRepository:
    session = get_session()
    return PaperRepository(session, get_object_store())


def _applied_api_key() -> str | None:
    if st.session_state.get(SESSION_KEY_APPLIED) and st.session_state.get(SESSION_API_KEY):
        return st.session_state[SESSION_API_KEY]
    return None


def render_sidebar() -> None:
    st.sidebar.header("LLM Configuration")
    st.sidebar.text_input("API Key", type="password", key=SESSION_API_KEY, help="Used only in this session. Not written to the database or extraction JSON.")
    if st.sidebar.button("Apply"):
        key = st.session_state.get(SESSION_API_KEY) or ""
        if key.strip():
            st.session_state[SESSION_KEY_APPLIED] = True
            st.sidebar.success("API key applied for this session (value is not displayed).")
        else:
            st.session_state[SESSION_KEY_APPLIED] = False
            st.sidebar.warning("No API key entered.")
    elif st.session_state.get(SESSION_KEY_APPLIED):
        st.sidebar.caption("A session API key is configured (not shown).")

    st.sidebar.subheader("Model")
    if st.sidebar.button("Load models", help="List the Claude models this API key can use."):
        models, error = available_models(_applied_api_key())
        st.session_state[SESSION_AVAILABLE_MODELS] = models
        if error:
            st.sidebar.error(error)
    models = st.session_state.get(SESSION_AVAILABLE_MODELS) or []
    default_model = configured_llm_model()
    if models:
        ids = [m["id"] for m in models]
        current = st.session_state.get(SESSION_LLM_MODEL) or default_model
        if current not in ids:
            st.session_state[SESSION_LLM_MODEL] = default_model if default_model in ids else ids[0]
        st.sidebar.selectbox(
            "LLM model",
            ids,
            key=SESSION_LLM_MODEL,
            format_func=lambda i: next((f"{m['display_name']} ({i})" for m in models if m["id"] == i), i),
        )
    else:
        if not st.session_state.get(SESSION_LLM_MODEL):
            st.session_state[SESSION_LLM_MODEL] = default_model
        st.sidebar.text_input("LLM model", key=SESSION_LLM_MODEL, help="Claude model id. Click 'Load models' to pick from a list.")
    if SESSION_MAX_TOKENS not in st.session_state:
        st.session_state[SESSION_MAX_TOKENS] = configured_max_tokens()
    st.sidebar.number_input(
        "Max output tokens", min_value=1024, max_value=64000, step=1024, key=SESSION_MAX_TOKENS,
        help="Upper limit for the length of the LLM's JSON answer.",
    )


def _on_preset_change() -> None:
    presets = prompt_presets()
    name = st.session_state.get(SESSION_PROMPT_PRESET)
    if name in presets:
        st.session_state[SESSION_PROMPT] = presets[name]


def render_prompt_editor() -> None:
    """Editable extraction prompt used by the Prompt + LLM and Hybrid strategies."""
    presets = prompt_presets()
    names = list(presets)
    if SESSION_PROMPT_PRESET not in st.session_state or st.session_state[SESSION_PROMPT_PRESET] not in names:
        st.session_state[SESSION_PROMPT_PRESET] = names[0]
    if not st.session_state.get(SESSION_PROMPT):
        st.session_state[SESSION_PROMPT] = presets[st.session_state[SESSION_PROMPT_PRESET]]

    with st.expander("Extraction prompt", expanded=True):
        st.selectbox(
            "Start from preset (files in the prompts/ folder)",
            names,
            key=SESSION_PROMPT_PRESET,
            on_change=_on_preset_change,
        )
        st.text_area(
            "Prompt (edit freely — the paper text is added automatically; use {document} to place it yourself)",
            key=SESSION_PROMPT,
            height=380,
        )
        if st.session_state[SESSION_PROMPT].strip() != presets[st.session_state[SESSION_PROMPT_PRESET]].strip():
            st.caption("Edited — this run will be stored as a new version (the prompt hash is part of the version).")
        st.caption(
            "The LLM's JSON is always shown as-is in the 'LLM JSON' tab. Keys that match the project schema "
            "(material, precursors, solvent, temperature, reaction_time, pH, synthesis_method, steps, "
            "characterization, properties) are also mapped into `prediction` for fusion, normalization and evaluation."
        )


def _llm_options() -> dict:
    presets = prompt_presets()
    prompt = st.session_state.get(SESSION_PROMPT) or None
    preset = st.session_state.get(SESSION_PROMPT_PRESET)
    edited = prompt is not None and preset in presets and prompt.strip() != presets[preset].strip()
    return {
        "model": st.session_state.get(SESSION_LLM_MODEL) or None,
        "prompt": prompt,
        "prompt_name": f"{preset} (edited)" if edited else preset,
        "max_tokens": int(st.session_state.get(SESSION_MAX_TOKENS) or configured_max_tokens()),
    }


def render_app() -> None:
    st.set_page_config(page_title="Scientific PDF Extraction", layout="wide")
    init_session(st.session_state)
    # Streamlit drops a widget's state when the widget isn't drawn in a run
    # (e.g. the prompt editor while "Rule-based" is selected). Re-assigning
    # keeps the user's prompt/model edits across strategy switches.
    for key in (SESSION_PROMPT, SESSION_PROMPT_PRESET, SESSION_LLM_MODEL, SESSION_MAX_TOKENS):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    render_sidebar()
    render_upload_header()

    uploads = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if uploads:
        if len(uploads) > MAX_FILES:
            st.error(f"Maximum {MAX_FILES} files.")
        else:
            fingerprint = tuple((u.name, u.size) for u in uploads)
            if st.session_state.get(UPLOAD_FINGERPRINT) != fingerprint:
                repo = _repo()
                records = []
                for uploaded in uploads:
                    try:
                        records.append(register_upload(repo, uploaded.name, uploaded.getvalue()))
                    except Exception as exc:
                        records.append(
                            {
                                "filename": uploaded.name,
                                "size": getattr(uploaded, "size", 0),
                                "status": "Failed",
                                "paper_id": None,
                                "version_id": None,
                                "error": str(exc),
                                "canonical_doc": None,
                                "extractions": {},
                            }
                        )
                st.session_state[SESSION_DOCUMENTS] = records
                st.session_state[UPLOAD_FINGERPRINT] = fingerprint

    documents: list = st.session_state.get(SESSION_DOCUMENTS) or []
    render_document_table(documents)

    if documents and st.button("Process Documents"):
        repo = _repo()
        st.session_state[SESSION_DOCUMENTS] = ingest_records(repo, documents)
        documents = st.session_state[SESSION_DOCUMENTS]

    if documents:
        render_batch_status(documents)
        ready = [d for d in documents if d.get("status") == "Ready"]
        st.markdown(f"Documents ready: {len(ready)}")

        statuses = availability(api_key=_applied_api_key(), model=st.session_state.get(SESSION_LLM_MODEL))
        render_extractor_availability(statuses)

        available_ids = [s["id"] for s in statuses if s["available"]]
        labels = {s["id"]: s["label"] for s in statuses}
        if not available_ids:
            st.error("No extractors are currently available.")
            return

        st.markdown("Choose extraction strategy:")
        selected_label = st.radio(
            "Choose extraction strategy:",
            [labels[i] for i in available_ids],
            index=0,
            label_visibility="collapsed",
        )
        selected_id = next(i for i in available_ids if labels[i] == selected_label)
        render_extractor_config(statuses, selected_id)
        if selected_id in ("llm", "hybrid"):
            render_prompt_editor()

        locked = [s for s in statuses if not s["available"]]
        if locked:
            st.caption("Unavailable until an API key is applied: " + ", ".join(s["label"] for s in locked))

        if st.button("Run Extraction"):
            repo = _repo()
            llm_options = _llm_options() if selected_id in ("llm", "hybrid") else None
            with st.spinner("Running extraction… (LLM calls can take up to a minute per paper)"):
                st.session_state[SESSION_DOCUMENTS] = extract_records(
                    repo, documents, selected_id, api_key=_applied_api_key(), llm_options=llm_options
                )
            documents = st.session_state[SESSION_DOCUMENTS]

        for rec in documents:
            attempt = (rec.get("extractions") or {}).get(selected_id)
            if attempt:
                render_extraction_result(rec, selected_id, STRATEGY_LABELS.get(selected_id, selected_id))


if __name__ == "__main__":
    render_app()
