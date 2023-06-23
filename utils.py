import os
import re
from types import ModuleType
from typing import List

import openai
import streamlit as st
from vega_datasets import data as vega_data

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")


def page_setup() -> None:
    st.secrets.load_if_toml_exists()

    st.set_page_config(
        page_title="ChartGPT",
        page_icon="📈",
        initial_sidebar_state="collapsed" if openai.api_key else "expanded",
    )

    PASSCODE = os.environ.get("PASSCODE")
    OPENAPI_API_KEY = os.environ.get("OPENAPI_API_KEY")

    if OPENAPI_API_KEY:
        openai.api_key = OPENAPI_API_KEY

    if PASSCODE and "authenticate" not in st.session_state:
        if st.text_input("What is the passcode?", type="password") == PASSCODE:
            st.session_state["authenticate"] = True
            st.experimental_rerun()
        else:
            st.stop()

    # Fix the buttons to always only use a single line:
    st.markdown(
        """
    <style>
    div[data-testid="tooltipHoverTarget"] > button {
        white-space: nowrap;
        text-overflow: ellipsis;
        width: 100%;
        overflow: hidden;
        justify-content: flex-start !important;
    }
    div[data-testid="tooltipHoverTarget"] {
        width: 100%;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.session_state["openai_model"] = st.selectbox(
            "OpenAI Chat Model",
            options=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"],
            help="Make sure that the provided API key is activated for the selected model.",
        )
        if not OPENAPI_API_KEY:
            openai.api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="You can find your API keys [here](https://platform.openai.com/account/api-keys).",
            )


def name_to_variable(name: str) -> str:
    """Converts a name to a valid snake_case variable name."""
    name = _RE_COMBINE_WHITESPACE.sub(" ", name).strip().replace(" ", "_").lower()
    # Remove all non-alphanumeric characters
    name = re.sub(r"[^A-Za-z0-9_]+", "", name)
    return name


def load_code_as_module(code: str) -> ModuleType:
    import importlib.util

    spec = importlib.util.spec_from_loader("loaded_module", loader=None)
    assert spec is not None
    loaded_module = importlib.util.module_from_spec(spec)
    exec(code, loaded_module.__dict__)
    return loaded_module


def get_toy_datasets() -> List[str]:
    IGNORLIST = [
        "earthquakes",
        "ffox",
        "gimp",
        "graticule",
        "world-110m",
        "us-10m",
        "movies",
        "miserables",
        "londonTubeLines",
        "londonBoroughs",
        "annual-precip",
        "7zip",
    ]

    TOY_DATASETS = []
    try:
        TOY_DATASETS = [
            dataset_name
            for dataset_name in vega_data.list_datasets()
            if dataset_name not in IGNORLIST
        ]
    except Exception:
        pass

    return TOY_DATASETS
