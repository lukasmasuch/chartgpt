import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict

import openai
import pandas as pd
import streamlit as st
import tiktoken


class BaseTask(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the task (library/framework)."""
        return NotImplemented

    @abstractmethod
    def get_task_prompt(
        self,
        task_name: str,
        description: str,
    ) -> str:
        return NotImplemented

    @abstractmethod
    def handle_output(self, task_output: Any) -> Optional[pd.DataFrame]:
        pass


_PLOTLY_PROMPT = """
import pandas as pd
import numpy as np
from typing import *
import plotly

def {method_name}(data: pd.DataFrame) -> Union["plotly.graph_objs.Figure", "plotly.graph_objs.Data"]:
    \"\"\"
    {description}

    Visualize the data using the Plotly library.

    Args:
        data (pd.DataFrame): The data to visualize.

    Returns:
        Union[plotly.graph_objs.Figure, plotly.graph_objs.Data]: The plotly visualization figure.
    \"\"\"
    # All task-specific imports here:
    import plotly

    {{Task implementation}}
"""


class PlotlyVisualizationTask(BaseTask):
    """A task that visualizes data using Plotly."""

    @property
    def name(self) -> str:
        return "Plotly"

    def get_task_prompt(
        self,
        task_name: str,
        description: str,
    ) -> str:
        return _PLOTLY_PROMPT.format(
            method_name=task_name,
            description=description,
        )

    def handle_output(self, task_output: Any) -> None:
        import streamlit as st

        st.plotly_chart(task_output, use_container_width=True)


_MATPLOTLIB_PROMPT = """
import pandas as pd
import numpy as np
from typing import *
from matplotlib.figure import Figure

def {method_name}(data: pd.DataFrame) -> "Figure":
    \"\"\"
    {description}

    Visualize the data using the Matplotlib library.

    Args:
        data (pd.DataFrame): The data to visualize.

    Returns:
        matplotlib.figure.Figure: The matplotlib visualization figure.
    \"\"\"
    # All task-specific imports here:
    import matplotlib

    {{Task implementation}}
"""


class MatplotlibVisualizationTask(BaseTask):
    """A task that visualizes data using Matplotlib."""

    @property
    def name(self) -> str:
        return "Matplotlib"

    def get_task_prompt(
        self,
        task_name: str,
        description: str,
    ) -> str:
        return _MATPLOTLIB_PROMPT.format(
            method_name=task_name,
            description=description,
        )

    def handle_output(self, task_output: Any) -> None:
        import streamlit as st

        st.pyplot(task_output)


_ALTAIR_PROMPT = """

import pandas as pd
import numpy as np
from typing import *
import altair as alt

def {method_name}(data: pd.DataFrame) -> "alt.Chart":
    \"\"\"
    {description}

    Visualize the data using the Altair library.

    Args:
        data (pd.DataFrame): The data to visualize.

    Returns:
        alt.Chart: The altair visualization chart.
    \"\"\"
    # All task-specific imports here:
    import altair as alt

    {{Task implementation}}
"""


class AltairVisualizationTask(BaseTask):
    """A task that visualizes data using Altair."""

    @property
    def name(self) -> str:
        return "Altair"

    def get_task_prompt(
        self,
        task_name: str,
        description: str,
    ) -> str:
        return _ALTAIR_PROMPT.format(
            method_name=task_name,
            description=description,
        )

    def handle_output(self, task_output: Any) -> None:
        import streamlit as st

        st.altair_chart(task_output, use_container_width=True)


TOOLS: List[BaseTask] = [
    PlotlyVisualizationTask(),
    AltairVisualizationTask(),
    MatplotlibVisualizationTask(),
]


def extract_first_code_block(markdown_text: str) -> str:
    if (
        "```" not in markdown_text
        and "return" in markdown_text
        and "def" in markdown_text
    ):
        # Assume that everything is code
        return markdown_text

    pattern = r"(?<=```).+?(?=```)"

    # Use re.DOTALL flag to make '.' match any character, including newlines
    first_code_block = re.search(pattern, markdown_text, flags=re.DOTALL)

    code = first_code_block.group(0) if first_code_block else ""
    code = code.strip().lstrip("python").strip()
    return code


def num_tokens_from_messages(
    messages: List[dict], model: str = "gpt-3.5-turbo-0301"
) -> int:
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )


def get_df_types_info(df: pd.DataFrame = None) -> str:
    column_info = {
        "dtype": [str(dtype) for dtype in df.dtypes],
        "inferred dtype": [
            pd.api.types.infer_dtype(column) for _, column in df.items()
        ],
        "missing values": df.isna().sum().to_list(),
    }

    return pd.DataFrame(
        column_info,
        index=df.columns,
    ).to_csv()


def get_column_descriptions_info(column_desc_df: pd.DataFrame) -> str:
    return "Column descriptions: \n\n" + column_desc_df.to_csv()


def get_df_stats_info(df: pd.DataFrame) -> str:
    return df.describe().to_csv()


def get_df_sample_info(df: pd.DataFrame, sample_size: int = 20) -> str:
    return df.sample(min(len(df), sample_size), random_state=1).to_csv()


def get_dataset_description_prompt(
    dataset_df: pd.DataFrame,
    column_desc_df: Optional[pd.DataFrame] = None,
    light: bool = False,
) -> str:
    prompt = f"""
I have a dataset (as Pandas DataFrame) that contains data with the following characteristics:

Column data types:

{get_df_types_info(dataset_df)}

{get_column_descriptions_info(column_desc_df) if column_desc_df is not None else ""}

Data sample:

{get_df_sample_info(dataset_df, sample_size=10 if light else 20)}
    """

    if light:
        return prompt

    return (
        prompt
        + f"""
Data statistics:

{get_df_stats_info(dataset_df)}
    """
    )


class DataDescription(TypedDict):
    data_description: str
    columns: List[Dict[str, str]]
    observations: List[str]
    chart_ideas: List[str]


@st.cache_data(show_spinner="Analyzing your data...")
def get_data_description(
    dataset_df: pd.DataFrame,
    openai_model: str = "gpt-3.5-turbo",
) -> DataDescription:
    user_prompt = f"""
{get_dataset_description_prompt(dataset_df)}

Please provide the following information based on this JSON template:

{{
    "data_description": "{{A concise description of the dataset}}",
    "columns": [
        {{ "name": "{{column name}}", "description": "{{A short description about the column}}" }},
    ]
    "observations": [
        "{{Interesting or surprising observations about the data}}",
    ]
    "chart_ideas": [
        "{{Concise description of your best chart or visualization idea}}",
        "{{Concise description of your second best chart or visualization idea}}",
        "{{Concise description of the third best chart or visualization idea}}",
        "{{Concise description of the fourth best chart or visualization idea}}",
    ]
}}

Please only respond with a valid JSON and fill out all {{placeholders}}:
    """

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps with describing data. The user will share some information about a Pandas dataframe, and you will create a description of the data based on a user-provided JSON template. Please provide your answer as valid JSON.",
        },
        {"role": "user", "content": user_prompt},
    ]
    print("Prompt tokens", num_tokens_from_messages(messages))

    completion = openai.ChatCompletion.create(model=openai_model, messages=messages)
    return json.loads(completion.choices[0].message.content)


class TaskSummary(TypedDict):
    name: str
    method_name: str
    task_description: str
    emoji: str


@st.cache_data(show_spinner=False)
def get_task_summary(
    dataset_df: pd.DataFrame,
    task_instruction: str,
    openai_model: str = "gpt-3.5-turbo",
    column_desc_df: Optional[pd.DataFrame] = None,
) -> TaskSummary:
    user_prompt = f"""
{get_dataset_description_prompt(dataset_df, column_desc_df)}

Based on this data, I have the following task instruction:

{task_instruction}

Please provide me the following information based on the task instruction:

{{
    "name": "{{Short Task Name}}",
    "method_name": "{{python_method_name}}",
    "task_description": "{{a concise description of the task}}",
    "emoji": "{{a descriptive emoji}}",
}}

The JSON needs to be compatible with the following TypeDict:

```python
class TaskSummary(TypedDict):
    name: str
    method_name: str
    task_description: str
    emoji: str
```

Please only respond with a valid JSON:
    """

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. The user will share some characteristics of a dataset and a data exploration or visualization task instructions that the user likes to perform on this dataset. You will create some descriptive information about the task instruction based on a provided JSON template. Please only respond with a valid JSON.",
        },
        {"role": "user", "content": user_prompt},
    ]
    print("Prompt tokens", num_tokens_from_messages(messages))
    completion = openai.ChatCompletion.create(model=openai_model, messages=messages)
    return json.loads(completion.choices[0].message.content)


@st.cache_data(show_spinner=False)
def get_task_code(
    dataset_df: pd.DataFrame,
    task_instruction: str,
    task_code_template: str,
    task_library: str,
    openai_model: str = "gpt-3.5-turbo",
    column_desc_df: Optional[pd.DataFrame] = None,
) -> str:
    user_prompt = f"""
{get_dataset_description_prompt(dataset_df, column_desc_df)}

Based on this data, please create a function that performs the following Data Visualization task with {task_library}:

{task_instruction}

Please use the following template and implement all {{placeholders}}:

```
{task_code_template}
```

Implement all {{placeholders}}, make sure that the Python code is valid. Please put the code in a markdown code block (```) and ONLY respond with the code:
    """

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps create python functions to perform analysis, visualization, processing, or other tasks on a Pandas dataframe. The user will provide a description about the dataframe as well as a code template and instructions. Please only answer with valid Python code.",
            # "content": "You are a helpful assistant that helps create python functions to perform data visualizations on a Pandas dataframe. The user will provide a description about the dataframe as well as a code template and instructions. Please only answer with valid Python code.",
        },
        {"role": "user", "content": user_prompt},
    ]

    print("Prompt tokens", num_tokens_from_messages(messages))
    completion = openai.ChatCompletion.create(model=openai_model, messages=messages)
    response = completion.choices[0].message.content.strip()
    code_block = extract_first_code_block(response)
    if not code_block and response:
        st.info(response, icon="🤖")
    return code_block


@st.cache_data(show_spinner=False)
def get_fixed_code(
    task_code: str,
    exception_message: str,
    dataset_df: pd.DataFrame,
    column_desc_df: Optional[pd.DataFrame] = None,
    openai_model: str = "gpt-3.5-turbo",
) -> str:
    user_prompt = f"""
{get_dataset_description_prompt(dataset_df, column_desc_df, light=True)}

Based on this data, I implemented the following code:

```python
{task_code}
```

But it throws the following exception:

{exception_message}

Please fix the code but don't change any method or class names. Put the code in a markdown code block (```) and only respond with entire code that includes the fixes:
"""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps to fix python functions to perform analysis, visualization, processing, or other tasks on a Pandas dataframe. The user will provide a function implementation and an exception message. Your job is it to fix the code so that it doesn't run into this exception. Please only answer with valid Python code.",
        },
        {"role": "user", "content": user_prompt},
    ]

    print("Prompt tokens", num_tokens_from_messages(messages))
    completion = openai.ChatCompletion.create(model=openai_model, messages=messages)
    response = completion.choices[0].message.content.strip()
    code_block = extract_first_code_block(response)
    if not code_block and response:
        st.info(response, icon="🤖")
    print(code_block)
    return code_block


@st.cache_data(show_spinner=False)
def get_modified_code(
    task_code: str,
    instruction: str,
    dataset_df: pd.DataFrame,
    column_desc_df: Optional[pd.DataFrame] = None,
    openai_model: str = "gpt-3.5-turbo",
) -> str:
    user_prompt = f"""
{get_dataset_description_prompt(dataset_df, column_desc_df, light=True)}

Based on this data, I implemented the following python function:

```python
{task_code}
```

Please modify this code based on the following instruction:

```
{instruction}
```

Keep the function name and config class name. Only respond with entire code that includes the modifications:
"""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps to modify python functions that perform analysis, visualization, processing, or other tasks on a Pandas dataframe. The user will provide a function implementation and a modification instruction. Please modify the code based on the instruction and only answer with valid Python code.",
        },
        {"role": "user", "content": user_prompt},
    ]

    print("Prompt tokens", num_tokens_from_messages(messages))
    completion = openai.ChatCompletion.create(model=openai_model, messages=messages)
    response = completion.choices[0].message.content.strip()
    code_block = extract_first_code_block(response)
    if not code_block and response:
        st.info(response, icon="🤖")
    return code_block
