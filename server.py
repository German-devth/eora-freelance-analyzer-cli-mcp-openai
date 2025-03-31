import json
from typing import List, Literal, Optional

import pandas
from pandas import DataFrame
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

mcp = FastMCP("Server")


# utils
def get_csv_data() -> DataFrame:
    print("📥 Загрузка CSV данных...")
    return pandas.read_csv("data/freelancer_earnings_bd.csv")



def get_column_names() -> list[str]:
    df = pandas.read_csv("data/freelancer_earnings_bd.csv")
    return df.columns.tolist()


def load_prompt(name: str, path: str = "instructions/prompts.json") -> str:
    with open(path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts[name]


try:
    with open("instructions/tools_desc.json", "r", encoding="utf-8") as f:
        tool_descriptor = json.load(f)
except Exception as e:
    print(f"❌ Ошибка загрузки tools_desc.json: {e}")
    tool_descriptor = {}


tool_str = "\n".join(
    [f"{name}: {desc}" for name, desc in tool_descriptor.items()]
)


# Prompts
@mcp.prompt("tool-choicer")
def tool_choicer(user_query: str, tools: str) -> list[base.Message]:
    prompt_template = load_prompt("tool_choicer")
    filled_prompt = prompt_template.format(
        user_query=user_query,
        tools=tools,
        tool_str=tool_str
    )
    return [base.UserMessage(filled_prompt)]


@mcp.prompt("csv-data-decider")
def csv_data_decider_prompt(user_query: str) -> list[base.Message]:
    prompt_template = load_prompt("csv_data_decider")
    prompt_filled = prompt_template.format(
        user_query=user_query,
        column_names=", ".join(get_column_names())
    )
    return [base.UserMessage(prompt_filled)]


@mcp.prompt("csv-data-columns-getter")
def csv_data_columns_getter_prompt(user_query: str) -> list[base.Message]:
    prompt_template = load_prompt("csv_data_columns_getter")
    filled = prompt_template.format(
        user_query=user_query,
        column_names=", ".join(get_column_names())
    )
    return [base.UserMessage(filled)]


@mcp.prompt("analyzer")
def data_analyzer(user_query: str, data_result) -> list[base.Message]:
    prompt_template = load_prompt("analyzer")
    filled = prompt_template.format(
        user_query=user_query,
        data_result=data_result
    )
    return [base.UserMessage(filled)]


# Tools
@mcp.tool("csv-data-decider")
def group_and_aggregate(
        row_names: List[str],
        agg_funcs: List[Literal['mean', 'sum', 'count']],
        columns: Optional[List[str]] = None
) -> str:
    df = get_csv_data()
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    grouped = df.groupby(row_names)[numeric_columns].agg(agg_funcs)

    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()

    if columns is not None:
        if len(agg_funcs) == 1:
            # Если агрегируется одна функция,
            # добавляем суффикс к каждому базовому имени
            suffix = "_" + agg_funcs[0]
            requested_columns = row_names + [col + suffix for col in columns if
                                             col not in row_names]
        else:
            # Если агрегируется несколько функций,
            # для каждого базового имени выбираем все агрегированные варианты
            requested_columns = []
            for col in columns:
                if col in row_names:
                    requested_columns.append(col)
                else:
                    # Добавляем все колонки, которые начинаются с col + "_"
                    found = [c for c in grouped.columns if
                             c.startswith(col + "_")]
                    requested_columns.extend(found)

        # Проверяем, что все запрошенные колонки существуют
        missing = set(requested_columns) - set(grouped.columns)
        if missing:
            raise KeyError(f"Запрошенные колонки не найдены: {missing}")
        grouped = grouped[requested_columns]

    return grouped.to_json(orient='records')


@mcp.tool("csv-data-columns-getter")
def get_csvs_columns(columns: List[str]):
    df = get_csv_data()
    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"Запрошенные колонки не найдены: {missing}")

    result_df = df[columns]
    return result_df.to_json(orient="records")


if __name__ == "__main__":
    print("🚀 MCP сервер стартанул!")
    mcp.run(transport="stdio")
