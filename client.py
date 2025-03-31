import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai_client import OpenAIClient

# Настройка параметров MCP-сервера
server_params = StdioServerParameters(
    command="python",
    args=["server.py"],
    env=None,
    debug=False
)

openai_client = OpenAIClient()
client = openai_client.get_client()

async def run():
    # Устанавливаем соединение с MCP-сервером
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Получаем список доступных инструментов
            tools = await session.list_tools()
            tools_names = [tool.name for tool in tools.tools]

            while True:
                # Запрашиваем вопрос у пользователя через консоль
                user_query = input("Введите ваш вопрос: ")

                # Получаем prompt для выбора инструмента
                prompt = await session.get_prompt(
                    name="tool-choicer",
                    arguments={"user_query": user_query,
                               "tools": ", ".join(tools_names)}
                )

                # Отправляем запрос к LLM, используя полученный prompt
                choicer_response = openai_client.get_completion(prompt=prompt)
                response = choicer_response.choices[0].message.content
                data = json.loads(response)
                tool_name = data.get("use_tool")
                print(f"Выбранный инструмент: {tool_name}")

                # Получаем prompt для выбранного инструмента
                tool_prompt = await session.get_prompt(
                    name=tool_name,
                    arguments={"user_query": user_query}
                )

                tool_response = openai_client.get_completion(prompt=tool_prompt)
                response = tool_response.choices[0].message.content
                tool_args = json.loads(response)

                # Вызываем инструмент и получаем результаты
                data_result = await session.call_tool(tool_name, arguments=tool_args)
                result = data_result.content[0].text
                print("Результат вызова инструмента:", result[:300])

                # Передаём данные для аналитики
                analyzer_prompt = await session.get_prompt(
                    name="analyzer",
                    arguments={"user_query": user_query, "data_result": result}
                )
                analyzer_response = openai_client.get_completion(prompt=analyzer_prompt)

                print("\nОтвет аналитика:")
                print(analyzer_response.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(run())
