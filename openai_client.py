import httpx
import requests
from openai import OpenAI
from environs import Env, EnvError


class OpenAIClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Загрузка переменных окружения
        self.env = Env()
        self.env.read_env()
        try:
            self.openai_api_key = self.env('OPENAI_API_KEY')
            self.proxy = self.env('PROXY')
        except EnvError:
            self.openai_api_key = input("Введите OPENAI_API_KEY: ")
            self.proxy = input("Введите PROXY (http://username:password@url:port): ")

        # Настройка прокси для httpx клиента
        session_requests = requests.Session()
        session_requests.proxies = {'http://': self.proxy, 'https://': self.proxy}
        self.http_client = httpx.Client(proxies=session_requests.proxies)

        if not hasattr(self, 'initialized'):
            self.initialized = True
            print("Инициализация OpenAIClient")

    def get_client(self):
        return OpenAI(api_key=self.openai_api_key, http_client=self.http_client)

    def get_completion(self, prompt):
        client_instance = self.get_client()
        completion = client_instance.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user",
                       "content": prompt.messages[0].content.text}],
            temperature=0.1
        )
        return completion