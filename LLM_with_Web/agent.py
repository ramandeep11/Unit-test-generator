from duckduckgo_search import DDGS
from selenium import webdriver
import requests
from bs4 import BeautifulSoup
from langchain_community.llms import Ollama
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate


# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = Ollama(model="deepseek-r1")
class Agent:
    def __init__(self):
        self.ddgs = DDGS()

    def search(self, query, region):
        return self.ddgs.text(query, region, max_results=1)

    def example_search(self, company_name, region):
        results = self.search(f"What the are current compaigns for {company_name}", region)
        print(results[0])
        print(results[0].href)
        print(results[0].body)

class scrape_website:
    def __init__(self):
        # self.browser = webdriver.Chrome()
        self.agent = Agent()
        self.prompt = PromptTemplate(
            template="""
            You are a helpful assistant that can scrape a website and return the content of the website.
            Here is the content of the website:
            {content}
            Return single line from each website related to the compains for {company_name}
            Do not return any other text than the single line.
            And try to keep it short and concise.
            """,
            input_variables=["content", "company_name"]
        )

    def scrape(self, url):
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        text = ' '.join(chunk for chunk in lines if chunk)
        return text

    def get_compaign_from_website_content(self, company_name, region):
        chain = LLMChain(llm=llm, prompt=self.prompt)
        websites = self.agent.search(f"What are the current campaigns for {company_name}", region)
        results = []
        for website in websites:
            content = self.scrape(website['href'])
            result = chain.run({"content": content, "company_name": company_name})
            results.append(result)
        return "\n".join(results)





