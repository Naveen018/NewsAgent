#!/usr/bin/env python
import os
import random
import requests
import asyncio
import re
from datetime import datetime

from pydantic import BaseModel

from crewai.flow import Flow, listen, start
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from dotenv import load_dotenv

from crews.poem_crew.news_agent_crew import NewsAgentCrew

load_dotenv()

class NewsState(BaseModel):
    topic: str = ""
    urls: list = []
    content: str = ""
    article: str = ""
    filename: str = ""

class NewsFlow(Flow[NewsState]):
    def __init__(self):
        super().__init__()
        self.crew = NewsAgentCrew()

    @start()
    def generate_topic(self):
        """Fetch a trending topic from NewsAPI or fallback to varied list"""
        print("Generating trending topic")
        api_key = os.getenv("NEWSAPI_KEY")
        try:
            url = f"https://newsapi.org/v2/top-headlines?category=technology&apiKey={api_key}"
            response = requests.get(url)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            if articles:
                topics = [article["title"].split(" - ")[0] for article in articles[:5]]
                self.state.topic = random.choice(topics)
            else:
                raise ValueError("No articles found")
        except Exception as e:
            print(f"Failed to fetch topic: {e}")
            fallback_topics = [
                "Artificial Intelligence",
                "Blockchain Technology",
                "Climate Change Solutions",
                "Quantum Computing",
                "Biotechnology Advances"
            ]
            self.state.topic = random.choice(fallback_topics)
        print(f"Topic selected: {self.state.topic}")

    @listen(generate_topic)
    def retrieve_news(self):
        """Run retrieve_news_task to get 10 URLs with retries"""
        print(f"Retrieving news for topic: {self.state.topic}")
        task = self.crew.retrieve_news_task()
        agent = self.crew.retrieve_news()
        max_retries = 2
        attempt = 0

        while attempt <= max_retries:
            crew = self.crew.crew()
            crew.tasks = [task]
            crew.agents = [agent]
            result = crew.kickoff(inputs={"topic": self.state.topic})
            # print(result)
            url_matches = re.findall(r'\(https?://[^\s)]+\)', result.raw)
            # Remove parentheses
            urls = [url.strip('()') for url in url_matches if url.strip('()')]
            self.state.urls = urls
            print(f"Retrieved {self.state.urls} URLs")
            if len(self.state.urls) >= 10:
                print(f"Retrieved {len(self.state.urls)} URLs")
                break
            attempt += 1
            print(f"Only {len(self.state.urls)} URLs found, retrying ({attempt}/{max_retries})...")
        
        if len(self.state.urls) < 10:
            print(f"Final attempt: {len(self.state.urls)} URLs retrieved, proceeding...")

    @listen(retrieve_news)
    async def scrape_websites(self):
        """Run website_scrape_task in parallel for all URLs"""
        print(f"Scraping {len(self.state.urls)} websites in parallel")
        task = self.crew.website_scrape_task()
        agent = self.crew.website_scraper()

        async def scrape_url(url):
            try:
                loop = asyncio.get_event_loop()
                crew = self.crew.crew()
                crew.tasks = [task]
                crew.agents = [agent]
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.crew.crew().kickoff(
                            inputs={"url": url, "topic": self.state.topic},
                        )
                    ),
                    timeout=30.0
                )
                return result
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")
                return ""

        scrape_tasks = [scrape_url(url) for url in self.state.urls]
        contents = await asyncio.gather(*scrape_tasks, return_exceptions=True)
        print(contents)
        valid_contents = [c for c in contents if isinstance(c, str) and c.strip()]
        self.state.content = "\n".join(valid_contents)
        if not self.state.content.strip():
            print("No content scraped, proceeding with empty content...")

    @listen(scrape_websites)
    def write_article(self):
        """Run ai_news_write_task"""
        print("Writing article")
        task = self.crew.ai_news_write_task()
        agent = self.crew.ai_news_writer()
        crew = self.crew.crew()
        crew.tasks = [task]
        crew.agents = [agent]
        result = crew.kickoff(
            inputs={"content": self.state.content, "topic": self.state.topic}
        )
        print(result.raw)
        self.state.article = result.raw if result.raw else ""
        # print(f"Article written: {self.state.article}...")  # Print first 50 chars

    @listen(write_article)
    def validate_article(self):
        """Ensure article quality"""
        print("Validating article")
        word_count = len(self.state.article.split())
        if word_count < 200:
            print(f"Article too short ({word_count} words), rewriting...")
        else:
            self.state.filename = f"news/{self.state.topic}_news_article.md"
            os.makedirs("news", exist_ok=True)

    @listen(validate_article)
    def save_article(self):
        """Save article to file"""
        print(f"Saving article to {self.state.filename}")
        with open(self.state.filename, "w") as f:
            f.write(self.state.article)

def kickoff():
    flow = NewsFlow()
    flow.kickoff()

def plot():
    flow = NewsFlow()
    flow.plot()

if __name__ == "__main__":
    kickoff()