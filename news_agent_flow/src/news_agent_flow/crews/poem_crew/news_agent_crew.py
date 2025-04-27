from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from dotenv import load_dotenv

load_dotenv()

@CrewBase
class NewsAgentCrew():
    """NewsAgent crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def retrieve_news(self) -> Agent:
        return Agent(
            config=self.agents_config['retrieve_news'],
            tools=[SerperDevTool()],
            verbose=True
        )

    @agent
    def website_scraper(self) -> Agent:
        return Agent(
            config=self.agents_config['website_scraper'],
            tools=[ScrapeWebsiteTool()],
            verbose=True
        )
        
    @agent
    def ai_news_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['ai_news_writer'],
            verbose=True
        )

    @task
    def retrieve_news_task(self) -> Task:
        return Task(
            config=self.tasks_config['retrieve_news_task'],
        )

    @task
    def website_scrape_task(self) -> Task:
        return Task(
            config=self.tasks_config['website_scrape_task'],
        )
    
    @task
    def ai_news_write_task(self) -> Task:
        return Task(
            config=self.tasks_config['ai_news_write_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the NewsAgent crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )