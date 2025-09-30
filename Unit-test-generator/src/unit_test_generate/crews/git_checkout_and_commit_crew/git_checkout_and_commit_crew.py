from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os

from unit_test_generate.tools.git_tool import GitTool
from unit_test_generate.tools.git_checkout_tool import GitCheckoutTool

@CrewBase
class GitCheckoutAndCommitCrew():
    """GitCheckoutAndCommitCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    llm = LLM(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    @agent
    def git_checkout_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['git_checkout_agent'],
            verbose=True,
            llm=self.llm,
            tools=[GitCheckoutTool()],
        )

    @agent
    def git_commit_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['git_commit_agent'],
            verbose=True,
            llm=self.llm,
            tools=[GitTool()],
        )
    
    @task
    def git_checkout_task(self) -> Task:
        return Task(
            config=self.tasks_config['git_checkout_task'],
        )

    @task
    def git_commit_task(self) -> Task:
        return Task(
            config=self.tasks_config['git_commit_task'],
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the GitCheckoutAndCommitCrew crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
