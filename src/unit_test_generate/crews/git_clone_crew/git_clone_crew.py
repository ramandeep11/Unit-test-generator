from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os

from unit_test_generate.tools.git_clone_tool import GitCloneTool
from unit_test_generate.tools.directory_list_tool import DirectoryListTool
from unit_test_generate.tools.project_build_tool import ProjectBuildTool

@CrewBase
class GitCloneCrew():
    """GitCloneCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    llm = LLM(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    GitCloneTool = GitCloneTool()
    DirectoryListTool = DirectoryListTool()
    ProjectBuildTool = ProjectBuildTool()

    @agent
    def git_cloner_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['git_cloner_agent'],
            verbose=True,
            llm=self.llm,
            tools=[self.GitCloneTool, self.DirectoryListTool],
        )
    
    @agent
    def project_builder_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['project_builder_agent'],
            verbose=True,
            llm=self.llm,
            tools=[self.DirectoryListTool, self.ProjectBuildTool],
        )

    @task
    def git_clone_task(self) -> Task:
        return Task(
            config=self.tasks_config['git_clone_task'],
            verbose=True,
            tools=[self.GitCloneTool, self.DirectoryListTool],
            agent = self.git_cloner_agent(),
        )
    
    @task
    def project_build_task(self) -> Task:
        return Task(
            config=self.tasks_config['project_build_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the GitCloneCrew crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
