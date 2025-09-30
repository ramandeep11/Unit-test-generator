from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pydantic import BaseModel

from unit_test_generate.tools.directory_list_tool import DirectoryListTool

import os

class ServiceClassesList(BaseModel):
    service_classes: List[str]

@CrewBase
class ServiceClassesListCrew():
    """ServiceClassesListCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    directoryListTool = DirectoryListTool()

    llm = LLM(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    @agent
    def service_classes_lister_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['service_classes_lister_agent'],
            verbose=True,
            llm=self.llm,
            tools=[self.directoryListTool],
        )
    
    @task
    def service_classes_list_task(self) -> Task:
        return Task(
            config=self.tasks_config['service_classes_list_task'],
            output_pydantic=ServiceClassesList
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ServiceClassesListCrew crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
