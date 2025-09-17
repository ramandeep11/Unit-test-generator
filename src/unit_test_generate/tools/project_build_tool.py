from typing import Type
import subprocess
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class ProjectBuildToolInput(BaseModel):
    """Input schema for ProjectBuildTool."""

    build_tool: str = Field(..., description="The build tool to use (e.g., maven, gradle, npm, yarn, make).")
    path: str = Field(..., description="The path where the build command should be executed.")


class ProjectBuildTool(BaseTool):
    name: str = "Project Build Tool"
    description: str = (
        "Executes the relevant build command for a given build tool (e.g., maven, gradle, npm, yarn, make) "
        "on the specified path and returns the build result."
    )
    args_schema: Type[BaseModel] = ProjectBuildToolInput
    cache: bool = False

    def _run(self, build_tool: str, path: str) -> str:
        build_commands = {
            "maven": "mvn clean install",
            "gradle": "gradle build",
            "npm": "npm install && npm run build",
            "yarn": "yarn install && yarn build",
            "make": "make"
        }

        if build_tool not in build_commands:
            return f"Unsupported build tool: {build_tool}. Supported tools are: {', '.join(build_commands.keys())}."

        command = build_commands[build_tool]
        try:
            result = subprocess.run(
                command,
                cwd=path,
                shell=True,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return f"Build succeeded:\n{result.stdout}"
        except subprocess.CalledProcessError as e:
            return f"Build failed with error:\n{e.stderr}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"