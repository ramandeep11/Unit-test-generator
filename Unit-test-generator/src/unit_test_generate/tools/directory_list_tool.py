import os
from typing import Type, List

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class DirectoryListToolInput(BaseModel):
    """Input schema for DirectoryListTool."""

    path: str = Field(..., description="The directory path to list contents of.")


class DirectoryListTool(BaseTool):
    name: str = "Directory List Tool"
    description: str = (
        "A tool to list immediate directories and files in a given path."
    )
    args_schema: Type[BaseModel] = DirectoryListToolInput

    def _run(self, path: str) -> List[str]:
        """
        Lists the immediate directories and files in the given path.

        Args:
            path (str): The directory path to list contents of.

        Returns:
            List[str]: A list of immediate directories and files in the path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path '{path}' does not exist.")
        if not os.path.isdir(path):
            raise NotADirectoryError(f"The path '{path}' is not a directory.")

        return os.listdir(path)