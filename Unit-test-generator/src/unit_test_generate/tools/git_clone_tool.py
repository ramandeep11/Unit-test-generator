from typing import Type
import git

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class GitCloneToolInput(BaseModel):
    """Input schema for GitCloneTool."""
    git_url: str = Field(..., description="The URL of the Git repository to clone.")
    git_branch: str = Field(..., description="The branch of the Git repository to clone.")
    clone_path: str = Field(..., description="The local path where the repository should be cloned.")


class GitCloneTool(BaseTool):
    name: str = "Cloning a git repostory from a url to a specified path"
    description: str = (
        "A tool to clone a Git repository from a given URL and branch to a specified local path."
    )
    args_schema: Type[BaseModel] = GitCloneToolInput

    def _run(self, git_url: str, git_branch: str, clone_path: str) -> str:
        """
        Clones a Git repository from a given URL and branch to a specified path.

        Args:
            git_url (str): The URL of the Git repository to clone.
            git_branch (str): The branch of the Git repository to clone.
            clone_path (str): The local path where the repository should be cloned.

        Returns:
            str: A message indicating success or failure of the clone operation.
        """
        try:
            repo = git.Repo.clone_from(git_url, clone_path)
            
            if git_branch != "main" and git_branch != "master":
                repo.git.checkout(git_branch)
            
            if repo.working_dir:
                print(f"Project cloned to: {repo.working_dir}")
                return f"Repository cloned successfully to {clone_path}"
            else:
                return "Failed to clone repository: Working directory not found"
                
        except git.GitCommandError as git_err:
            return f"Git error: {str(git_err)}"
        except Exception as e:
            return f"Failed to clone repository: {str(e)}"
