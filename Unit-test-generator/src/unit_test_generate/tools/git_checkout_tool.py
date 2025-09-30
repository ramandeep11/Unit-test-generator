from typing import Type
import subprocess
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class GitCheckoutToolInput(BaseModel):
    """Input schema for GitCheckoutTool."""
    repo_path: str = Field(..., description="The local path of the Git repository")
    branch_name: str = Field(..., description="The name of the branch to checkout or create")

class GitCheckoutTool(BaseTool):
    name: str = "Git Checkout Tool"
    description: str = (
        "A tool to checkout to an existing branch or create a new branch in a Git repository. "
        "If the branch does not exist, it will be created."
    )
    args_schema: Type[BaseModel] = GitCheckoutToolInput

    def _run(self, repo_path: str, branch_name: str) -> str:
        try:
            # Check if it's a valid git repository
            result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                                     cwd=repo_path, 
                                     capture_output=True, 
                                     text=True)
            if result.returncode != 0:
                return "Invalid Git repository"

            # Check if the branch already exists
            branch_check = subprocess.run(['git', 'branch', '--list', branch_name], 
                                          cwd=repo_path, 
                                          capture_output=True, 
                                          text=True)
            if branch_check.stdout.strip():
                # Branch exists, checkout to it
                checkout_result = subprocess.run(['git', 'checkout', branch_name], 
                                                 cwd=repo_path, 
                                                 capture_output=True, 
                                                 text=True)
                if checkout_result.returncode == 0:
                    return f"Successfully checked out to existing branch: {branch_name}"
                else:
                    return f"Failed to checkout to branch: {checkout_result.stderr}"
            else:
                # Branch does not exist, create and checkout to it
                create_branch_result = subprocess.run(['git', 'checkout', '-b', branch_name], 
                                                      cwd=repo_path, 
                                                      capture_output=True, 
                                                      text=True)
                if create_branch_result.returncode == 0:
                    return f"Successfully created and checked out to new branch: {branch_name}"
                else:
                    return f"Failed to create and checkout to branch: {create_branch_result.stderr}"

        except FileNotFoundError:
            return f"Error: Path {repo_path} does not exist"
        except Exception as e:
            return f"Operation failed: {str(e)}"