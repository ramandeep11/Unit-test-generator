from typing import Type
import subprocess
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class GitToolInput(BaseModel):
    """Input schema for GitTool."""
    repo_path: str = Field(..., description="The local path of the Git repository")
    operation: str = Field(..., description="The git operation to perform: 'status', 'commit', 'add'")
    message: str = Field(None, description="Commit message (required for commit operation)")
    files: list = Field(None, description="List of files to add (required for add operation)")

class GitTool(BaseTool):
    name: str = "Git Repository Operations Tool"
    description: str = (
        "A tool to perform git operations (status, commit, add) on a Git repository " \
        "status -> Get the current status of the repository " \
        "commit -> Commit changes with a message " \
        "add -> Add files to staging area " \
    )
    args_schema: Type[BaseModel] = GitToolInput

    def _run(self, repo_path: str, operation: str, message: str = None, files: list = None) -> str:
        try:
            # Check if it's a valid git repository
            result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                                 cwd=repo_path, 
                                 capture_output=True, 
                                 text=True)
            if result.returncode != 0:
                return "Invalid Git repository"

            if operation.lower() == "status":
                return self._get_status(repo_path)
            elif operation.lower() == "commit":
                if not message:
                    return "Error: Commit message is required"
                return self._commit_changes(repo_path, message)
            elif operation.lower() == "add":
                if not files:
                    return "Error: Files list is required for add operation"
                return self._add_files(repo_path, files)
            else:
                return "Error: Invalid operation. Use 'status', 'commit', 'add'"

        except FileNotFoundError:
            return f"Error: Path {repo_path} does not exist"
        except Exception as e:
            return f"Operation failed: {str(e)}"

    def _get_status(self, repo_path: str) -> str:
        # Get current branch
        branch_result = subprocess.run(['git', 'branch', '--show-current'], 
                                    cwd=repo_path, 
                                    capture_output=True, 
                                    text=True)
        current_branch = branch_result.stdout.strip()

        # Get status
        status_result = subprocess.run(['git', 'status'], 
                                    cwd=repo_path, 
                                    capture_output=True, 
                                    text=True)
        
        # Get untracked files
        untracked_result = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard'], 
                                        cwd=repo_path, 
                                        capture_output=True, 
                                        text=True)
        
        status_message = f"Current branch: {current_branch}\n\n"
        status_message += f"Git Status:\n{status_result.stdout}"
        
        if untracked_result.stdout:
            status_message += f"\nUntracked files:\n"
            for file in untracked_result.stdout.splitlines():
                status_message += f"- {file}\n"
        
        return status_message

    def _commit_changes(self, repo_path: str, message: str) -> str:
        result = subprocess.run(['git', 'commit', '-m', message], 
                              cwd=repo_path, 
                              capture_output=True, 
                              text=True)
        if result.returncode == 0:
            return f"Successfully committed changes with message: {message}"
        else:
            return f"Failed to commit: {result.stderr}"

    def _add_files(self, repo_path: str, files: list) -> str:
        cmd = ['git', 'add'] + files
        result = subprocess.run(cmd, 
                              cwd=repo_path, 
                              capture_output=True, 
                              text=True)
        if result.returncode == 0:
            return f"Successfully added files: {', '.join(files)}"
        else:
            return f"Failed to add files: {result.stderr}"