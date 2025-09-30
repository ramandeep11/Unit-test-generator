# Unit Test Generator

An intelligent unit test generation system built with CrewAI that automatically generates, writes, and validates unit tests for Java projects.

## Overview

This project uses multiple AI agents organized into specialized crews that work together to clone repositories, generate comprehensive unit tests, validate them through builds, and commit the results to a new branch. The system iteratively refines tests based on build feedback to ensure they compile and integrate correctly.

## Architecture

The system is built using the CrewAI Flow framework, which orchestrates multiple crews in a coordinated workflow:

```
┌─────────────────────┐
│ UnitTestGenerator   │
│      Flow           │
└──────────┬──────────┘
           │
           │ @start()
           ▼
    ┌──────────────┐
    │ GitCloneCrew │
    └──────┬───────┘
           │
           │ @router()
           ▼
┌──────────────────────┐
│GenerateUnitTestCrew  │◄──┐
│  (with iteration)    │   │ @listen("failed")
└──────────┬───────────┘   │ (max 3 iterations)
           │               │
           │ success       │
           ▼               │
    ┌──────────────┐      │
    │GitCheckout   │      │
    │AndCommitCrew │──────┘
    └──────────────┘
           │
           ▼
        @listen("end")
```

## Crews

### 1. GitCloneCrew

**Purpose**: Clones the target repository and performs initial build validation.

**Process**: Sequential

**Agents**:
- `git_cloner_agent`: Clones the repository from the specified URL and branch
- `project_builder_agent`: Builds the project to ensure it's in a working state

**Tasks**:
- `git_clone_task`: Clone repository to local path
- `project_build_task`: Build project using specified build tool (Maven/Gradle)

**Tools**:
- `GitCloneTool`: Handles git clone operations with branch checkout
- `DirectoryListTool`: Lists files and directories
- `ProjectBuildTool`: Executes Maven/Gradle builds

**Output**: Cloned repository at specified path with successful initial build

---

### 2. GenerateUnitTestCrew

**Purpose**: Generates, writes, and validates unit tests for service classes.

**Process**: Hierarchical (with manager agent)

**Agents**:
- `managers_agent`: Oversees the entire unit test generation process
- `generate_unit_test_agent`: Analyzes service classes and generates comprehensive unit tests
- `write_unit_test_agent`: Writes generated tests to the file system (with human-in-the-loop)
- `build_project_agent`: Builds the project to validate generated tests

**Tasks**:
- `generate_unit_test_task`: Generate unit tests based on service class analysis
- `write_unit_test_task`: Write test files to specified test directory
- `build_project_task`: Build and validate the project with new tests

**Tools**:
- `DirectoryListTool`: Navigate project structure
- `FileReadTool`: Read source files for analysis
- `FileWriterTool`: Write test files
- `ProjectBuildTool`: Execute builds and capture errors

**Output**:
- `GenerateUnitTestState` with:
  - `generated_code`: The unit test code
  - `build_result`: Build output/errors
  - `build_success`: Boolean indicating build status

**Iterative Refinement**: This crew receives build feedback and previously generated code, allowing it to fix compilation errors and improve tests across iterations (max 3 attempts).

---

### 3. ServiceClassesListCrew (Optional - Currently Commented Out)

**Purpose**: Lists all service classes in a package for batch processing.

**Process**: Sequential

**Agents**:
- `service_classes_lister_agent`: Scans directories to find service classes

**Tasks**:
- `service_classes_list_task`: List all service classes in the specified package

**Tools**:
- `DirectoryListTool`: Scan directories for Java service classes

**Output**: `ServiceClassesList` containing array of service class names

---

### 4. GitCheckoutAndCommitCrew

**Purpose**: Creates a new branch and commits generated tests.

**Process**: Sequential

**Agents**:
- `git_checkout_agent`: Creates and checks out new branch
- `git_commit_agent`: Stages and commits changes

**Tasks**:
- `git_checkout_task`: Create new branch with specified name
- `git_commit_task`: Add and commit generated test files

**Tools**:
- `GitCheckoutTool`: Branch creation and checkout operations
- `GitTool`: Git add, commit, and status operations

**Output**: Committed unit tests on a new branch ready for pull request

---

## Flow Execution

The `UnitTestGeneratorFlow` in `main.py` orchestrates the crews:

1. **Start** (`generate_unit_test_flow_start`)
   - Entry point of the flow

2. **Git Clone** (`git_clone`)
   - Triggers `GitCloneCrew` to clone and build the project
   - Receives: `git_url`, `git_branch`, `clone_path`, `build_tool`

3. **Generate Unit Test** (`generate_unit_test_for_the_given_class`)
   - Uses `@router()` decorator to handle different outcomes
   - Triggers `GenerateUnitTestCrew` with service class details
   - Receives build feedback from previous iteration (if any)
   - Increments iteration counter
   - Returns: "success", "failed", or "end" (if max iterations reached)
   - On "failed": Routes back to itself for retry with build feedback
   - On "success": Proceeds to git checkout

4. **Git Checkout and Commit** (`git_checkout`)
   - Triggers `GitCheckoutAndCommitCrew` to commit tests
   - Creates new branch and commits generated tests
   - Returns: "end"

5. **End** (`end_method`)
   - Final cleanup and summary

### State Management

The flow maintains state using the `State` Pydantic model:

```python
class State(BaseModel):
    git_url: str                      # Repository URL
    git_branch: str                   # Target branch
    clone_path: str                   # Local clone directory
    service_package_name: str         # Package containing service classes
    test_write_path: str              # Where to write tests
    test_package_name: str            # Test package name
    coding_language: str              # Programming language (Java)
    build_tool: str                   # Maven or Gradle
    service_classes_list: list[str]   # List of service classes
    service_class_name: str           # Current service class
    generated_test_code: str          # Generated test code
    build_error_log: str              # Build errors for feedback
    build_status: str                 # success/failed/end
    iteration: int                    # Current iteration count
    max_iterations: int               # Maximum retry attempts
    new_branch_name: str              # Branch for commits
```

## Installation

```bash
cd Unit-test-generator

# Install dependencies (requires Python 3.10-3.13)
pip install -e .

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"
```

## Usage

### Running the Flow

```bash
# Run the unit test generation flow
kickoff
# or
run_crew
```

### Visualizing the Flow

```bash
# Generate flow diagram
plot
```

### Configuration

Edit the default values in `src/unit_test_generate/main.py`:

```python
class State(BaseModel):
    git_url: str = "https://github.com/SuhasKamate/Business_Management_Project.git"
    git_branch: str = "master"
    clone_path: str = "./Business_Management_Project/"
    service_package_name: str = "com.business.services"
    test_write_path: str = "./Business_Management_Project/src/test/java/com/business/services/"
    test_package_name: str = "com.business.services"
    coding_language: str = "Java"
    build_tool: str = "Maven"
    service_class_name: str = "OrderServices.java"
    new_branch_name: str = "Agent-branch"
    max_iterations: int = 3
```

## Custom Tools

All custom tools are located in `src/unit_test_generate/tools/`:

- **git_clone_tool.py**: Clones repositories with branch support
- **git_checkout_tool.py**: Creates and checks out branches
- **git_tool.py**: Git operations (add, commit, status)
- **file_writer_tool.py**: Writes files to specified paths
- **directory_list_tool.py**: Lists directory contents and filters files
- **project_build_tool.py**: Executes Maven/Gradle builds and captures output

## Features

- **Intelligent Test Generation**: AI analyzes service classes and generates comprehensive unit tests
- **Iterative Refinement**: Automatically fixes compilation errors based on build feedback (up to 3 iterations)
- **Build Validation**: Every generated test is validated through actual project builds
- **Git Integration**: Automatically creates branches and commits working tests
- **Human-in-the-Loop**: Optional human review before writing tests
- **Multi-Tool Support**: Supports both Maven and Gradle build systems
- **Hierarchical Process**: Manager agent coordinates multiple specialized agents

## Limitations

- Currently only supports Java projects
- Requires OpenAI API key (uses GPT-4.1-mini)
- Limited to 3 refinement iterations per test class
- ServiceClassesListCrew is implemented but not currently active in the flow

## Future Enhancements

The commented-out code in `main.py` shows plans for:
- Batch processing of all service classes in a package
- Automatic service class discovery
- Processing multiple classes in a single flow execution