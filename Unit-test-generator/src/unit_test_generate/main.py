#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router

from unit_test_generate.crews.poem_crew.poem_crew import PoemCrew
from unit_test_generate.crews.git_clone_crew.git_clone_crew import GitCloneCrew
from unit_test_generate.crews.service_classes_list_crew.service_classes_list_crew import ServiceClassesListCrew
from unit_test_generate.crews.generate_unit_test_crew.generate_unit_test_crew import GenerateUnitTestCrew
from unit_test_generate.crews.git_checkout_and_commit_crew.git_checkout_and_commit_crew import GitCheckoutAndCommitCrew

class State(BaseModel):
    git_url: str = "https://github.com/SuhasKamate/Business_Management_Project.git"
    git_branch: str = "master"
    clone_path: str = "./Business_Management_Project/"
    service_package_name: str = "com.business.services"
    test_write_path: str = "./Business_Management_Project/src/test/java/com/business/services/"
    test_package_name: str = "com.business.services"
    coding_language: str = "Java"
    build_tool: str = "Maven"
    service_classes_list: list[str] = []
    service_class_name: str = "OrderServices.java"
    generated_test_code: str = ""
    build_error_log: str = ""
    build_status: str = ""
    iteration: int = 0
    max_iterations: int = 3
    new_branch_name: str = "Agent-branch"

class UnitTestGeneratorFlow(Flow[State]):

    @start()
    def generate_unit_test_flow_start(self):
        print("Generating Unit Tests")

    @listen(generate_unit_test_flow_start)
    def git_clone(self):
        print("Cloning repo and building project")
        result = (
            GitCloneCrew()
            .crew()
            .kickoff(
                inputs={
                    "git_url": self.state.git_url,
                    "git_branch": self.state.git_branch,
                    "clone_path": self.state.clone_path,
                    "build_tool": self.state.build_tool,
                }
            )
        )

        print("Git Clone Result:", result)

    # @listen(git_clone)
    # def service_classes_list(self):
    #     print("Listing service classes")
    #     result = (
    #         ServiceClassesListCrew()
    #         .crew()
    #         .kickoff(
    #             inputs={
    #                 "clone_path": self.state.clone_path,
    #                 "service_package_name": self.state.service_package_name,
    #                 "coding_language": self.state.coding_language,
    #                 "build_tool": self.state.build_tool,
    #             }
    #         )
    #     )
    #     self.state.service_classes_list = result.pydantic.service_classes
    #     print("Service Classes List Result:", result.raw)

    # need to return the generated code and the build error

    # @listen(service_classes_list)
    # def generate_unit_test_for_all_classes(self):
    #     print("Generating unit tests for all service classes")


    #     for service_class in self.state.service_classes_list:
    #         print(f"Generating unit tests for {service_class}")
    #         result = (
    #             GenerateUnitTestCrew()
    #             .crew()
    #             .kickoff(
    #                 inputs={
    #                     "clone_path": self.state.clone_path,
    #                     "service_class_name": service_class,
    #                     "test_write_path": self.state.test_write_path,
    #                     "test_package_name": self.state.test_package_name,
    #                     "coding_language": self.state.coding_language,
    #                     "build_tool": self.state.build_tool,
    #                     "service_package_name": self.state.service_package_name,
    #                 }
    #             )
    #         )
    #         print(f"Generated Unit Test for {service_class} and result", result.raw)


    @router(git_clone)
    @listen("failed")
    def generate_unit_test_for_the_given_class(self):
        print("Generating unit tests for all service classes")
        self.state.iteration += 1
        result = (
            GenerateUnitTestCrew()
            .crew()
            .kickoff(
                inputs={
                    "clone_path": self.state.clone_path,
                    "service_class_name": self.state.service_class_name,
                    "test_write_path": self.state.test_write_path,
                    "test_package_name": self.state.test_package_name,
                    "coding_language": self.state.coding_language,
                    "build_tool": self.state.build_tool,
                    "service_package_name": self.state.service_package_name,
                    "build_feedback": self.state.build_error_log,
                    "generated_test_code": self.state.generated_test_code,
                }
            )
        )

        print("Generated Unit Test Result:", result.raw)
        print("Build Result:", result.pydantic)

        self.state.generated_test_code = result.pydantic.generated_code
        self.state.build_error_log = result.pydantic.build_result
        self.state.build_status = "success" if result.pydantic.build_success else "failed"
        if self.state.iteration > self.state.max_iterations:
            print("Max iterations reached. Ending flow.")
            self.end()
            return "end"
        print(f"Build Status for debugging: {self.state.build_status}")
        return self.state.build_status

    @listen("success")
    def git_checkout(self):
        result = (
            GitCheckoutAndCommitCrew()
            .crew()
            .kickoff(
                inputs={
                    "clone_path": self.state.clone_path,
                    "git_url": self.state.git_url,
                    "new_branch_name": self.state.new_branch_name,
                    "service_class_name": self.state.service_class_name,
                    "generated_test_code": self.state.generated_test_code,
                    "test_write_path": self.state.test_write_path,
                    "commit_message": f"Added unit tests for {self.state.service_class_name}",
                }
            )
        )
        print("Git Checkout and Commit Result:", result.raw)
        return "end"
    
    @listen("end")
    def end_method(self):
        print(f"Ending the flow after {self.state.iteration} iterations.")

def kickoff():
    unit_test_flow = UnitTestGeneratorFlow()
    unit_test_flow.kickoff()


def plot():
    unit_test_flow = UnitTestGeneratorFlow()
    unit_test_flow.plot()


if __name__ == "__main__":
    kickoff()
