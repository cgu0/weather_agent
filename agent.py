import json
import os
from typing import Optional
from unittest.mock import Base
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import requests

load_dotenv()

client = OpenAI()


class MyOutputFormat(BaseModel):
    step: str = Field(
        ..., description="The id of the step. Example: PLAN, OUTPUT, TOOL, etc."
    )
    content: Optional[str] = Field(
        None, description="The optional string content for the step"
    )
    tool: Optional[str] = Field(None, description="The id of tool to call")
    input: Optional[str] = Field(None, description="The input params for the tool")


def run_command(cmd: str):
    result = os.system(cmd)
    return result


def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"
    return "Something is wrong."


available_tools = {"get_weather": get_weather, "run_command": run_command}

print("\n\n\n")

SYSTEM_PROMPT = """
    You're an expert AI Assistant in resolving user queries using chain of thoughts
    You work on START, PLAN abd OUTPUT steps.
    You need to fitst PLAN what needs to be done. The PLAN can be mutiple steps
    Once you think enough PLAN has been done, finally you can give an OUTPUT
    You can call a tool if required from the list of available tools

    Rules:
    - Strictly follow the given JSON output format
    - Only run one step at a time
    - The sequence of steps is START (where user gives an input), PLAN (That can be mutiple steps) and OUTPUT (where you need to give the user an answer which meet their needs and can be as accurate as you could).

    Output JSON Format:
    {"step": "START" | "PLAN" | "OUTPUT" | "TOOL" | "OBSERVE", "content": "string"}

    Available Tools:
    - get_weather(city:str): Takes city as input and returns the weather info about a city
    - run_command(cmd:str): Takes a system Linux command as string and excute the command on users' system and return the output of the command

    Example1:
    START: Hey, Can you solve 2 + 2 * 3 / 10
    PLAN: {"step": "PLAN", "content": "Seems like user is interested in math problem"}
    PLAN: {"step": "PLAN", "content": "Looking at the promblem, we should solve this problem by using BOSMAS method"}
    PLAN: {"step": "PLAN", "content": "Yes, The BODMAS is correct thing to be done here"}
    PLAN: {"step": "PLAN", "content": "first we should mutiple 2 and 3 and get 6 here"}
    PLAN: {"step": "PLAN", "content": "Now the new equatuon is 2 + 6 / 10"}
    PLAN: {"step": "PLAN", "content": "Then we must perform divide that is 6 / 10 which equals to 0.6"}
    PLAN: {"step": "PLAN", "content": "Now the new equatuon is 2 + 0.6"}
    PLAN: {"step": "PLAN", "content": "Finally lets perform the add 2.6"}
    PLAN: {"step": "PLAN", "content": "Great, we habe solved and finally get the result of 2.6"}
    OUTPUT: {"step": "OUTPUT", "content": "2.6" }

    Example2:
    START: Hey, What is the weather like in Shanghai?
    PLAN: {"step": "PLAN", "content": "Seems like user is interested in the weather like in Shanghai"}
    PLAN: {"step": "PLAN", "content": "Lets see if we have any available tool from the list of availble tools"}
    PLAN: {"step": "PLAN", "content": "Great, we have the get_weather tool to look up the weather for a city"}
    PLAN: {"step": "PLAN", "content": "I need to call the tool to check what is the weather in Shanghai"}
    PLAN: {"step": "TOOL", "tool": "get_weather", "input": "Shanghai"}
    PLAN: {"step": "OBSERVE", "tool": "get_weather", "output": "The weather in Shanghai is  rainy with 8 degrees"}
    PLAN: {"step": "PLAN", "content": "Great, I got the weather info in Shanghai"}
    OUTPUT: {"step": "OUTPUT", "content": "The current weather in Shanghai is 8 degree with some rains" }
"""


def run_agent():
    print("\n\n\n")

    while True:
        user_query = input("🫵（输入 q 退出） ")
        if user_query.strip().lower() in {"q", "quit", "exit"}:
            break

        message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        message_history.append({"role": "user", "content": user_query})

        while True:
            response = client.chat.completions.parse(
                model="gpt-4o",
                response_format=MyOutputFormat,
                messages=message_history,
            )

            raw_result = response.choices[0].message.content
            message_history.append({"role": "assistant", "content": raw_result})

            parsed_result = response.choices[0].message.parsed

            if parsed_result.step == "START":
                print("🔥", parsed_result.content)
                continue

            if parsed_result.step == "TOOL":
                tool_to_call = parsed_result.tool
                tool_input = parsed_result.input
                print(f"🛠️: {tool_to_call} ({tool_input})")

                total_response = available_tools[tool_to_call](tool_input)
                print(f"🛠️: {tool_to_call} ({tool_input}) = {total_response}")
                message_history.append(
                    {
                        "role": "developer",
                        "content": json.dumps(
                            {
                                "step": "OBSERVE",
                                "tool": tool_to_call,
                                "input": tool_input,
                                "output": total_response,
                            }
                        ),
                    }
                )
                continue

            if parsed_result.step == "PLAN":
                print("🧠", parsed_result.content)
                continue

            if parsed_result.step == "OUTPUT":
                print("🤖", parsed_result.content)
                break

        print()

    print("\n\n\n")


if __name__ == "__main__":
    run_agent()
