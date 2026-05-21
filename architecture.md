This is a computer use agent project. We will use python and interactive cli to finish the input task. The project's architecture is as follow:

# Setting

- Agent settings: defaults in `src/common/settings.py`; user overrides in `runs/agent_settings.json` (editable via hub settings dialog). Includes `brain_lm` (default Gemma 4 e4b), `llm_backend`, hosts, and `debug`.
- "mcp" folder: This will store a python file with mcp tools which will be initiated in the begining, and contain folders for each tool to store their logics. We will gradually add tools afterward.
- prompts.json: Store all the prompt in this file to better maintain the project. Each prompt will have a model list meaning that this prompt can be used for these models.

```
{
  "summarize_context":[
    {
      "prompt":"...",
      "models":["model_0", "model_1"]
    },
    {
      "prompt":"...",
      "models":["model_0", "model_1"]
    }
  ],
  "describe_screenshot":[
    {
      "prompt":"...",
      "models":["model_0", "model_1"]
    }
  ]
}
```

# Initiation

- The whole thing will start with an input task in text. 
- Use gemma4 e2b to generate a folder name and then add a new folder under the runs folder, everything about this task will be stored in this folder.
  - Add a folder named "eye" to store the first and every different screenshot and their descriptions sent to the Brain. Use timestamp to name files.
  - Add a csv file called "hand.csv" to record every action made by the Hand. Add the image name to each action that it reacts to and the timestamp that the action is executed.
  - Add a text file named "long_term_memory.txt" to store the long term memory. This context will be kept under {constant default to 16k} length. When the file exceed the threshold, ask {model in constant.json} to summarize/shorten the file.
  - Add a folder called "thinking" to store the retrieved image name, thinking process and the final decision/result for each screenshot sent to the Brain.
  - Add a folder named "storage" to store data information that the user want to store in the process.
  - Add a storage.json to store the summary of each file in the storage folder and the timestamp that it is stored.
  - Add a .log file to log all the details in the process, this is only for the debug purpose.

# Main modules

### Eye Module:

- Take a screenshot for every {constant defualt to 2} seconds throughout the whole task until it is done.
- If the Hand server has ongoing execution, we don't need to take any screenshot because the screen is expected to be different when the Hand is doing something.
- Compare the screenshot to the previous one and calculate a similarity value.
- if the new screenshot is similar to the previous one (set a threshold for this), then do nothing.
- If the new screenshot is different from the previous one, then start the following processing:
  - Ask the {model in the constant.json} to make sure the new screenshot is really different from the previous one that require the Brain server to think about the next move.
  - Use {model in the constant.json} to describe the details of the current screenshot. For example the current window program name, some text on the screenshot...
- Return the newest capture event to the coordinator for Brain processing.
- The very first screenshot will always be processed and sent to the Brain in order to initiate the whole task.

### Brain Module:

- If the brain retrieves a new screenshot description from Eye when it is still thinking. Ask the {model in the constant} if the new screenshot is an interruption or not. 
  - If it is an interruption then put the current processing subtask in a stack.
  - If it is a new state that replace the previous one, then stop the previous thinking and start the new thinking on the new arrived screenshot.
- If the brain retrieves a new screenshot in idle state, check if the screenshot is similar to any of the screenshot image in the stack.
  - If there is a screenshot in the stack highly similar to the new screenshot, then take out this unfinished thinking process, and proceed with it.
  - If there is no screenshot found matching in the stack, then start a new one.
- Based on the input description of the image, the Brain decides the next action with {model in the constant.json} provided with MCP tools and the previous action from the Hand module if it exists.
- There are two kinds of MCP tools:
  - Interact with the computer: open the cmd window, click a target, type something, moving mouse... 
    - This type of tools are highly possible to  cause the screen change.
    - When the llm choose this type of tools, we will send a command to the Hand and end the thinking process.
  - Retrieve information: run yolo and ocr, run icon captioner, get the running programs list... 
    - This type of tool will never make the computer screen change.
    - When the llm choose this type of tools, we will retrieve the data and send them to the llm again.

### Hand Module:

- Retrieve the command from the Brain module and execute it.
- All the possible action should be defined in the mcp folder.
- Return the done action to the coordinator, which forwards it to Brain as a "previous action".

# Technical Implementation: Single-Coordinator Orchestration

To ensure the **Eye**, **Brain**, and **Hand** can operate asynchronously and support the "Interrupt" logic (abandoning subtasks when the screen changes), the project uses a **single in-process coordinator** that calls module APIs directly.

### 1. Master Process (Coordinator Runtime)

A central `main.py` script serves as the lifecycle manager. It initializes runtime state, then runs one coordinator loop that orchestrates eye capture, brain decisions, and hand execution.

- **Resilience:** Runtime failures are captured in run logs; orchestration remains centralized and debuggable.
- **Lifecycle:** On `KeyboardInterrupt`, the runtime exits cleanly and preserves run artifacts under `runs/`.

### 2. Communication Layer (Direct Module Calls)

The coordinator invokes module methods directly in-process instead of cross-process HTTP callbacks.


| Module | Primary Responsibility |
| ------ | ---------------------- |
| Eye | Capture screen events |
| Brain | Plan and select tool calls |
| Hand | Execute tool calls |


### 3. Asynchronous Interrupt Flow

This architecture is specifically designed to handle the **Interrupt Logic** required for dynamic UI elements like warning dialogues:

1. **Eye** captures a new screenshot event and returns it to coordinator.
2. **Brain** evaluates the event and computes one or more tool commands.
3. **Hand** executes commands, then coordinator submits action results back to Brain for the next cycle.

### 4. Data Consistency

All modules maintain access to the shared `runs` directory.

- **Eye** writes to `eye/`.
- **Brain** reads from `eye/`, writes to `long_term_memory.txt`, `thinking/`, and `storage.json`.
- **Hand** reads commands from the Brain and appends results to `hand.csv`.

