This is a computer use agent project. We will use python and interactive cli to finish the input task. The project's architecture is as follow:

# Setting

- constants.json: Store all the constant mentioned in this document.
  - eye_vlm: default to Gemma 4 e2B
  - brain_lm: default to Gemma 4 e2B
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
  - Add a text file named "brain.txt" to store the long term memory. This context will be kept under {constant default to 16k} length. When the file exceed the threshold, ask {model in constant.json} to summarize/shorten the file.
  - Add a folder called "thinking" to store the retrieved image name, thinking process and the final decision/result for each screenshot sent to the Brain.
  - Add a folder named "storage" to store data information that the user want to store in the process.
  - Add a storage.json to store the summary of each file in the storage folder and the timestamp that it is stored.
  - Add a .log file to log all the details in the process, this is only for the debug purpose.

# Three main servers

### Eye Server:

- Take a screenshot for every {constant defualt to 2} seconds throughout the whole task until it is done.
- If the Hand server has ongoing execution, we don't need to take any screenshot because the screen is expected to be different when the Hand is doing something.
- Compare the screenshot to the previous one and calculate a similarity value.
- if the new screenshot is similar to the previous one (set a threshold for this), then do nothing.
- If the new screenshot is different from the previous one, then start the following processing:
  - Ask the {model in the constant.json} to make sure the new screenshot is really different from the previous one that require the Brain server to think about the next move.
  - Use {model in the constant.json} to describe the details of the current screenshot. For example the current window program name, some text on the screenshot...
- Send the description to the Brain to determine the next action no matter what status the Brain is.
- The very first screenshot will always be processed and sent to the Brain in order to initiate the whole task.

### Brain Server:

- If the brain retrieves a new screenshot description from Eye when it is still thinking. Ask the {model in the constant} if the new screenshot is an interruption or not. 
  - If it is an interruption then put the current processing subtask in a stack.
  - If it is a new state that replace the previous one, then stop the previous thinking and start the new thinking on the new arrived screenshot.
- If the brain retrieves a new screenshot in idle state, check if the screenshot is similar to any of the screenshot image in the stack.
  - If there is a screenshot in the stack highly similar to the new screenshot, then take out this unfinished thinking process, and proceed with it.
  - If there is no screenshot found matching in the stack, then start a new one.
- Based on the input description of the image, the Brain will decide the new action with {model in the constant.json} provided with MCP tools and the "previous action" from the "Hand Server" if it exists.
- There are two kinds of MCP tools:
  - Interact with the computer: open the cmd window, click a target, type something, moving mouse... 
    - This type of tools are highly possible to  cause the screen change.
    - When the llm choose this type of tools, we will send a command to the Hand and end the thinking process.
  - Retrieve information: run yolo and ocr, run icon captioner, get the running programs list... 
    - This type of tool will never make the computer screen change.
    - When the llm choose this type of tools, we will retrieve the data and send them to the llm again.

### Hand Server:

- Retrieve the command from the Brain and execute it. 
- All the possible action should be defined in the mcp folder.
- Send the done action to the Brain server, the Brain server will receive it as a "previous action".

# Technical Implementation: Multi-Server Orchestration

To ensure the **Eye**, **Brain**, and **Hand** can operate asynchronously and support the "Interrupt" logic (abandoning subtasks when the screen changes), the project utilizes a **Master-Subprocess Manager** pattern combined with **FastAPI** for inter-process communication (IPC).

### 1. Master Process (Subprocess Manager)

A central `main.py` script serves as the lifecycle manager. It uses Python’s `subprocess` module to launch and monitor the three servers.

- **Resilience:** If any server (e.g., the Hand) crashes due to a driver error, the Master process can log the event and restart the specific service without losing the entire task context.
- **Lifecycle:** On `KeyboardInterrupt`, the Master sends `SIGTERM` to all children to ensure clean-up of screenshots and temporary logs in the `runs` folder.

### 2. Communication Layer (HTTP/FastAPI)

Each server hosts a lightweight FastAPI instance on a dedicated local port. This enables low-latency, structured data exchange.


|            |          |                                                                                                         |
| ---------- | -------- | ------------------------------------------------------------------------------------------------------- |
| **Server** | **Port** | **Primary Responsibility**                                                                              |
| **Eye**    | `8001`   | Monitors screen similarity; sends `POST /interrupt` to the Brain when a change is detected.             |
| **Brain**  | `8002`   | Processes descriptions; receives interrupts to cancel current LLM inference.                            |
| **Hand**   | `8003`   | Receives JSON-encoded commands (e.g., `{ "action": "click", "coords": [100, 200] }`) and executes them. |


### 3. Asynchronous Interrupt Flow

This architecture is specifically designed to handle the **Interrupt Logic** required for dynamic UI elements like warning dialogues:

1. **The Eye** detects a new dialogue box and sends a request to `localhost:8002/new_event`.
2. **The Brain's** FastAPI endpoint run the llm to check if it is truly an interruption and then sets an internal `interrupt_flag` if it is an interruption.
3. The **Brain's** logic loop checks this flag; if `True`, it put the current thinking process in a stack, logs the interruption in `brain.txt`, and starts a new inference based on the latest screenshot.

### 4. Data Consistency

All processes maintain access to the shared `runs` directory.

- **Eye** writes to `eye/` and `runs/brain.txt`.
- **Brain** reads from `eye/` and writes to `thinking/` and `storage.json`.
- **Hand** reads commands from the Brain and appends results to `hand.csv`.

