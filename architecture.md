This is a computer use agent project. We will use python and interactive cli to finish the input task. The project's architecture is as follow:

# Setting

- constants.json: Store all the constant mentioned in this document.
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
  - Add a folder named "eye" to store every screenshot and its description sent to the Brain. Use timestamp to name files.
  - Add a csv file called "hand.csv" to record every action made by the Hand. Add the image name to each action that it reacts to and the timestamp that the action is executed.
  - Add a text file named "brain.txt" to store the long term memory. This context will be kept under {constant default to 16k} length. When the file exceed the threshold, ask {model in constant.json} to summarize/shorten the file.
  - Add a folder called "thinking" to store the retrieved image name, thinking process and the final decision/result for each screenshot sent to the Brain.
  - Add a folder named "storage" to store data information that the user want to store in the process.
  - Add a storage.json to store the summary of each file in the storage folder and the timestamp that it is stored.
  - Add a .log file to log all the details in the process, this is only for the debug purpose.

# Eye

- Take a screenshot for every {constant defualt to 2} seconds throughout the whole task until it is done.
- Compare the screenshot to the previous one and calculate a similarity value.
- if the new screenshot is similar to the previous one (set a threshold for this), then do nothing.
- If the new screenshot is different from the previous one, then start the following processing:
  - Use Gemma 4 e2B to describe the details of the current screenshot. For example the current window program name, some text on the screenshot...
- Send the description to the Brain to determine the action no matter what status the Brain is.
- The very first screenshot will always be processed and sent to the Brain in order to initiate the whole task.
- If both the Brain and Hand are idle status, process the new screenshot and send its description to the Brain to make sure we are on the task.

# Brain:

- Whenever the brain retrieve a new screenshot description from Eye, abandon the current processing subtask if there is any and mark the result of the thinking as being interupted by new screenshot image.
- Based on the input description of the image, the Brain will decide the new action by adding the mcp tools and let the {constant model} to decide the next move.
- There are two kinds of tools:
  - Interact with the computer: open the cmd window, click a target, type something, moving mouse... 
    - This type of tools are highly possible to  cause the screen change.
    - When the llm choose this type of tools, we will send a command to the Hand and end the thinking process.
  - Retrieve information: run yolo and ocr, run icon captioner, get the running programs list... 
    - This type of tool will never make the computer screen change.
    - When the llm choose this type of tools, we will retrieve the data and send them to the llm again.

# Hand:

- Retrieve the command from the Brain and execute it. 
- All the possible action should be defined in the mcp folder.

