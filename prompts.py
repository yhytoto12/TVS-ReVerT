############################################################
# Special Tokens
############################################################
BOV_TOKEN = "<bov>"
EOV_TOKEN = "<eov>"
CON_TOKEN = "<con>"
DELIMITER = "\n"

############################################################
# System prompts for experiments
############################################################
QWEN_CHAT_TEMPLATE = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {%- if message.role == \"assistant\" %}\n            {{- '<|im_start|>' + message.role + '\\n'}}{% generation %}{{ message.content + '<|im_end|>'}}{% endgeneration %}{{ '\\n' }}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n        {%- endif %}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
LLAMA_CHAT_TEMPLATE = ""

SYSTEM_PROMPT_FOR_VERBALIZER = (
    "Your task is to provide step-by-step reasoning with incremental, speech-friendly summaries.\n"
    "You will be given a question and must reason through it step by step.\n"
    f"Each time you generate the token {BOV_TOKEN}, provide a clear and concise summary of the reasoning so far, suitable for spoken delivery.\n"
    "Do not include equations, LaTeX, or complex symbols in these summaries.\n"
    f"Conclude each summary with the token {EOV_TOKEN}.\n"
    "Ensure each summary connects naturally to the one before it."
)

SYSTEM_PROMPT_FOR_THINK = "Provide a step-by-step reasoning process before arriving at the final answer."

SYSTEM_PROMPT_FOR_SPEECH_FRIENDLY_RESPONSE = (
    "You are a voice assistant that responds in a way that is easy to understand when spoken aloud. "
    "Your responses should be concise, clear, and listener-friendly. "
    "Avoid using equations, LaTeX, or complex symbols that are hard to pronounce or understand in speech."
)

SYSTEM_PROMPTS = {
    "STP": "You are a helpful assistant.",
    "CoT": "You are a helpful assistant that provides a step-by-step reasoning process before arriving at the final answer.",
    "SFP": SYSTEM_PROMPT_FOR_SPEECH_FRIENDLY_RESPONSE,
}