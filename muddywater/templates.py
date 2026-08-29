from __future__ import annotations


DEFAULT_SYSTEM_PROMPT = "你是一个有帮助、严谨的中文助手。"
SUPPORTED_MESSAGE_ROLES = {"system", "user", "assistant"}
TEMPLATE_ERROR = "chat_template must be one of: chatml, alpaca, chinese_alpaca, legacy, custom"


def _normalize_template_name(name: str | None) -> str:
    return (name or "chatml").strip().lower().replace("-", "_")


def _build_user_content(instruction: str, input_text: str = "") -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def _message_role_and_content(msg: dict[str, str]) -> tuple[str, str]:
    role = msg.get("role", "user").strip().lower()
    if role not in SUPPORTED_MESSAGE_ROLES:
        raise ValueError(f"Unsupported role in messages: {role}")
    return role, msg.get("content", "").strip()


def _append_masked(chars: list[tuple[str, bool]], text: str, trainable: bool) -> None:
    chars.extend((char, trainable) for char in text)


def format_sft_example(
    instruction: str,
    output: str,
    input_text: str = "",
    chat_template: str = "chatml",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    instruction_template: str | None = None,
    instruction_template_no_input: str | None = None,
) -> str:
    """Format one instruction/input/output record into one LM training string."""
    template_name = _normalize_template_name(chat_template)
    instruction = instruction.strip()
    input_text = input_text.strip()
    output = output.strip()
    system_prompt = system_prompt.strip()
    user_content = _build_user_content(instruction, input_text)

    if template_name == "chatml":
        parts: list[str] = []
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
        parts.append(f"<|im_start|>user\n{user_content}<|im_end|>")
        parts.append(f"<|im_start|>assistant\n{output}<|im_end|>")
        return "\n".join(parts)

    if template_name == "alpaca":
        if input_text:
            return (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    if template_name == "chinese_alpaca":
        if input_text:
            return f"### 指令：\n{instruction}\n\n### 输入：\n{input_text}\n\n### 回答：\n{output}"
        return f"### 指令：\n{instruction}\n\n### 回答：\n{output}"

    if template_name in {"legacy", "custom"}:
        template = instruction_template if input_text else instruction_template_no_input
        if not template:
            template = (
                "用户：{instruction}\n输入：{input}\n助手：{output}"
                if input_text
                else "用户：{instruction}\n助手：{output}"
            )
        return template.format(instruction=instruction, input=input_text, output=output)

    raise ValueError(TEMPLATE_ERROR)


def format_sft_example_with_labels(
    instruction: str,
    output: str,
    input_text: str = "",
    chat_template: str = "chatml",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> tuple[str, str]:
    """Format an instruction/input/output record and mask assistant tokens only."""
    user_content = _build_user_content(instruction, input_text)
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output.strip()},
    ]
    return format_messages_with_labels(
        messages=messages,
        chat_template=chat_template,
        system_prompt=system_prompt,
    )


def format_messages_with_labels(
    messages: list[dict[str, str]],
    chat_template: str = "chatml",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> tuple[str, str]:
    """Format messages into a ChatML string and a label-mask string."""
    template_name = _normalize_template_name(chat_template)
    chars: list[tuple[str, bool]] = []

    if template_name == "chatml":
        need_system = messages and messages[0].get("role") != "system" and system_prompt
        any_output = False

        if need_system:
            line = f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>"
            _append_masked(chars, line, False)
            any_output = True

        for index, msg in enumerate(messages):
            if index > 0 or any_output:
                _append_masked(chars, "\n", False)
            role, content = _message_role_and_content(msg)
            is_assistant = role == "assistant"
            _append_masked(chars, f"<|im_start|>{role}\n", False)
            _append_masked(chars, content, is_assistant)
            _append_masked(chars, "<|im_end|>", is_assistant)

    elif template_name in {"alpaca", "chinese_alpaca"}:
        headers = (
            {"system": "### Instruction:\n", "user": "### Input:\n", "assistant": "### Response:\n"}
            if template_name == "alpaca"
            else {"system": "### 指令：\n", "user": "### 输入：\n", "assistant": "### 回答：\n"}
        )
        for index, msg in enumerate(messages):
            if index > 0:
                _append_masked(chars, "\n", False)
            role, content = _message_role_and_content(msg)
            is_assistant = role == "assistant"
            _append_masked(chars, headers[role], False)
            _append_masked(chars, content, is_assistant)
            _append_masked(chars, "\n", is_assistant)

    elif template_name in {"legacy", "custom"}:
        prefixes = {"system": "系统：", "user": "用户：", "assistant": "助手："}
        for index, msg in enumerate(messages):
            if index > 0:
                previous_role, _ = _message_role_and_content(messages[index - 1])
                _append_masked(chars, "\n", previous_role == "assistant")
            role, content = _message_role_and_content(msg)
            is_assistant = role == "assistant"
            _append_masked(chars, prefixes[role], False)
            _append_masked(chars, content, is_assistant)

    else:
        raise ValueError(TEMPLATE_ERROR)

    text = "".join(char for char, _ in chars)
    mask = "".join("1" if trainable else "0" for _, trainable in chars)
    return text, mask


def format_messages(
    messages: list[dict[str, str]],
    chat_template: str = "chatml",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Format OpenAI-style messages into one LM training string."""
    template_name = _normalize_template_name(chat_template)

    if template_name == "chatml":
        parts: list[str] = []
        if messages and messages[0].get("role") != "system" and system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>")
        for msg in messages:
            role, content = _message_role_and_content(msg)
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(parts)

    if template_name == "alpaca":
        headers = {"system": "### Instruction:", "user": "### Input:", "assistant": "### Response:"}
    elif template_name == "chinese_alpaca":
        headers = {"system": "### 指令：", "user": "### 输入：", "assistant": "### 回答："}
    elif template_name in {"legacy", "custom"}:
        headers = {"system": "系统：", "user": "用户：", "assistant": "助手："}
    else:
        raise ValueError(TEMPLATE_ERROR)

    lines: list[str] = []
    for msg in messages:
        role, content = _message_role_and_content(msg)
        lines.append(f"{headers[role]}\n{content}")
    return "\n".join(lines)


def format_generation_prompt(
    instruction: str,
    input_text: str = "",
    chat_template: str = "chatml",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Format a user prompt into the same prompt prefix used during SFT."""
    template_name = _normalize_template_name(chat_template)
    instruction = instruction.strip()
    input_text = input_text.strip()
    system_prompt = system_prompt.strip()
    user_content = _build_user_content(instruction, input_text)

    if template_name == "chatml":
        parts: list[str] = []
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
        parts.append(f"<|im_start|>user\n{user_content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    if template_name == "alpaca":
        if input_text:
            return (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                "### Response:\n"
            )
        return f"### Instruction:\n{instruction}\n\n### Response:\n"

    if template_name == "chinese_alpaca":
        if input_text:
            return f"### 指令：\n{instruction}\n\n### 输入：\n{input_text}\n\n### 回答：\n"
        return f"### 指令：\n{instruction}\n\n### 回答：\n"

    if template_name in {"legacy", "custom"}:
        if input_text:
            return f"用户：{instruction}\n输入：{input_text}\n助手："
        return f"用户：{instruction}\n助手："

    raise ValueError(TEMPLATE_ERROR)
