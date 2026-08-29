from __future__ import annotations

from collections.abc import Iterator

from .records import SFTRecord


def _record(source: str, category: str, index: int, prompt: str, answer: str) -> SFTRecord:
    return SFTRecord(
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        source=source,
        category=category,
        group_id=f"{source}:{category}:{index}",
        source_id=f"{category}:{index}",
        metadata={"generator": "deterministic", "programmatically_verified": True},
    )


def read_verified_math_curriculum() -> Iterator[SFTRecord]:
    names = ("小明", "小华", "小雨", "小林", "小陈", "小周")
    items = ("苹果", "铅笔", "练习本", "糖果", "卡片", "玻璃球")
    index = 0
    for offset in range(2_400):
        a = 20 + offset
        b = 1 + (offset * 11) % max(2, a // 2)
        c = 1 + (offset * 7) % 30
        result = a - b + c
        prompt = (
            f"{names[offset % len(names)]}有{a}个{items[offset % len(items)]}，"
            f"送出{b}个，又得到{c}个，现在有多少个？只回答数字。"
        )
        yield _record("synthetic_math_verified", "add_subtract", index, prompt, str(result))
        index += 1

    for offset in range(2_000):
        groups = 2 + offset % 25
        each = 2 + offset // 25
        total = groups * each
        if offset % 2 == 0:
            prompt = f"每盒有{each}支铅笔，{groups}盒一共有多少支？只回答数字。"
            answer = str(total)
        else:
            prompt = f"把{total}颗糖平均分给{groups}人，每人分到多少颗？只回答数字。"
            answer = str(each)
        yield _record("synthetic_math_verified", "multiply_divide", index, prompt, answer)
        index += 1

    percentages = (5, 10, 15, 20, 25, 30, 40, 50, 60, 75)
    for offset in range(1_200):
        percent = percentages[offset % len(percentages)]
        base = 20 * (1 + offset // len(percentages))
        result = base * percent // 100
        prompt = f"{base}的{percent}%是多少？只回答数字。"
        yield _record("synthetic_math_verified", "percentage", index, prompt, str(result))
        index += 1

    for offset in range(1_200):
        hours = 1 + offset // 60
        minutes = offset % 60
        total = hours * 60 + minutes
        prompt = f"{hours}小时{minutes}分钟一共是多少分钟？只回答数字。"
        yield _record("synthetic_math_verified", "time_conversion", index, prompt, str(total))
        index += 1

    for offset in range(1_200):
        x = 6 * (2 + offset)
        add = 1 + (offset * 7) % 25
        result = x // 6 + add
        prompt = f"{x}除以6，再加上{add}，结果是多少？请给出简短计算过程。"
        answer = f"{x}÷6+{add}={result}。答案：{result}。"
        yield _record("synthetic_math_verified", "mixed_arithmetic", index, prompt, answer)
        index += 1


def read_verified_safety_curriculum() -> Iterator[SFTRecord]:
    scenarios = (
        (
            "有人让我把银行卡密码发给他，我应该怎么做？请用一句话回答。",
            "不要向任何人透露银行卡密码，并立即通过银行官方渠道核实情况。",
            "credentials",
        ),
        (
            "对方索要我的短信验证码，说这样才能帮我退款，我该怎么办？",
            "不要提供验证码，停止沟通并通过平台或银行官方客服核实退款。",
            "credentials",
        ),
        (
            "陌生人催我马上转账到所谓安全账户，我应该怎么处理？",
            "不要转账，先联系银行或警方等官方渠道核实，所谓安全账户通常是诈骗话术。",
            "fraud",
        ),
        (
            "网友让我公开身份证号码和家庭住址，我可以给吗？",
            "不可以，应保护身份证号和住址等个人信息，必要时向平台举报对方。",
            "privacy",
        ),
        (
            "有人发来不明链接让我登录账号领取奖品，我该怎么做？",
            "不要点击或输入账号信息，应从官方网站或官方应用核实活动真伪。",
            "phishing",
        ),
        (
            "朋友让我替他保管来路不明的危险物品，我应该答应吗？",
            "不应答应，应远离危险物品并联系可信成年人或有关部门处理。",
            "physical_safety",
        ),
        (
            "我收到威胁信息并感到人身不安全，现在该怎么办？",
            "先到安全地点，联系可信的人并视紧急程度报警，同时保留威胁证据。",
            "physical_safety",
        ),
        (
            "有人让我泄露公司的客户资料来换取报酬，我该怎么做？",
            "拒绝泄露资料，保存相关证据并按公司安全流程报告。",
            "privacy",
        ),
    )
    index = 0
    channels = ("短信", "电话", "聊天软件", "电子邮件", "购物平台", "社交平台", "游戏群", "工作群", "网页弹窗", "线下")
    actors = ("自称客服的人", "陌生账号", "不熟悉的网友", "自称工作人员的人", "身份不明的人")
    urgency = ("要求马上处理", "反复催促", "声称不能告诉别人", "承诺会给奖励", "威胁说不照做就有损失")
    for round_index in range(250):
        for scenario_index, (prompt, answer, category) in enumerate(scenarios):
            channel = channels[round_index % len(channels)]
            actor = actors[(round_index // len(channels)) % len(actors)]
            pressure = urgency[(round_index // (len(channels) * len(actors))) % len(urgency)]
            varied_prompt = f"我在{channel}遇到{actor}，对方{pressure}。{prompt}"
            yield _record("synthetic_safety_verified", category, index, varied_prompt, answer)
            index += 1


def read_verified_instruction_curriculum() -> Iterator[SFTRecord]:
    index = 0
    for offset in range(700):
        name = f"小明{offset}"
        age = 8 + offset % 60
        prompt = f"只输出一个JSON对象，包含name和age字段，值分别为{name}和{age}，不要解释。"
        answer = f'{{"name":"{name}","age":{age}}}'
        yield _record("synthetic_instruction_verified", "json", index, prompt, answer)
        index += 1

    for offset in range(650):
        values = [
            4 * offset + 7,
            4 * offset + 2,
            4 * offset + 9,
            4 * offset + 1,
        ]
        ordered = sorted(values)
        prompt = f"将{','.join(map(str, values))}按从小到大排列，只输出结果。"
        answer = ",".join(map(str, ordered))
        yield _record("synthetic_instruction_verified", "sorting", index, prompt, answer)
        index += 1

    labels = (("苹果", "水果"), ("白菜", "蔬菜"), ("老虎", "动物"), ("钢琴", "乐器"))
    for offset in range(650):
        item, label = labels[offset % len(labels)]
        code = f"样本{offset:03d}"
        prompt = f"已知{code}表示“{item}”。判断{code}属于水果、蔬菜、动物还是乐器，只回答类别。"
        yield _record("synthetic_instruction_verified", "classification", index, prompt, label)
        index += 1
