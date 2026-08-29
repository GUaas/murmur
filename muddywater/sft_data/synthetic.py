from __future__ import annotations

import json
from collections.abc import Iterator
from itertools import combinations, islice

from .records import SFTRecord


PRODUCTS = ("无线耳机", "机械键盘", "显示器", "移动硬盘", "路由器", "保温杯", "台灯", "双肩包")
CITIES = ("北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京")
STATUSES = ("待付款", "待发货", "运输中", "已签收", "已取消")
INTENTS = {
    "查询物流": ("我的包裹到哪里了", "帮我查一下快递进度", "订单什么时候送到"),
    "取消订单": ("这个订单不要了", "请帮我取消购买", "我想撤销刚才的订单"),
    "申请退款": ("商品有问题我要退款", "怎么申请退钱", "这个产品不合适想退款"),
    "修改地址": ("收货地址写错了", "帮我换一个配送地址", "订单地址需要修改"),
    "产品咨询": ("这款产品支持蓝牙吗", "可以介绍一下这个商品吗", "这个型号有哪些功能"),
    "投诉建议": ("我要反馈服务问题", "客服态度不好怎么投诉", "我有一条改进建议"),
}
LIST_TOPICS = {
    "颜色": ("红色", "蓝色", "绿色", "黄色", "白色", "黑色", "紫色", "橙色", "灰色", "棕色"),
    "水果": ("苹果", "香蕉", "橙子", "葡萄", "草莓", "梨", "桃子", "西瓜", "芒果", "樱桃"),
    "城市": ("北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京", "西安", "苏州"),
    "动物": ("熊猫", "海豚", "长颈鹿", "企鹅", "大象", "老虎", "狮子", "斑马", "河马", "袋鼠"),
}


def _record(index: int, category: str, user: str, assistant: str) -> SFTRecord:
    return SFTRecord(
        messages=[{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        source="synthetic_verified",
        category=category,
        group_id=f"synthetic:{category}:{index}",
        source_id=f"{category}:{index}",
    )


def read_verified_synthetic() -> Iterator[SFTRecord]:
    index = 0
    for order_number in range(1000, 1700):
        product = PRODUCTS[order_number % len(PRODUCTS)]
        city = CITIES[(order_number // 2) % len(CITIES)]
        status = STATUSES[(order_number // 3) % len(STATUSES)]
        quantity = order_number % 5 + 1
        user = (
            "请从文本中提取order_id、product、quantity、city和status，"
            "只输出合法JSON，不要解释。\n"
            f"文本：订单{order_number}购买了{quantity}件{product}，收货城市为{city}，当前状态是{status}。"
        )
        answer = json.dumps(
            {
                "order_id": str(order_number),
                "product": product,
                "quantity": quantity,
                "city": city,
                "status": status,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        json.loads(answer)
        yield _record(index, "json_extraction", user, answer)
        index += 1

    for label, utterances in INTENTS.items():
        for repeat in range(90):
            utterance = utterances[repeat % len(utterances)]
            product = PRODUCTS[(repeat + len(label)) % len(PRODUCTS)]
            ticket_id = f"TK{index:05d}"
            user = (
                "请判断用户意图，只能输出以下一个标签：查询物流、取消订单、申请退款、"
                "修改地址、产品咨询、投诉建议。\n"
                f"工单{ticket_id}的用户消息：{utterance}，相关商品是{product}。"
            )
            assert label in INTENTS
            yield _record(index, "intent_classification", user, label)
            index += 1

    for topic, values in LIST_TOPICS.items():
        candidates = (
            selected
            for count in range(2, 6)
            for selected in combinations(values, count)
        )
        for selected in islice(candidates, 100):
            count = len(selected)
            answer = "，".join(selected)
            assert len(answer.split("，")) == count
            user = (
                f"请按原顺序输出以下{count}个{topic}名称，使用中文逗号分隔，"
                f"不要编号和解释。\n候选项：{'、'.join(selected)}"
            )
            yield _record(index, "fixed_list", user, answer)
            index += 1

    for item in range(300):
        name = PRODUCTS[item % len(PRODUCTS)]
        city = CITIES[(item // 2) % len(CITIES)]
        status = STATUSES[(item // 3) % len(STATUSES)]
        order_number = f"KV{item + 1000}"
        answer = f"商品：{name}\n城市：{city}\n状态：{status}"
        assert len(answer.splitlines()) == 3
        user = (
            "请把以下信息整理成三行，严格使用“商品：值”“城市：值”“状态：值”的格式。\n"
            f"信息：编号{order_number}，{city}的一件{name}订单目前{status}。"
        )
        yield _record(index, "key_value_format", user, answer)
        index += 1

    for item in range(300):
        first = PRODUCTS[item % len(PRODUCTS)]
        second = PRODUCTS[(item + 3) % len(PRODUCTS)]
        first_count = item % 5 + 1
        second_count = (item + 2) % 5 + 1
        batch = f"B{item + 1000}"
        answer = (
            "| 商品 | 数量 | 批次 |\n|---|---:|---|\n"
            f"| {first} | {first_count} | {batch} |\n"
            f"| {second} | {second_count} | {batch} |"
        )
        assert answer.count("|") == 16
        user = (
            "请只用Markdown表格整理下面的数据，列名为“商品”“数量”和“批次”，不要添加解释。\n"
            f"数据：批次{batch}包含{first}{first_count}件、{second}{second_count}件。"
        )
        yield _record(index, "markdown_table", user, answer)
        index += 1
