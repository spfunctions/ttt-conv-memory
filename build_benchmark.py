"""
build_benchmark.py — generate benchmark_v1.json (300 samples × 3 layers).

Template-based with a Chinese fact pool. Reproducible (fixed seed).

Run:
    python build_benchmark.py [--seed 42] [--out benchmark_v1.json]

Output schema:
{
  "version": "v1",
  "seed": 42,
  "n_samples_per_level": 100,
  "samples": [
    {
      "sample_id": "L1-0042",
      "level": 1,
      "conversation": "...",
      "facts": [{"fact_id": "F1", "fact_text": "工号是 7742", "category": "numeric"}, ...],
      "probes": [{"probe_id": "P1", "question": "...", "gold_answer": "...", "required_facts": ["F1"]}, ...],
      "distractor": null  // string only for L3
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Fact pools — Chinese-primary because the user spec gave Chinese examples,
# and Qwen3-8B is multilingual so this works fine. English-equivalent pools
# could be added later but aren't needed for the literal validation.
# -----------------------------------------------------------------------------

NAMES = [
    "张伟", "李明", "王芳", "陈强", "刘洋", "周静", "吴磊", "黄敏", "赵刚", "孙丽",
    "钱进", "钟华", "马涛", "朱琳", "胡军", "郭斌", "林雪", "何鑫", "高帆", "罗娜",
    "梁志", "宋燕", "唐杰", "韩冰", "冯雷", "邓超", "曹颖", "彭浩", "曾婷", "肖敏",
    "田勇", "贾静", "傅明", "丁辉", "沈虹", "魏强", "蒋鹏", "薛莉", "段宇", "雷鸣",
    "苏晴", "潘越", "卢凯", "蔡蓓", "汪涛", "顾嘉", "侯亮", "邱阳", "石磊", "金雯",
    "戚浩", "邹雪", "尹辉", "崔颖", "钮晨", "万峰", "毛伟", "施静", "覃浩", "项云",
]

ROLES = [
    "市场部负责人", "技术总监", "项目经理", "产品经理", "财务总监", "运营经理",
    "销售总监", "设计主管", "研发主管", "人力资源经理", "运维总监", "战略顾问",
    "首席架构师", "测试经理", "客服主管", "采购经理", "法务总监", "数据分析主管",
]

DEPARTMENTS = [
    "市场部", "技术部", "产品部", "财务部", "运营部", "销售部",
    "设计部", "研发部", "人力资源部", "运维部", "战略部", "数据部",
]

PROJECTS = [
    "A项目", "B项目", "新产品发布", "Q2规划", "海外扩张", "用户调研",
    "迁移升级", "品牌重塑", "客户大会", "年度审计", "供应链优化", "渠道建设",
    "OKR复盘", "数据治理", "降本增效", "技术债清理", "合规整改", "客户体验改造",
]

LOCATIONS = [
    "3号楼12层", "总部大楼", "二楼会议室", "5号楼A区", "西区办公园", "南方分部",
    "北京总部", "上海办公室", "深圳分公司", "杭州研发中心", "成都团队", "武汉运营中心",
    "1号楼会议厅", "10楼茶水间", "8楼大会议室", "2号楼路演厅", "园区咖啡厅",
]

PREFERENCES = [
    "不吃辣", "偏好清淡", "喜欢早起", "不喝咖啡", "对花生过敏", "习惯靠窗座位",
    "喜欢安静的会议室", "讨厌长会", "偏好书面沟通", "晨型人", "夜猫子型",
    "工作时不接电话", "只在邮件里讨论合同", "需要全套素食", "不喝酒", "习惯站立工作",
]

# Sample some plausible numeric / temporal lookups
def emp_id_pool(seed_rng: random.Random) -> list[str]:
    return [f"{seed_rng.randint(1000, 9999)}" for _ in range(200)]

def budget_pool(seed_rng: random.Random) -> list[str]:
    return [f"{seed_rng.choice([n for n in range(50, 999, 5)])}万" for _ in range(200)]

def headcount_pool(seed_rng: random.Random) -> list[str]:
    return [str(seed_rng.choice([8, 12, 15, 20, 25, 30, 40, 50, 80, 100, 120])) for _ in range(50)]

WEEKDAYS = ["周一", "周二", "周三", "周四", "周五"]
DAYPARTS = ["上午", "下午", "晚上"]
HOURS = ["八点", "九点", "十点", "十一点", "一点", "两点", "三点", "四点", "五点", "六点", "七点"]

def time_pool(seed_rng: random.Random) -> list[str]:
    out = []
    for _ in range(200):
        out.append(f"{seed_rng.choice(WEEKDAYS)}{seed_rng.choice(DAYPARTS)}{seed_rng.choice(HOURS)}")
    return out

MONTHS = list(range(1, 13))
DAYS = list(range(1, 29))

def date_pool(seed_rng: random.Random) -> list[str]:
    return [f"{seed_rng.choice(MONTHS)}月{seed_rng.choice(DAYS)}号" for _ in range(200)]


# -----------------------------------------------------------------------------
# Conversation skeletons.
# Each skeleton is (template_string, slot_specs).
# slot_specs = list of (slot_name, category) — used both to fill the conversation
# and to register facts and generate probes.
# Probes use the slot_name as the canonical handle.
# -----------------------------------------------------------------------------

# A "skeleton" is a multi-turn dialogue with {slot} placeholders that get filled
# from the fact pool. After filling, each slot becomes a registered fact + probe.

SKELETONS = [
    # SK-1 — workplace coordination
    {
        "id": "SK-1",
        "text": (
            "{p1}：你好，{name_a}，最近怎么样？\n"
            "{p2}：还行，最近在忙{project}。{name_b}也参与了。\n"
            "{p1}：哦{project}啊，截止日期记得是{deadline}对吧？预算多少？\n"
            "{p2}：嗯，{deadline}前要交。预算{budget}。{name_b}是{role_b}，主要负责对接客户。\n"
            "{p1}：{name_b}向谁汇报？\n"
            "{p2}：向{name_c}汇报，{name_c}是{role_c}。\n"
            "{p1}：明白。你工号还是{empid}吧？\n"
            "{p2}：嗯，{empid}。会议室订{location}，行不行？\n"
            "{p1}：行。我{pref}，午餐记得安排清淡点。\n"
            "{p2}：好。"
        ),
        "slots": [
            ("p1", "literal_pronoun"),
            ("p2", "literal_pronoun"),
            ("name_a", "person"),
            ("name_b", "person"),
            ("name_c", "person"),
            ("project", "project"),
            ("deadline", "date"),
            ("budget", "numeric"),
            ("role_b", "role"),
            ("role_c", "role"),
            ("empid", "numeric"),
            ("location", "spatial"),
            ("pref", "preference"),
        ],
        # For each registerable slot (i.e. excluding p1/p2 literals), how to phrase
        # the fact and probe.
        "facts_and_probes": [
            # (slot, category, fact_template, question_template, gold_from_slot)
            ("name_b", "person", "{name_b} 是 {project} 的参与者", "{project} 由谁参与？", "name_b"),
            ("name_c", "person", "{name_c} 是 {name_b} 的上级", "{name_b} 向谁汇报？", "name_c"),
            ("project", "project", "目前在做的项目是 {project}", "目前在做什么项目？", "project"),
            ("deadline", "temporal", "{project} 的截止日期是 {deadline}", "{project} 什么时候截止？", "deadline"),
            ("budget", "numeric", "{project} 的预算是 {budget}", "{project} 的预算是多少？", "budget"),
            ("role_b", "relational", "{name_b} 的职位是 {role_b}", "{name_b} 是什么职位？", "role_b"),
            ("role_c", "relational", "{name_c} 的职位是 {role_c}", "{name_c} 是什么职位？", "role_c"),
            ("empid", "numeric", "{p2} 的工号是 {empid}", "{p2} 的工号是多少？", "empid"),
            ("location", "spatial", "会议室在 {location}", "会议室在哪里？", "location"),
            ("pref", "preference", "{p1} {pref}", "{p1} 的饮食偏好是什么？", "pref"),
        ],
    },
    # SK-2 — onboarding / HR
    {
        "id": "SK-2",
        "text": (
            "{p1}：欢迎来到公司，{name_a}。先做个简单介绍吧。\n"
            "{p2}：好的。我叫{name_a}，加入{dept}，工号{empid}。\n"
            "{p1}：你的直属领导是{name_b}，{name_b}是{role_b}。办公室在{location}。\n"
            "{p2}：明白。我入职日期是{deadline}对吗？\n"
            "{p1}：对，{deadline}正式入职。第一个项目是{project}，预算{budget}。\n"
            "{p2}：人手够吗？\n"
            "{p1}：团队总共{headcount}人。会议时间是{daytime}，每周一次。\n"
            "{p2}：好的。另外我{pref}，午餐时麻烦留意一下。"
        ),
        "slots": [
            ("p1", "literal_pronoun"),
            ("p2", "literal_pronoun"),
            ("name_a", "person"),
            ("name_b", "person"),
            ("dept", "department"),
            ("empid", "numeric"),
            ("role_b", "role"),
            ("location", "spatial"),
            ("deadline", "date"),
            ("project", "project"),
            ("budget", "numeric"),
            ("headcount", "numeric"),
            ("daytime", "temporal"),
            ("pref", "preference"),
        ],
        "facts_and_probes": [
            ("name_a", "person", "新同事叫 {name_a}", "新同事叫什么？", "name_a"),
            ("name_b", "person", "{name_a} 的直属领导是 {name_b}", "{name_a} 的直属领导是谁？", "name_b"),
            ("dept", "relational", "{name_a} 加入 {dept}", "{name_a} 在哪个部门？", "dept"),
            ("empid", "numeric", "{name_a} 的工号是 {empid}", "{name_a} 的工号是多少？", "empid"),
            ("role_b", "relational", "{name_b} 的职位是 {role_b}", "{name_b} 是什么职位？", "role_b"),
            ("location", "spatial", "{name_a} 办公室在 {location}", "{name_a} 办公室在哪里？", "location"),
            ("deadline", "temporal", "{name_a} 入职日期是 {deadline}", "{name_a} 什么时候入职？", "deadline"),
            ("project", "project", "{name_a} 的第一个项目是 {project}", "{name_a} 的第一个项目是什么？", "project"),
            ("budget", "numeric", "{project} 的预算是 {budget}", "{project} 的预算是多少？", "budget"),
            ("headcount", "numeric", "团队总共 {headcount} 人", "团队多少人？", "headcount"),
            ("daytime", "temporal", "每周会议时间是 {daytime}", "每周会议什么时候？", "daytime"),
            ("pref", "preference", "{name_a} {pref}", "{name_a} 的饮食偏好是什么？", "pref"),
        ],
    },
    # SK-3 — vendor / partner discussion
    {
        "id": "SK-3",
        "text": (
            "{p1}：{name_a}，关于{project}的供应商对接你做得怎么样了？\n"
            "{p2}：进度还行。供应商联系人是{name_b}，{role_b}。报价{budget}。\n"
            "{p1}：对方公司在哪？\n"
            "{p2}：办公地点在{location}。{name_b}的工号或对方代号是{empid}。\n"
            "{p1}：交付时间敲定了吗？\n"
            "{p2}：交付定在{deadline}，会议安排在{daytime}走流程。\n"
            "{p1}：他们老板是{name_c}吧？\n"
            "{p2}：是的，{name_c}是{role_c}。我跟{name_b}打交道更多。\n"
            "{p1}：行。对了，下次客户晚宴别忘了，他们{pref}。"
        ),
        "slots": [
            ("p1", "literal_pronoun"),
            ("p2", "literal_pronoun"),
            ("name_a", "person"),
            ("name_b", "person"),
            ("name_c", "person"),
            ("project", "project"),
            ("role_b", "role"),
            ("role_c", "role"),
            ("budget", "numeric"),
            ("location", "spatial"),
            ("empid", "numeric"),
            ("deadline", "date"),
            ("daytime", "temporal"),
            ("pref", "preference"),
        ],
        "facts_and_probes": [
            ("name_b", "person", "{project} 的供应商联系人是 {name_b}", "{project} 的供应商联系人是谁？", "name_b"),
            ("role_b", "relational", "{name_b} 的职位是 {role_b}", "{name_b} 是什么职位？", "role_b"),
            ("budget", "numeric", "供应商报价是 {budget}", "供应商报价是多少？", "budget"),
            ("location", "spatial", "供应商办公地点是 {location}", "供应商办公地点在哪里？", "location"),
            ("empid", "numeric", "对方代号是 {empid}", "对方代号是什么？", "empid"),
            ("deadline", "temporal", "交付时间是 {deadline}", "交付时间是什么时候？", "deadline"),
            ("daytime", "temporal", "对接会议时间是 {daytime}", "对接会议什么时候？", "daytime"),
            ("name_c", "person", "{name_c} 是供应商方老板", "供应商方老板是谁？", "name_c"),
            ("role_c", "relational", "{name_c} 的职位是 {role_c}", "{name_c} 的职位是什么？", "role_c"),
            ("pref", "preference", "客户 {pref}", "客户的饮食偏好是什么？", "pref"),
        ],
    },
    # SK-4 — personal / scheduling
    {
        "id": "SK-4",
        "text": (
            "{p1}：{name_a}，下周客户拜访的安排发我一下。\n"
            "{p2}：好。客户公司在{location}，对接人{name_b}，{role_b}。\n"
            "{p1}：哪天去？\n"
            "{p2}：{daytime}出发，回来定在{deadline}。预算开支{budget}。\n"
            "{p1}：客户那边其他需要注意的吗？\n"
            "{p2}：{name_b}的助理叫{name_c}，他们老板{role_c}也会出席。客户编号是{empid}。\n"
            "{p1}：行。请确认{name_b}{pref}，午餐做相应安排。\n"
            "{p2}：好的。这次重点是{project}的合同签订。"
        ),
        "slots": [
            ("p1", "literal_pronoun"),
            ("p2", "literal_pronoun"),
            ("name_a", "person"),
            ("name_b", "person"),
            ("name_c", "person"),
            ("project", "project"),
            ("role_b", "role"),
            ("role_c", "role"),
            ("budget", "numeric"),
            ("location", "spatial"),
            ("empid", "numeric"),
            ("deadline", "date"),
            ("daytime", "temporal"),
            ("pref", "preference"),
        ],
        "facts_and_probes": [
            ("location", "spatial", "客户公司在 {location}", "客户公司在哪？", "location"),
            ("name_b", "person", "客户对接人是 {name_b}", "客户对接人是谁？", "name_b"),
            ("role_b", "relational", "{name_b} 的职位是 {role_b}", "{name_b} 的职位是什么？", "role_b"),
            ("daytime", "temporal", "出发时间是 {daytime}", "什么时候出发？", "daytime"),
            ("deadline", "temporal", "回来时间是 {deadline}", "什么时候回来？", "deadline"),
            ("budget", "numeric", "预算开支 {budget}", "预算开支多少？", "budget"),
            ("name_c", "person", "{name_b} 的助理是 {name_c}", "{name_b} 的助理是谁？", "name_c"),
            ("role_c", "relational", "客户老板的职位是 {role_c}", "客户老板的职位是什么？", "role_c"),
            ("empid", "numeric", "客户编号是 {empid}", "客户编号是多少？", "empid"),
            ("project", "project", "本次重点是 {project} 合同", "本次重点是什么？", "project"),
            ("pref", "preference", "{name_b} {pref}", "{name_b} 的饮食偏好是什么？", "pref"),
        ],
    },
    # SK-5 — internal review meeting
    {
        "id": "SK-5",
        "text": (
            "{p1}：{daytime}的复盘会，谁主持？\n"
            "{p2}：{name_a}主持。{name_a}是{role_b}。\n"
            "{p1}：哪个会议室？\n"
            "{p2}：{location}。预计{headcount}人参加。\n"
            "{p1}：议题呢？\n"
            "{p2}：主要复盘{project}，KPI完成度和{budget}的预算执行情况。\n"
            "{p1}：{name_b}过来吗？\n"
            "{p2}：会的。{name_b}是{role_c}，向{name_a}汇报。{name_b}工号{empid}。\n"
            "{p1}：截止节点呢？\n"
            "{p2}：{deadline}前要出复盘报告。{name_a}{pref}，会议茶歇时注意。"
        ),
        "slots": [
            ("p1", "literal_pronoun"),
            ("p2", "literal_pronoun"),
            ("name_a", "person"),
            ("name_b", "person"),
            ("project", "project"),
            ("role_b", "role"),
            ("role_c", "role"),
            ("budget", "numeric"),
            ("location", "spatial"),
            ("empid", "numeric"),
            ("deadline", "date"),
            ("daytime", "temporal"),
            ("headcount", "numeric"),
            ("pref", "preference"),
        ],
        "facts_and_probes": [
            ("daytime", "temporal", "复盘会时间是 {daytime}", "复盘会什么时候？", "daytime"),
            ("name_a", "person", "复盘会主持人是 {name_a}", "复盘会主持人是谁？", "name_a"),
            ("role_b", "relational", "{name_a} 的职位是 {role_b}", "{name_a} 的职位是什么？", "role_b"),
            ("location", "spatial", "复盘会会议室是 {location}", "复盘会会议室在哪？", "location"),
            ("headcount", "numeric", "复盘会参加人数 {headcount}", "复盘会预计多少人？", "headcount"),
            ("project", "project", "复盘的项目是 {project}", "复盘的项目是什么？", "project"),
            ("budget", "numeric", "复盘的预算是 {budget}", "复盘的预算是多少？", "budget"),
            ("name_b", "person", "{name_b} 也参加复盘会", "复盘会还有谁参加？", "name_b"),
            ("role_c", "relational", "{name_b} 的职位是 {role_c}", "{name_b} 的职位是什么？", "role_c"),
            ("empid", "numeric", "{name_b} 工号 {empid}", "{name_b} 的工号是多少？", "empid"),
            ("deadline", "temporal", "复盘报告截止 {deadline}", "复盘报告什么时候截止？", "deadline"),
            ("pref", "preference", "{name_a} {pref}", "{name_a} 的饮食偏好是什么？", "pref"),
        ],
    },
]


# -----------------------------------------------------------------------------
# Distractor passages — used in level 3.
# Each passage is 2000-4000 tokens of unrelated content. We use a few canned
# Chinese-language technical/news articles of sufficient length, then truncate
# / repeat to land in the target window.
# -----------------------------------------------------------------------------

DISTRACTOR_SEEDS = [
    # ~600 character Chinese tech essay; we'll repeat to fill 2000-4000 tokens
    """随着深度学习模型规模的不断扩大，训练成本和推理成本同步增长。从GPT-3的1750亿参数到目前主流的万亿级别，硬件资源的瓶颈已经从计算单元逐渐转向了内存带宽和数据传输。研究人员正在探索多种优化方向，包括混合精度训练、张量并行、流水线并行、ZeRO优化器等技术。其中，混合精度训练通过在FP16/BF16和FP32之间灵活切换，可以在保证模型收敛的前提下显著降低显存占用。张量并行则将单个矩阵乘法拆分到多个设备上，适合于Transformer的注意力和前馈层。流水线并行将模型按层切分，相邻设备之间通过激活值传递。ZeRO的核心思想是将优化器状态、梯度、参数分别分片到不同设备，最大化利用集群整体内存。这些技术的组合使用是当前万亿级别模型训练的标准配方。在推理侧，KV缓存管理、推测解码、量化等技术也成为研究热点。FlashAttention通过重新组织注意力计算的内存访问模式，将长上下文场景下的吞吐显著提高。PagedAttention则借鉴了操作系统的虚拟内存思想，将KV缓存分页管理。在量化方面，INT8、INT4甚至更低比特宽度的量化方案不断涌现，配合相应的解量化和重要性度量，可以在精度损失可控的前提下大幅缩减部署成本。""",
    """近年来，量子计算从理论走向工程实现的进程明显加速。超导量子比特、离子阱、光子和拓扑等多种物理实现路径百花齐放。超导路线以其相对成熟的微纳工艺基础，在比特数和保真度上取得了显著进展。IBM、谷歌等公司公布的处理器规模已经突破千比特量级，错误率也在不断下降。离子阱路线则以高保真度和长相干时间见长，在量子化学和优化问题上展现出独特优势。光子量子计算具有室温运行的潜力，在通信和分布式计算领域有应用前景。拓扑量子计算虽然尚未实现工程化的量子比特，但其拓扑保护带来的天然容错性是长期研究方向。在算法层面，量子近似优化算法、变分量子本征求解器、量子机器学习等NISQ时代的应用范式不断完善。在容错量子计算方向，表面码、Bacon-Shor码、彩色码等纠错方案各有侧重。距离量子优势的真正实现还有相当长的路要走，但工程化进度令人鼓舞。""",
    """生命科学与人工智能的交叉正在催生新的科研范式。AlphaFold系列工作彻底改变了蛋白质结构预测的格局，原本需要数月甚至数年的实验确定结构的工作现在可以在几小时内完成预测。这一进展直接推动了药物发现、酶工程、合成生物学等下游领域的加速。在基因组学领域，大规模序列模型如Evo、ProtGPT、ESM等开始在基因组语言建模、突变效应预测、功能注释等任务上展现出超越传统方法的能力。在神经科学方向，借助海量神经成像数据训练的基础模型，开始能够从fMRI信号中重建图像和语言。在精准医学方面，多模态生物医学大模型整合临床记录、影像、基因组、代谢组等数据，为个体化诊疗提供决策支持。这些进展的共同特征是数据规模的飞跃和模型架构的通用化。基础模型的"涌现"现象在生命科学领域同样可见——当训练数据和参数规模超过某个临界点，模型会展现出未在训练目标中明确指定的能力。这种"通用智能"在医学领域的实现路径，可能比许多人预想的更近。""",
]


# -----------------------------------------------------------------------------
# Pronouns for placeholders (kept literal, not registered as facts)
# -----------------------------------------------------------------------------

PRONOUN_PAIRS = [
    ("A", "B"),
    ("我", "对方"),
    ("领导", "我"),
    ("HR", "新同事"),
    ("项目方", "供应商"),
]


# -----------------------------------------------------------------------------
# Sample generation
# -----------------------------------------------------------------------------

def pool_for_category(cat: str, rng: random.Random) -> list[str]:
    """Return the fact-pool list for a given category."""
    if cat == "person": return NAMES
    if cat == "role": return ROLES
    if cat == "department": return DEPARTMENTS
    if cat == "project": return PROJECTS
    if cat == "spatial": return LOCATIONS
    if cat == "preference": return PREFERENCES
    if cat == "numeric": return rng.choice([emp_id_pool(rng), budget_pool(rng), headcount_pool(rng)])
    if cat == "date": return date_pool(rng)
    if cat == "temporal": return time_pool(rng)
    raise ValueError(f"Unknown category: {cat}")


def gen_l1_sample(sample_idx: int, rng: random.Random) -> dict[str, Any]:
    """Generate a single Level 1 sample."""
    skel = rng.choice(SKELETONS)
    pron = rng.choice(PRONOUN_PAIRS)

    # Build slot-value map. Every slot gets a unique value; for category=person
    # we ensure no repeats within a sample.
    seen = {"p1": pron[0], "p2": pron[1]}
    for slot, cat in skel["slots"]:
        if slot in seen:
            continue
        candidates = pool_for_category(cat, rng) if cat != "literal_pronoun" else None
        if not candidates:
            seen[slot] = ""
            continue
        # avoid clashing names within a sample
        for _ in range(50):
            v = rng.choice(candidates)
            if cat == "person" and v in seen.values():
                continue
            seen[slot] = v
            break
        else:
            seen[slot] = rng.choice(candidates)

    # Render conversation
    conversation = skel["text"].format(**seen)

    # Build facts and probes
    facts: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    for i, (slot, cat, fact_tpl, q_tpl, gold_slot) in enumerate(skel["facts_and_probes"]):
        fact_id = f"F{i+1}"
        probe_id = f"P{i+1}"
        facts.append({
            "fact_id": fact_id,
            "fact_text": fact_tpl.format(**seen),
            "category": cat,
            "slot": slot,
        })
        probes.append({
            "probe_id": probe_id,
            "question": q_tpl.format(**seen),
            "gold_answer": seen[gold_slot],
            "required_facts": [fact_id],
        })

    return {
        "sample_id": f"L1-{sample_idx:04d}",
        "level": 1,
        "skeleton_id": skel["id"],
        "conversation": conversation,
        "facts": facts,
        "probes": probes,
        "distractor": None,
    }


def gen_l2_sample(sample_idx: int, l1_sample: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Build a Level-2 (composed-fact) probe set on top of an L1 sample.

    Strategy: pair facts that share a person or project entity, then ask a
    question that requires both. Falls back to L1 probes if not enough pairs.
    """
    facts = l1_sample["facts"]
    skeleton_id = l1_sample["skeleton_id"]
    skel = next(s for s in SKELETONS if s["id"] == skeleton_id)

    # Build a slot→fact_id map for cross-references
    slot_to_fid = {f["slot"]: f["fact_id"] for f in facts}

    # Composed-probe templates per skeleton — pre-authored so probes are unambiguous.
    # Each composed probe = (slot_a, slot_b, question_template, gold_template)
    composed_specs = {
        "SK-1": [
            ("name_b", "role_b", "项目的参与者职位是什么？", "{role_b}"),
            ("project", "deadline", "目前在做的项目什么时候截止？", "{deadline}"),
            ("project", "budget", "目前在做的项目预算多少？", "{budget}"),
            ("name_b", "name_c", "{name_b} 向谁汇报？", "{name_c}"),
            ("location", "pref", "会议室在哪，{p1} 的饮食偏好是什么？", "{location}；{pref}"),
        ],
        "SK-2": [
            ("name_a", "dept", "新同事在哪个部门？", "{dept}"),
            ("name_a", "empid", "新同事的工号是多少？", "{empid}"),
            ("name_a", "deadline", "新同事什么时候入职？", "{deadline}"),
            ("name_a", "name_b", "新同事的领导是谁？", "{name_b}"),
            ("project", "budget", "新同事的第一个项目预算多少？", "{budget}"),
        ],
        "SK-3": [
            ("project", "name_b", "项目的供应商联系人是谁？", "{name_b}"),
            ("name_b", "role_b", "供应商联系人是什么职位？", "{role_b}"),
            ("project", "budget", "供应商对项目的报价是多少？", "{budget}"),
            ("name_c", "role_c", "供应商方老板的职位是什么？", "{role_c}"),
            ("project", "deadline", "项目交付时间是什么时候？", "{deadline}"),
        ],
        "SK-4": [
            ("name_b", "role_b", "客户对接人的职位是什么？", "{role_b}"),
            ("name_b", "location", "客户对接人公司在哪？", "{location}"),
            ("name_b", "name_c", "客户对接人的助理是谁？", "{name_c}"),
            ("project", "budget", "本次重点项目的预算是多少？", "{budget}"),
            ("daytime", "deadline", "出发和回来分别是什么时候？", "{daytime}；{deadline}"),
        ],
        "SK-5": [
            ("name_a", "role_b", "复盘会主持人是什么职位？", "{role_b}"),
            ("project", "budget", "复盘项目的预算是多少？", "{budget}"),
            ("name_b", "role_c", "{name_b} 是什么职位？", "{role_c}"),
            ("name_b", "empid", "{name_b} 的工号是多少？", "{empid}"),
            ("daytime", "location", "复盘会什么时候开，在哪个会议室？", "{daytime}；{location}"),
        ],
    }

    specs = composed_specs.get(skeleton_id, [])

    # Render the slot → value map back from the L1 sample so we can build gold answers
    # by re-parsing facts. Easier: regenerate slot map from skeleton + sample.
    # We saved the slot in each fact, so we can reconstruct.
    fact_slots = {f["slot"]: f for f in facts}

    # We need the literal value of each slot. Pull from the conversation by way of
    # the templated fact texts: each fact_text was generated from `fact_tpl.format(**seen)`
    # which means the `seen` map is what we need. Easiest: re-derive from probes (which
    # used `gold_slot=slot` for primary L1 probes).
    slot_values: dict[str, str] = {}
    for p in l1_sample["probes"]:
        # primary L1 probes carry one required_fact that maps 1:1 to a slot
        fid = p["required_facts"][0]
        f = next(x for x in facts if x["fact_id"] == fid)
        slot_values[f["slot"]] = p["gold_answer"]
    # Add literal pronouns
    # (We don't actually need them for L2 probes since composed templates above only
    #  use registered slots, not p1/p2 directly — but they're also used in some.)
    # Best-effort: leave undefined slots empty.

    # Fill literal pronouns from the conversation by parsing the first character before "："
    # Hack: skip — composed_specs above uses {p1} only in SK-1 and we'll just substitute
    # an empty string. To avoid that, we reuse SK-1's pronoun pair from the conversation.
    # Simpler: set p1/p2 by detecting the literal first turn.
    first_line = l1_sample["conversation"].split("\n")[0]
    if "：" in first_line:
        p1_value = first_line.split("：", 1)[0]
        slot_values["p1"] = p1_value
    second_line_match = [ln for ln in l1_sample["conversation"].split("\n") if "：" in ln and ln.split("：", 1)[0] != slot_values.get("p1", "")]
    if second_line_match:
        slot_values["p2"] = second_line_match[0].split("：", 1)[0]

    composed_probes: list[dict[str, Any]] = []
    composed_facts: list[dict[str, Any]] = list(facts)  # carry the original facts
    for j, (slot_a, slot_b, q_tpl, gold_tpl) in enumerate(specs):
        if slot_a not in slot_values or slot_b not in slot_values:
            continue
        try:
            question = q_tpl.format(**slot_values)
            gold = gold_tpl.format(**slot_values)
        except KeyError:
            continue
        composed_probes.append({
            "probe_id": f"C{j+1}",
            "question": question,
            "gold_answer": gold,
            "required_facts": [
                slot_to_fid.get(slot_a, ""),
                slot_to_fid.get(slot_b, ""),
            ],
        })

    return {
        "sample_id": f"L2-{sample_idx:04d}",
        "level": 2,
        "skeleton_id": skeleton_id,
        "conversation": l1_sample["conversation"],
        "facts": composed_facts,
        "probes": composed_probes,
        "distractor": None,
    }


def gen_l3_sample(sample_idx: int, l1_sample: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Level 3: same as L1 but with appended distractor (2000-4000 tokens)."""
    target_tokens = rng.randint(2000, 4000)
    # rough heuristic: 1 Chinese char ≈ 1 token; we use char count as proxy
    seed_passages = list(DISTRACTOR_SEEDS)
    rng.shuffle(seed_passages)
    out = ""
    while len(out) < target_tokens:
        out += rng.choice(seed_passages) + "\n\n"
    distractor = out[:target_tokens]

    return {
        "sample_id": f"L3-{sample_idx:04d}",
        "level": 3,
        "skeleton_id": l1_sample["skeleton_id"],
        "conversation": l1_sample["conversation"],
        "facts": l1_sample["facts"],
        "probes": l1_sample["probes"],  # same probes as L1
        "distractor": distractor,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-per-level", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("benchmark_v1.json"))
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # L1 first — L2 and L3 are derived from L1 conversations
    l1_samples = [gen_l1_sample(i, rng) for i in range(args.n_per_level)]
    l2_samples = [gen_l2_sample(i, l1_samples[i], rng) for i in range(args.n_per_level)]
    l3_samples = [gen_l3_sample(i, l1_samples[i], rng) for i in range(args.n_per_level)]

    all_samples = l1_samples + l2_samples + l3_samples

    out = {
        "version": "v1",
        "seed": args.seed,
        "n_samples_per_level": args.n_per_level,
        "n_total": len(all_samples),
        "samples": all_samples,
    }

    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    # Print summary
    print(f"Wrote {args.out} with {len(all_samples)} samples")
    print(f"  L1: {len(l1_samples)} (avg {sum(len(s['probes']) for s in l1_samples) / len(l1_samples):.1f} probes/sample)")
    print(f"  L2: {len(l2_samples)} (avg {sum(len(s['probes']) for s in l2_samples) / len(l2_samples):.1f} probes/sample)")
    print(f"  L3: {len(l3_samples)} (avg {sum(len(s['probes']) for s in l3_samples) / len(l3_samples):.1f} probes/sample, "
          f"avg {sum(len(s['distractor']) for s in l3_samples) / len(l3_samples):.0f} distractor chars)")
    cat_counts: dict[str, int] = {}
    for s in l1_samples:
        for f in s["facts"]:
            cat_counts[f["category"]] = cat_counts.get(f["category"], 0) + 1
    print(f"  fact category distribution: {sorted(cat_counts.items(), key=lambda x: -x[1])}")


if __name__ == "__main__":
    main()
