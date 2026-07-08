"""
Prompt Builder (中文版) - 組裝最終的 System Prompt
根據 behavior_specs、角色資訊、RAG 檢索結果來構建 system prompt

🆕 v2 結構（對應 behavior_specs v2 schema）：
    - Core Principles / Roleplay Rules 為純字串陣列
    - Examples 完整放入 prompt（輸出範例段落）
    - 角色段落使用固定索引檢索的性格描述（character_personality）
    - 以當前對話當索引檢索的 RAG 段落，標題一律加上「與當前對話最相關的」
    - 對話歷史不在此組裝——由 chat_service 以原生 chat messages 格式送出

※ 本檔為中文對照版，實際運行使用英文版 prompt_builder.py
"""


class PromptBuilder:
    """System Prompt 組裝工具"""

    @staticmethod
    def build_system_prompt(behavior_specs: dict, character_info: dict, conversation_history: list = None, rag_context: dict = None, protagonist_name: str = None) -> str:
        """
        組裝最終的 system prompt

        參數：
            behavior_specs: dict，從 behavior_specs JSON 讀入（v2 schema）
                包含：Mission、Core Principles（字串陣列）、Roleplay Rules（字串陣列）、
                      Writing Style（字串陣列）、Response Format（{rules: 字串陣列}）、
                      Examples（字串陣列）
            character_info: dict，角色基本資訊 {name, gender, tags}
            conversation_history: （未使用）歷史以原生 chat messages 格式送出
            rag_context: dict，RAG 檢索結果（可選）
                {character_background, fewshots, summaries,
                 protagonist_background, character_personality}
            protagonist_name: str，主角（使用者扮演的人物）名稱，可選

        返回：
            str，最終的 system prompt
        """

        # 1. 提取 behavior_specs 各部分（v2 schema：純字串陣列）
        mission = behavior_specs.get("Mission", "")
        core_principles = behavior_specs.get("Core Principles", [])
        roleplay_rules = behavior_specs.get("Roleplay Rules", [])
        writing_style = behavior_specs.get("Writing Style", [])
        response_format = behavior_specs.get("Response Format", {})
        examples = behavior_specs.get("Examples", [])

        # 2. 提取角色資訊
        char_name = character_info.get("name", "未定義")
        char_gender = character_info.get("gender", "未知")
        char_tags = character_info.get("tags", [])
        tag_str = ", ".join(char_tags) if char_tags else "無"

        # 3. 核心原則（純字串）
        principles_str = ""
        if core_principles:
            principles_str = "\n## 核心原則\n"
            for principle in core_principles:
                principles_str += f"- {principle}\n"

        # 4. 角色扮演規則
        roleplay_rules_str = ""
        if roleplay_rules:
            roleplay_rules_str = "\n## 角色扮演規則\n"
            for rule in roleplay_rules:
                roleplay_rules_str += f"- {rule}\n"

        # 5. 寫作風格
        writing_style_str = ""
        if writing_style:
            writing_style_str = "\n## 寫作風格\n"
            for style in writing_style:
                writing_style_str += f"- {style}\n"

        # 6. 回應格式
        response_format_str = ""
        if response_format:
            rules = response_format.get("rules", [])
            if rules:
                response_format_str = "\n## 回應格式\n"
                for rule in rules:
                    response_format_str += f"- {rule}\n"

        # 7. 🆕 輸出範例（純字串，展示敘事+對話格式）
        examples_str = ""
        if examples:
            examples_str = "\n## 輸出範例\n"
            for idx, example in enumerate(examples, 1):
                examples_str += f"\n**範例 {idx}:**\n{example}\n"

        # 8. 🆕 你主要扮演的角色
        #    性格描述來自固定索引的 RAG 檢索（用固定語意關鍵詞，不用當前對話）
        character_str = f"""
## 你主要扮演的角色
- **姓名**: {char_name}
- **性別**: {char_gender}
- **核心特徵**: [{tag_str}]"""

        personality = rag_context.get("character_personality") if rag_context else None
        if personality:
            character_str += "\n### 你必須依照角色性格描述來扮演角色"
            for p in personality:
                character_str += f"\n- {p}"

        # 8.5 🆕 背景與對話範例緊接在性格描述之後（同屬「你主要扮演的角色」段落，無獨立 ## 標題）
        if rag_context:
            # 與當前對話最相關的背景資訊（列表，逐條列出）
            if rag_context.get("character_background"):
                character_str += "\n### 與當前對話最相關的背景資訊"
                for bg in rag_context["character_background"]:
                    character_str += f"\n- {bg}"

            # 與當前對話最相關的對話範例
            if rag_context.get("fewshots"):
                character_str += "\n### 與當前對話最相關的對話範例"
                for idx, fewshot in enumerate(rag_context["fewshots"], 1):
                    character_str += f"\n**範例 {idx}:**\n{fewshot}"

        # 9. 🆕 由使用者扮演的主角
        protagonist_str = ""
        protagonist_background = rag_context.get("protagonist_background") if rag_context else None
        if protagonist_name or protagonist_background:
            protagonist_str = "\n\n## 由使用者扮演的主角"
            if protagonist_name:
                protagonist_str += f"\n- **姓名**: {protagonist_name}"
            if protagonist_background:
                # 列表，逐條列出
                protagonist_str += "\n### 與當前對話最相關的主角背景"
                for pb in protagonist_background:
                    protagonist_str += f"\n- {pb}"

        # 10. 歷史摘要（放最後，獨立段落）
        rag_context_str = ""
        if rag_context and rag_context.get("summaries"):
            rag_context_str = "\n\n## 與當前對話最相關的歷史摘要"
            for idx, summary in enumerate(rag_context["summaries"], 1):
                rag_context_str += f"\n**摘要 {idx}:**\n{summary}"

        # 11. 組裝最終 system prompt
        # （對話歷史以原生 chat messages 格式送出，不在此組裝）
        prompt = f"""# 系統角色扮演指令

## 任務
{mission}

{principles_str}{roleplay_rules_str}{writing_style_str}{response_format_str}{examples_str}{character_str}{protagonist_str}{rag_context_str}
"""

        final_prompt = prompt.strip()
        print(f"\n🔧 [prompt_builder] 最終 System Prompt:\n{'='*80}")
        print(final_prompt)
        print(f"{'='*80}\n")

        return final_prompt


# 全局實例
prompt_builder = PromptBuilder()
