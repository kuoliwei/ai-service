"""
Prompt Builder - 組裝最終的 System Prompt
根據 behavior_specs、角色資訊、聊天紀錄來構建 system prompt
"""


class PromptBuilder:
    """System Prompt 組裝工具"""

    @staticmethod
    def build_system_prompt(behavior_specs: dict, character_info: dict, conversation_history: list = None, rag_context: dict = None) -> str:
        """
        組裝最終的 system prompt

        參數：
            behavior_specs: dict，從 behavior_specs JSON 讀進來
                包含：Mission, Core Principles, Conversation Policies,
                      Generation Policies, Writing Style, Response Format
            character_info: dict，角色基本資訊
                {name: str, gender: str, tags: list}
            conversation_history: list，對話歷史（可選，用於上下文）
                [{role: str, text: str}, ...]
            rag_context: dict，RAG 檢索結果（可選）
                {character_background: str, fewshots: list}

        返回：
            str，最終的 system prompt
        """

        # 1. 提取 behavior_specs 的各個部分
        mission = behavior_specs.get("Mission", "")
        core_principles = behavior_specs.get("Core Principles", [])
        conversation_policies = behavior_specs.get("Conversation Policies", [])
        generation_policies = behavior_specs.get("Generation Policies", [])
        writing_style = behavior_specs.get("Writing Style", [])
        response_format = behavior_specs.get("Response Format", {})

        # 2. 提取角色資訊
        char_name = character_info.get("name", "未定義")
        char_gender = character_info.get("gender", "未知")
        char_tags = character_info.get("tags", [])
        tag_str = ", ".join(char_tags) if char_tags else "無"

        # 3. 構建核心身份邊界
        boundary_warning = f"""### ⚠️ 核心身份邊界 (最高優先級):
- 你當前的身份是【{char_name}】。
- 嚴禁代替使用者決定心理、想法或行為。
- 當完成回應後，請立即停止輸出，等待使用者輸入。"""

        # 4. 構建 Core Principles 部分
        principles_str = ""
        if core_principles:
            principles_str = "\n## 核心原則\n"
            for principle in core_principles:
                title = principle.get("title", "")
                description = principle.get("description", "")
                principles_str += f"- **{title}**: {description}\n"

        # 5. 構建 Conversation Policies 部分
        conversation_policies_str = ""
        if conversation_policies:
            conversation_policies_str = "\n## 對話政策\n"
            for policy in conversation_policies:
                conversation_policies_str += f"- {policy}\n"

        # 6. 構建 Generation Policies 部分
        generation_policies_str = ""
        if generation_policies:
            generation_policies_str = "\n## 生成政策\n"
            for policy in generation_policies:
                generation_policies_str += f"- {policy}\n"

        # 7. 構建 Writing Style 部分
        writing_style_str = ""
        if writing_style:
            writing_style_str = "\n## 寫作風格\n"
            for style in writing_style:
                writing_style_str += f"- {style}\n"

        # 8. 構建 Response Format 部分
        response_format_str = ""
        if response_format:
            rules = response_format.get("rules", [])
            if rules:
                response_format_str = "\n## 回應格式\n"
                for rule in rules:
                    response_format_str += f"- {rule}\n"

        # 9. 構建角色資訊部分
        character_info_str = f"""
## 角色資訊
- **姓名**: {char_name}
- **性別**: {char_gender}
- **核心特徵**: [{tag_str}]"""

        # 10. 構建 RAG 上下文部分
        rag_context_str = ""
        if rag_context:
            rag_context_str = "\n## 角色背景與範例"

            # 添加角色背景
            if rag_context.get("character_background"):
                rag_context_str += f"\n### 相關背景資訊\n{rag_context['character_background']}"

            # 添加 few-shots
            if rag_context.get("fewshots"):
                rag_context_str += "\n### 對話範例"
                for idx, fewshot in enumerate(rag_context["fewshots"], 1):
                    rag_context_str += f"\n**範例 {idx}:**\n{fewshot}"

        # 11. 組裝最終 system prompt
        prompt = f"""# 系統角色扮演指令

## 任務
{mission}

{boundary_warning}

{principles_str}{conversation_policies_str}{generation_policies_str}{writing_style_str}{response_format_str}{character_info_str}{rag_context_str}
"""

        final_prompt = prompt.strip()
        print(f"\n🔧 [prompt_builder] 最終 System Prompt:\n{'='*80}")
        print(final_prompt)
        print(f"{'='*80}\n")

        return final_prompt


# 全局實例
prompt_builder = PromptBuilder()
