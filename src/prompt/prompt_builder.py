"""
Prompt Builder - Assemble the final System Prompt
Build system prompt based on behavior_specs, character info, and RAG context

🆕 v2 structure (matches behavior_specs v2 schema):
    - Core Principles / Roleplay Rules are plain string arrays
    - Examples are included in the prompt (Output Examples section)
    - Character section uses fixed-index personality retrieval (character_personality)
    - RAG sections retrieved with the current conversation as the query are titled
      "... Most Relevant to the Current Conversation"
    - Conversation history is NOT assembled here — it is sent natively as chat
      messages (role: user/assistant) by chat_service
"""


class PromptBuilder:
    """System Prompt assembly tool"""

    @staticmethod
    def build_system_prompt(behavior_specs: dict, character_info: dict, conversation_history: list = None, rag_context: dict = None, protagonist_name: str = None) -> str:
        """
        Assemble the final system prompt

        Parameters:
            behavior_specs: dict, loaded from behavior_specs JSON (v2 schema)
                Contains: Mission, Core Principles (list[str]), Roleplay Rules (list[str]),
                          Writing Style (list[str]), Response Format ({rules: list[str]}),
                          Examples (list[str])
            character_info: dict, character basic information
                {name: str, gender: str, tags: list}
            conversation_history: (unused) history is sent as native chat messages
            rag_context: dict, RAG retrieval results (optional)
                {character_background: str, fewshots: list, summaries: list,
                 protagonist_background: str, character_personality: list}
            protagonist_name: str, name of the protagonist (the user's character), optional

        Returns:
            str, final system prompt
        """

        # 1. Extract each part of behavior_specs (v2 schema: plain string arrays)
        mission = behavior_specs.get("Mission", "")
        core_principles = behavior_specs.get("Core Principles", [])
        roleplay_rules = behavior_specs.get("Roleplay Rules", [])
        writing_style = behavior_specs.get("Writing Style", [])
        response_format = behavior_specs.get("Response Format", {})
        examples = behavior_specs.get("Examples", [])

        # 2. Extract character information
        char_name = character_info.get("name", "Undefined")
        char_gender = character_info.get("gender", "Unknown")
        char_tags = character_info.get("tags", [])
        tag_str = ", ".join(char_tags) if char_tags else "None"

        # 3. Core Principles (plain strings)
        principles_str = ""
        if core_principles:
            principles_str = "\n## Core Principles\n"
            for principle in core_principles:
                principles_str += f"- {principle}\n"

        # 4. Roleplay Rules
        roleplay_rules_str = ""
        if roleplay_rules:
            roleplay_rules_str = "\n## Roleplay Rules\n"
            for rule in roleplay_rules:
                roleplay_rules_str += f"- {rule}\n"

        # 5. Writing Style
        writing_style_str = ""
        if writing_style:
            writing_style_str = "\n## Writing Style\n"
            for style in writing_style:
                writing_style_str += f"- {style}\n"

        # 6. Response Format
        response_format_str = ""
        if response_format:
            rules = response_format.get("rules", [])
            if rules:
                response_format_str = "\n## Response Format\n"
                for rule in rules:
                    response_format_str += f"- {rule}\n"

        # 7. 🆕 Output Examples (plain strings, demonstrate narrative + dialogue format)
        examples_str = ""
        if examples:
            examples_str = "\n## Output Examples\n"
            for idx, example in enumerate(examples, 1):
                examples_str += f"\n**Example {idx}:**\n{example}\n"

        # 8. 🆕 The character you primarily play
        #    Personality description comes from fixed-index RAG retrieval
        #    (character_personality: queried with a fixed semantic keyword, not the current conversation)
        character_str = f"""
## The Character You Primarily Play
- **Name**: {char_name}
- **Gender**: {char_gender}
- **Core Traits**: [{tag_str}]"""

        personality = rag_context.get("character_personality") if rag_context else None
        if personality:
            character_str += "\n### You must portray the character according to this personality description"
            for p in personality:
                character_str += f"\n- {p}"

        # 8.5 🆕 背景與對話範例緊接在性格描述之後（同屬「你主要扮演的角色」段落，無獨立 ## 標題）
        if rag_context:
            # Most relevant character background（列表，逐條列出）
            if rag_context.get("character_background"):
                character_str += "\n### Background Information Most Relevant to the Current Conversation"
                for bg in rag_context["character_background"]:
                    character_str += f"\n- {bg}"

            # Most relevant dialogue examples
            if rag_context.get("fewshots"):
                character_str += "\n### Dialogue Examples Most Relevant to the Current Conversation"
                for idx, fewshot in enumerate(rag_context["fewshots"], 1):
                    character_str += f"\n**Example {idx}:**\n{fewshot}"

        # 9. 🆕 The protagonist (played by the user)
        protagonist_str = ""
        protagonist_background = rag_context.get("protagonist_background") if rag_context else None
        if protagonist_name or protagonist_background:
            protagonist_str = "\n\n## The Protagonist (played by the user)"
            if protagonist_name:
                protagonist_str += f"\n- **Name**: {protagonist_name}"
            if protagonist_background:
                # 列表，逐條列出
                protagonist_str += "\n### Protagonist Background Most Relevant to the Current Conversation"
                for pb in protagonist_background:
                    protagonist_str += f"\n- {pb}"

        # 10. History summaries (placed at the end, standalone section)
        rag_context_str = ""
        if rag_context and rag_context.get("summaries"):
            rag_context_str = "\n\n## History Summaries Most Relevant to the Current Conversation"
            for idx, summary in enumerate(rag_context["summaries"], 1):
                rag_context_str += f"\n**Summary {idx}:**\n{summary}"

        # 11. Assemble the final system prompt
        # (conversation history is sent natively as chat messages — not assembled here)
        prompt = f"""# System Roleplay Instructions

## Mission
{mission}

{principles_str}{roleplay_rules_str}{writing_style_str}{response_format_str}{examples_str}{character_str}{protagonist_str}{rag_context_str}
"""

        final_prompt = prompt.strip()
        print(f"\n🔧 [prompt_builder] Final System Prompt:\n{'='*80}")
        print(final_prompt)
        print(f"{'='*80}\n")

        return final_prompt


# 全局實例
prompt_builder = PromptBuilder()
