"""
Prompt Builder - Assemble the final System Prompt
Build system prompt based on behavior_specs, character info, and conversation history
"""


class PromptBuilder:
    """System Prompt assembly tool"""

    @staticmethod
    def build_system_prompt(behavior_specs: dict, character_info: dict, conversation_history: list = None, rag_context: dict = None) -> str:
        """
        Assemble the final system prompt

        Parameters:
            behavior_specs: dict, loaded from behavior_specs JSON
                Contains: Mission, Core Principles, Conversation Policies,
                         Generation Policies, Writing Style, Response Format
            character_info: dict, character basic information
                {name: str, gender: str, tags: list}
            conversation_history: list, current unsummarized conversation history (optional, assembled into System Prompt)
                [{role: str, text: str}, ...]
            rag_context: dict, RAG retrieval results (optional, contains most relevant background, examples, summaries)
                {character_background: str, fewshots: list, summaries: list}

        Returns:
            str, final system prompt
        """

        # 1. Extract each part of behavior_specs
        mission = behavior_specs.get("Mission", "")
        core_principles = behavior_specs.get("Core Principles", [])
        conversation_policies = behavior_specs.get("Conversation Policies", [])
        generation_policies = behavior_specs.get("Generation Policies", [])
        writing_style = behavior_specs.get("Writing Style", [])
        response_format = behavior_specs.get("Response Format", {})

        # 2. Extract character information
        char_name = character_info.get("name", "Undefined")
        char_gender = character_info.get("gender", "Unknown")
        char_tags = character_info.get("tags", [])
        tag_str = ", ".join(char_tags) if char_tags else "None"

        # 3. Build Core Principles section
        principles_str = ""
        if core_principles:
            principles_str = "\n## Core Principles\n"
            for principle in core_principles:
                title = principle.get("title", "")
                description = principle.get("description", "")
                principles_str += f"- **{title}**: {description}\n"

        # 5. Build Conversation Policies section
        conversation_policies_str = ""
        if conversation_policies:
            conversation_policies_str = "\n## Conversation Policies\n"
            for policy in conversation_policies:
                conversation_policies_str += f"- {policy}\n"

        # 6. Build Generation Policies section
        generation_policies_str = ""
        if generation_policies:
            generation_policies_str = "\n## Generation Policies\n"
            for policy in generation_policies:
                generation_policies_str += f"- {policy}\n"

        # 7. Build Writing Style section
        writing_style_str = ""
        if writing_style:
            writing_style_str = "\n## Writing Style\n"
            for style in writing_style:
                writing_style_str += f"- {style}\n"

        # 8. Build Response Format section
        response_format_str = ""
        if response_format:
            rules = response_format.get("rules", [])
            if rules:
                response_format_str = "\n## Response Format\n"
                for rule in rules:
                    response_format_str += f"- {rule}\n"

        # 9. Build character information section
        character_info_str = f"""
## Character Information
- **Name**: {char_name}
- **Gender**: {char_gender}
- **Core Traits**: [{tag_str}]"""

        # 10. Build RAG context section
        rag_context_str = ""
        if rag_context:
            rag_context_str = "\n## Character Background and Examples"

            # Add most relevant character background
            if rag_context.get("character_background"):
                rag_context_str += f"\n### Most Relevant Background Information\n{rag_context['character_background']}"

            # Add most relevant dialogue examples
            if rag_context.get("fewshots"):
                rag_context_str += "\n### Most Relevant Dialogue Examples"
                for idx, fewshot in enumerate(rag_context["fewshots"], 1):
                    rag_context_str += f"\n**Most Relevant Example {idx}:**\n{fewshot}"

            # Add most relevant history summary (placed at the end)
            if rag_context.get("summaries"):
                rag_context_str += "\n### Most Relevant History Summary"
                for idx, summary in enumerate(rag_context["summaries"], 1):
                    rag_context_str += f"\n**Most Relevant Summary {idx}:**\n{summary}"

        # 11. Build current conversation context section
        conversation_context_str = ""
        if conversation_history:
            conversation_context_str = "\n## Current Conversation Context"
            for msg in conversation_history:
                role_label = "User" if msg.get("role") == "user" else "Character"
                conversation_context_str += f"\n【{role_label}】\n{msg.get('text', '')}"

        # 12. Assemble the final system prompt
        prompt = f"""# System Roleplay Instructions

## Mission
{mission}

{principles_str}{conversation_policies_str}{generation_policies_str}{writing_style_str}{response_format_str}{character_info_str}{rag_context_str}{conversation_context_str}
"""

        final_prompt = prompt.strip()
        print(f"\n🔧 [prompt_builder] Final System Prompt:\n{'='*80}")
        print(final_prompt)
        print(f"{'='*80}\n")

        return final_prompt


# Global instance
prompt_builder = PromptBuilder()
