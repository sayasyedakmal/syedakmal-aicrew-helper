import os
import json
import yaml
from datetime import datetime
import time

import google.generativeai as genai

# ---- role-specific prompts ----
ROLE_PROMPTS = {
    "highplanner": """
You are the **High-Level Planner**.
- Break down the project vision into milestones and phases.
- Focus on scope, dependencies, risks.
- Output strictly valid JSON: { "milestones": [...], "dependencies": [...], "risks": [...] }
""",
    "featureplanner": """
You are the **Feature Planner**.
- Take one milestone and break it into features and tasks.
- Be concrete but not yet code-level.
- Output strictly valid JSON: { "features": [...], "tasks": [...] }
""",
    "architect": """
You are the **Software Architect**.
- Design APIs, database schema, and system components.
- Make explicit ADRs (Architecture Decision Records).
- Output strictly valid JSON: { "apis": [...], "db_schema": {...}, "adrs": [...] }
""",
    "implementer": """
You are the **Implementer**.
- Write production-ready code for the design.
- Include explanations only in comments inside the code.
- Output strictly valid JSON: { "files": [{ "path": "file.py", "code": "..." }] }
""",
    "reviewer": """
You are the **Reviewer**.
- Review the code for correctness, security, and best practices.
- Suggest improvements explicitly.
- Output strictly valid JSON: { "findings": [...], "suggestions": [...] }
"""
}

# ---- load config ----
def load_config():
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("⚠️ config.yaml not found. Using defaults.")
        return {
            "ai_provider": "google",
            "model": "gemini-2.5-flash",
            "max_retries": 3,
            "retry_delay": 2,
            "temperature": 0.7,
            "max_output_tokens": 2048,
            "api_key_env": "GOOGLE_API_KEY",
        }

CONFIG = load_config()

# ---- configure Gemini ----
api_key = os.getenv(CONFIG.get("api_key_env", "GOOGLE_API_KEY"))
if not api_key:
    raise RuntimeError("❌ No Google API key found. Set it with export GOOGLE_API_KEY=...")

genai.configure(api_key=api_key)
model = genai.GenerativeModel(CONFIG["model"])

# ---- AI call ----
def run_ai(role, ticket_content, extra_context=""):
    role_prompt = ROLE_PROMPTS.get(role, f"You are acting as {role}.")
    prompt = f"""
{role_prompt}

Ticket context:
{ticket_content}

Extra notes / follow-ups:
{extra_context}
"""

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=CONFIG["temperature"],
            max_output_tokens=CONFIG["max_output_tokens"],
        ),
    )

    return {
        "role": role,
        "model": CONFIG["model"],
        "timestamp": datetime.now().isoformat(),
        "ticket_context_preview": ticket_content[:200],
        "extra_context": extra_context,
        "output_raw": response.text.strip(),
    }

# ---- validate JSON ----
def validate_json(text):
    try:
        return json.loads(text)
    except Exception as e:
        return None

# ---- interactive loop ----
def interactive_role(role, ticket_file, artifacts_dir="artifacts"):
    with open(ticket_file, "r") as f:
        ticket_content = f.read()

    extra_context = ""
    history = []

    while True:
        result = None
        for attempt in range(CONFIG["max_retries"]):
            try:
                result = run_ai(role, ticket_content, extra_context)
                break
            except Exception as e:
                print(f"\n❌ AI call failed (attempt {attempt+1}/{CONFIG['max_retries']}): {e}")
                if attempt < CONFIG["max_retries"] - 1:
                    print(f"⏳ Retrying in {CONFIG['retry_delay']}s...")
                    time.sleep(CONFIG["retry_delay"])
                else:
                    choice = input("Retry manually? [y]es / [n]o (quit without saving)\n> ").strip().lower()
                    if choice == "y":
                        continue
                    else:
                        print("🚪 Quit without saving due to repeated errors.")
                        return

        # validate JSON output
        parsed = validate_json(result["output_raw"])
        if not parsed:
            print("\n⚠️ Invalid JSON received from AI.")
            print("Raw output:\n", result["output_raw"])
            choice = input("\nOptions: [r]etry AI  [f]ix manually  [q]uit\n> ").strip().lower()
            if choice == "r":
                extra_context = "⚠️ Your last output was not valid JSON. Please try again with strictly valid JSON."
                continue
            elif choice == "f":
                manual = input("Paste corrected JSON:\n> ")
                try:
                    parsed = json.loads(manual)
                    result["output_json"] = parsed
                except Exception as e:
                    print(f"❌ Still invalid: {e}. Quit without saving.")
                    return
            elif choice == "q":
                print("🚪 Quit without saving due to invalid JSON.")
                return
            else:
                print("Invalid option, quitting.")
                return
        else:
            result["output_json"] = parsed

        history.append(result)

        print("\n--- AI Draft (validated JSON) ---")
        print(json.dumps(result["output_json"], indent=2))

        action = input("\nOptions: [a]ccept  [f]ollow-up  [q]uit without saving\n> ").strip().lower()

        if action == "a":
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            base = os.path.basename(ticket_file).replace(".md", "")

            os.makedirs(artifacts_dir, exist_ok=True)

            final_file = os.path.join(artifacts_dir, f"{base}-{role}-{ts}-final.json")
            with open(final_file, "w") as f:
                json.dump(result["output_json"], f, indent=2)

            history_file = os.path.join(artifacts_dir, f"{base}-{role}-{ts}-history.json")
            with open(history_file, "w") as f:
                json.dump(history, f, indent=2)

            print(f"✅ Accepted.\n   Final saved to {final_file}\n   History saved to {history_file}")
            break
        elif action == "f":
            extra_context = input("Enter your follow-up or clarification:\n> ")
        elif action == "q":
            print("🚪 Quit without saving.")
            break
        else:
            print("Invalid option, try again.")

# ---- main entry ----
def main():
    try:
        with open("status.yaml", "r") as f:
            status = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ status.yaml not found. Make sure you're in the right folder.")
        return

    ticket_file = os.path.join("tickets", f"{status['ticket']}.md")
    if not os.path.exists(ticket_file):
        print(f"❌ Ticket file {ticket_file} not found.")
        return

    print(f"🎯 Current ticket: {status['ticket']} ({ticket_file})")
    print(f"🤖 Using model: {CONFIG['model']} (provider={CONFIG['ai_provider']})")

    role = input("Choose role (highplanner/featureplanner/architect/implementer/reviewer): ").strip().lower()

    if role in ROLE_PROMPTS:
        interactive_role(role, ticket_file)
    else:
        print("❌ Unknown role")

if __name__ == "__main__":
    main()
