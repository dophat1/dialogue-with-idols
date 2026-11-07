import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from .env")

    model_name = os.getenv("OPENAI_MODEL")

    if not model_name:
        raise RuntimeError("OPENAI_MODEL is missing from .env")

    client = OpenAI(
        timeout=30.0,
        max_retries=0,
    )

    response = client.responses.create(
        model=model_name,
        input="Reply with exactly: Connection successful",
        max_output_tokens=50,
        store=False,
    )

    print(response.output_text)

    if response.usage:
        print(f"Input tokens: {response.usage.input_tokens}")
        print(f"Output tokens: {response.usage.output_tokens}")


if __name__ == "__main__":
    main()