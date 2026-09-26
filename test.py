import os
import pathlib
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in your .env file.")

client = genai.Client(api_key=api_key)

MODELS = [
    "gemini-3.8-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    
]

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
MAX_PROCESSING_WAIT_SECONDS = 120
SEPARATOR_WIDTH = 60

SUMMARY_PROMPT = """
Summarize this PDF clearly and concisely.

Requirements:
- Identify the main purpose of the document.
- Explain the major ideas.
- Preserve important facts and conclusions.
- Include important numbers, findings, or examples when relevant.
- Remove unnecessary repetition.
- Organize the summary logically.
- Do not invent information that is not present in the PDF.
- Make the summary easy to understand.
"""


def sanitize_path(pdf_path: str) -> str:
    resolved = pathlib.Path(pdf_path).resolve()
    allowed_directory = pathlib.Path(".").resolve()

    if not str(resolved).startswith(str(allowed_directory)):
        raise ValueError("Access to this file path is not permitted.")

    return str(resolved)


def upload_pdf(pdf_path: str):
    pdf_path = sanitize_path(pdf_path)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"File is not a PDF: {pdf_path}")

    try:
        print("Uploading PDF...")
        pdf_file = client.files.upload(file=pdf_path)
        print("PDF uploaded.")
    except Exception as error:
        raise RuntimeError(f"Failed to upload PDF: {error}") from error

    elapsed = 0
    poll_interval = 2

    while pdf_file.state and pdf_file.state.name == "PROCESSING":
        if elapsed >= MAX_PROCESSING_WAIT_SECONDS:
            raise TimeoutError("PDF processing exceeded maximum wait time.")

        print("Gemini is processing the PDF...")
        time.sleep(poll_interval)
        elapsed += poll_interval
        poll_interval = min(poll_interval * 2, 30)
        pdf_file = client.files.get(name=pdf_file.name)

    if pdf_file.state and pdf_file.state.name == "FAILED":
        raise RuntimeError("Gemini failed to process the uploaded PDF.")

    print("PDF is ready.")
    return pdf_file


def summarize_pdf(pdf_file) -> str:
    last_error = None

    for model in MODELS:
        for attempt in range(MAX_RETRIES):
            try:
                print(f"Model: {model} | Attempt: {attempt + 1}")

                response = client.models.generate_content(
                    model=model,
                    contents=[SUMMARY_PROMPT, pdf_file]
                )

                print(f"Summary generated using {model}.")
                return response.text

            except Exception as error:
                last_error = error
                print(f"Model {model}, attempt {attempt + 1} failed: {error}")

                if attempt < MAX_RETRIES - 1:
                    print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(RETRY_DELAY_SECONDS)

        print(f"All attempts exhausted for model: {model}. Trying next model.")

    raise RuntimeError(
        f"All models failed to generate a summary. Last error: {last_error}"
    ) from last_error


def display_summary(summary: str) -> None:
    separator = "=" * SEPARATOR_WIDTH
    print(f"\n{separator}\nSUMMARY\n{separator}")
    print(summary)


def main():
    pdf_path = "test.pdf"
    pdf_file = upload_pdf(pdf_path)
    summary = summarize_pdf(pdf_file)
    display_summary(summary)


if __name__ == "__main__":
    main()