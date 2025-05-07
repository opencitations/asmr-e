import os
import re
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import asyncio
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
import spacy
from jinja2 import Template


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_trf")  # Use "en_core_web_sm" if you have less resources
SEM_LIMIT = 3  # Limit concurrency to avoid overwhelming LLMs when running in local
semaphore = asyncio.Semaphore(SEM_LIMIT)

#Prompt template is slightly different because of a suggested oprimization for this ollama implementation
PROMPT_TEMPLATE = Template(
    """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are Professor John, an expert in entity extraction, tasked with identifying mentions of software, URLs, and programming languages in academic research text. Follow the descriptions and examples below for guidance on each category:\n

 **To extract entities correctly, follow these steps**:\n
        1. **Identify and extract the software name**: Look for specific software tools, platforms, or programs explicitly mentioned in the text. Make sure the software name is stated clearly and explicitly, not as part of a generic term like 'tool' or 'platform'.\n
        2. **Check if a version is mentioned**: After identifying the software, check if a version number or specific edition is mentioned. If no version is mentioned, leave the field empty.\n
        3. **Check if a URL is associated with the software**: Look for any URL that is directly associated with the software mentioned in the text. If a URL is explicitly given, include it in the output.\n
        4. **Identify any programming languages mentioned**: Check if any programming language is mentioned, especially if it is related to the software or research method. If found, include the language and its version if available.\n
        5. **Identify the Publisher**: For each software entity extracted check if there is a publisher associated to it.\n
        
        **Rules**:\n
        1. Extract only explicitly named software, URLs, and programming languages.\n
        2. URLs should only be extracted if they are directly related to a software. If a software is mentioned, include its associated URL if available.\n
        3. Programming languages should only be included if they are related to the software. For example, if `SciPy` is mentioned with `Python`, include Python under the `"language"` field for SciPy.\n
        4. Publishers should be extracted only if directly stated in the text, not desumed from your knowledge.
        4. Respond strictly in JSON array format and **nothing else**.\n
        5. If you don't find informations to fill the fields in the JSON array you must reproduce the structure as it is leaving the fields blank: []
        6. DO NOT include your thinking process in your answer.\n
        
        **Examples**:\n
        Format each identified entity as follows:\n
        Correct output:\n

Text:
{{ chunk }}

Return JSON like:
        [
        {
            "software": "<software_name>",
            "version": ["<software_version>"], 
            "publisher": ["<software_publisher>"],
            "url": ["<software_url>"], 
            "language": ["<software_language>"]
        },
        ]
            Incorrect output:\n

        Any introductory text, explanations, or lists outside the JSON array.\n
        Any response not formatted as JSON.\n

        When there are no entities to extract, respond only with: [].\n

        Return only the JSON array of identified entities and no other text. Follow the format precisely, or the response will be invalid.\n
        DO NOT USE EXAMPLES PROVIDED AS OUTPUT. STICK TO OUTPUT FORMAT. DO NOT ASSUME INFORMATION THAT AREN'T PRESENT IN THE TEXT.
        """
)

def segment_text_with_overlap(text: str, window_size: int, overlap_sentences: int) -> List[str]:
    """Split text into chunks of sentences with overlap."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    chunks = []
    start = 0
    while start < len(sentences):
        end = start + window_size
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        start += window_size - overlap_sentences
    return chunks


async def process_chunk(llm: Ollama, chunk: str, index: int) -> Dict[int, List[Dict]]:
    """Process a single chunk with the LLM."""
    try:
        full_prompt = PROMPT_TEMPLATE.render(chunk=chunk)

        logger.debug(f"Sending prompt for chunk {index}:\n{full_prompt[:500]}...")

        response = await llm.agenerate([full_prompt])

        raw_response = response.generations[0][0].text.strip()
        logger.debug(f"Raw LLM response for chunk {index}:\n{raw_response}")

        extracted = extract_json_from_response(raw_response)
        return {index: extracted}
    except Exception as e:
        logger.error(f"Error processing chunk {index}: {e}")
        return {index: []}


async def process_pdf(pdf_path: str, model_name: str, window_size: int, overlap_sentences: int):
    """Main function to process PDF and extract entities."""
    logger.info(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = " ".join(page.page_content for page in pages)

    logger.info("Segmenting text...")
    chunks = segment_text_with_overlap(full_text, window_size, overlap_sentences)
    logger.info(f"Split into {len(chunks)} chunks.")

    logger.info(f"Initializing Ollama model: {model_name}")
    llm = Ollama(model=model_name, temperature=0.2, top_p=0.9, num_predict=1024)

    logger.info("Processing chunks...")
    tasks = [process_chunk(llm, chunk, i) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)

    final_entities = []
    for result in results:
        for _, entities in result.items():
            final_entities.extend(entities)

    logger.info(f"Extracted {len(final_entities)} entities.")
    return (
        final_entities,
        model_name,
        window_size,
        overlap_sentences,
    )


def extract_json_from_response(response_text: str) -> List[Dict]:
    """Attempt to extract JSON from LLM response."""
    try:
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1
        if json_start != -1 and json_end != -1:
            json_text = response_text[json_start:json_end]
            return json.loads(json_text)
        else:
            logger.warning("No valid JSON found.")
            return []
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        return []


def generate_output(results, output_dir, start_time, end_time, args):
    os.makedirs(output_dir, exist_ok=True)

    def sanitize_filename(name: str) -> str:
        return re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = sanitize_filename(f"{args.model}")
    base_filename = f"{Path(args.pdf).stem}_{safe_model_name}_win{args.window}_ov{args.overlap}_{timestamp}"

    json_file = os.path.join(output_dir, f"{base_filename}.json")
    csv_file = os.path.join(output_dir, f"{base_filename}.csv")

    metadata = {
        "input_pdf": args.pdf,
        "model": args.model,
        "window_size": args.window,
        "overlap_sentences": args.overlap,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_time": str(end_time - start_time),
        "total_entities": len(results)
    }

    full_output = {
        "metadata": metadata,
        "entities": results
    }

    # Save as JSON
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(full_output, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON output to {json_file}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")

    if results:
        try:
            df = pd.DataFrame(results)
            df['input_pdf'] = args.pdf
            df['model'] = args.model
            df['window_size'] = args.window
            df['overlap_sentences'] = args.overlap
            df['extraction_time'] = str(end_time - start_time)
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved CSV output to {csv_file}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
    else:
        logger.warning("No entities found. CSV not generated.")

    return full_output

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Extract entities from academic PDFs using Ollama.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF file")
    parser.add_argument("--model", type=str, default="llama3", help="Ollama model name (e.g., llama3)")
    parser.add_argument("--window", type=int, default=10, help="Number of sentences per chunk")
    parser.add_argument("--overlap", type=int, default=2, help="Overlap between chunks")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files")

    args = parser.parse_args()

    start_time = datetime.now()
    loop = asyncio.new_event_loop()
    entities, model, window, overlap = loop.run_until_complete(
        process_pdf(args.pdf, args.model, args.window, overlap_sentences=args.overlap)
    )
    end_time = datetime.now()

    print("\nExtracted Entities:")
    for entity in entities:
        print(entity)

    print(f"\nTotal time: {end_time - start_time}")
    print(f"Model: {model}, Window: {window}, Overlap: {overlap}")

    generate_output(entities, args.output_dir, start_time, end_time, args)