import argparse
import os
import openai_secret_manager
import langchain
import repeat
import refine

# Load OpenAI API credentials from environment variables
secrets = openai_secret_manager.get_secret("openai")

# Set up the OpenAI API client
client = openai.SecretManagerClient(api_key=secrets["api_key"])

# Define the maximum length of each input chunk (in tokens)
CHUNK_SIZE = 512

def annotate(documents, annotations):
    # Add annotations to each document
    for i, doc in enumerate(documents):
        doc["input_tokens"] = []
        doc["output_tokens"] = []
        doc["is_repeat"] = False
        doc["is_refinement"] = False
        for j, token in enumerate(doc["tokens"]):
            if j in annotations[i]["input_indices"]:
                doc["input_tokens"].append(token)
            elif j in annotations[i]["output_indices"]:
                doc["output_tokens"].append(token)
            if i > 0 and j == 0 and annotations[i]["is_repeat"]:
                doc["is_repeat"] = True
            if i > 0 and j == 0 and annotations[i]["is_refinement"]:
                doc["is_refinement"] = True

def save_documents(documents, filename):
    # Save the annotated documents to a JSON file
    with open(filename, "w") as f:
        for doc in documents:
            f.write(f"{doc['text']}\n")
            f.write(f"Input tokens: {doc['input_tokens']}\n")
            f.write(f"Output tokens: {doc['output_tokens']}\n")
            if doc["is_repeat"]:
                f.write("Is repeat\n")
            elif doc["is_refinement"]:
                f.write("Is refinement\n")
            else:
                f.write("Is original\n")
            f.write("\n")

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Interactively generate and refine text using the OpenAI API.")
    parser.add_argument("filename", help="name of the file to save the annotated documents to")
    parser.add_argument("--model", default="text-davinci-002", help="name of the GPT-3 model to use (default: text-davinci-002)")
    args = parser.parse_args()

    # Load the input document
    with open("input.txt", "r") as f:
        input_text = f.read()

    # Split the input document into chunks of maximum length CHUNK_SIZE
    lc = langchain.LangChain(
        model_name_or_path="gpt3",
        tokenizers=["text"],
        max_length=CHUNK_SIZE,
        device="cpu",
    )
    input_tokens = lc.tokenize(input_text)
    input_chunks = lc.split_into_chunks(input_tokens)

    # Generate and refine text interactively
    documents = []
    annotations = []
    for i, input_chunk in enumerate(input_chunks):
        # Generate text using the OpenAI API
        response = client.gpt3.generate(
            prompt=input_chunk,
            model=args.model,
            max_tokens=1024,
            temperature=0.7,
        )
        generated_text = response.choices[0].text

        # Repeat or refine the generated text using user input
        if i == 0:
            modified_text, is_repeat, is_refinement = repeat.repeat(generated_text)
        else:
            modified_text, is_refinement = refine.refine(generated_text, args.model)
            is_repeat = False

        # Add the modified text to the
    documents.append({
        "text": generated_text,
        "tokens": lc.tokenize(generated_text),
    })
    annotations.append({
        "input_indices": range(len(input_chunk)),
        "output_indices": range(len(input_chunk), len(input_chunk) + len(documents[-1]["tokens"]) - len(input_tokens)),
        "is_repeat": is_repeat,
        "is_refinement": is_refinement,
    })

    # Print the modified text
    print(modified_text)

# Annotate the documents and save them to a file
annotate(documents, annotations)
save_documents(documents, args.filename)
