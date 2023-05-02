# grindGPT
"autotune" for prompt training and refinement


This code prompts the user to repeat or refine the generated text for each input chunk, and saves the annotated documents to a JSON file specified by the `filename` command-line argument. The `annotate` function adds annotations to each document indicating which tokens are inputs and outputs, and whether the document is a repeat or refinement. The `save_documents` function writes the annotated documents to the file in a human-readable format. Note that this implementation assumes that the input document has already been preprocessed and split into chunks of maximum length `CHUNK_SIZE`.


repeat.py asks the LLM to repeat some of the things it has been taught and verifies that the output conforms to the original prompt. If the output doesn't conform, it raises an error.

refine.py takes a piece of generated text as input and allows the user to modify it. The modified text is then sent back to the LLM for further refinement.

Note that the refine function takes a piece of generated text as input, allows the user to modify it, concatenates the modified text with the original input, and sends the modified input back to the OpenAI API for further refinement. The refined output is then printed to the console.

python main.py my_large_text_file.txt

This would break the contents of my_large_text_file.txt into chunks of 512 tokens (or fewer for the final chunk), and send each chunk to the GPT-3 API for processing using the specified parameters. The generated text for each input chunk would be printed to the console as it is received.

You can also specify the GPT-3 model to use, the temperature to use for generating output, and the maximum number of tokens to generate per input using the optional command-line arguments --model, --temperature, and --max-tokens, respectively.
