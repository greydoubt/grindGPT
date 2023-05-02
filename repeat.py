import argparse
import openai_secret_manager
import langchain

# Load OpenAI API credentials from environment variables
secrets = openai_secret_manager.get_secret("openai")

# Set up the OpenAI API client
client = openai.SecretManagerClient(api_key=secrets["api_key"])

# Define the maximum length of each input chunk (in tokens)
CHUNK_SIZE = 512

def repeat(prompt, response, model):
    # Create a LangChain instance with the appropriate language model
    lc = langchain.LangChain(
        model_name_or_path="gpt3",
        tokenizers=["text"],
        max_length=CHUNK_SIZE,
        device="cpu",
    )

    # Prepare the prompt using LangChain
    prompt_ids = lc.encode(prompt)

    # Prepare the response using LangChain
    response_ids = lc.encode(response)

    # Concatenate the prompt and response for testing
    test_input = prompt + "\n" + response

    # Send the input to the OpenAI API for testing
    test_response = client.gpt3.generate(
        prompt=test_input,
        model=model,
        max_tokens=1024,
        temperature=0.0,
    )

    # Check that the generated text matches the prompt and response
    test_text = test_response.choices[0].text.strip()
    if test_text != test_input:
        raise ValueError("Output does not match prompt and response.")

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Repeat things the GPT-3 language model has been taught and verify that the output conforms to the original prompt.")
    parser.add_argument("--prompt", required=True, help="prompt to repeat")
    parser.add_argument("--response", required=True, help="response to repeat")
    parser.add_argument("--model", default="text-davinci-002", help="name of the GPT-3 model to use (default: text-davinci-002)")
    args = parser.parse_args()

    # Repeat the prompt and response using the OpenAI API
    repeat(args.prompt, args.response, args.model)

if __name__ == "__main__":
    main()
