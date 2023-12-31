import nltk
import pandas as pd
import torch
import torch.nn.functional as F
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
from transformers import GPT2Tokenizer
from traning_model_with_gpt_2_for_abstract import GPT2Model
from torchtext.data.metrics import bleu_score
from lime.lime_text import LimeTextExplainer
import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


def generate_abstract(model, title, max_length=400, temperature=1.0):
    model.eval()
    with torch.no_grad():
        input_str = f"{title}{tokenizer.eos_token}"
        input_ids = torch.tensor(tokenizer.encode(input_str), dtype=torch.long).unsqueeze(0).to(device)

        for _ in range(max_length):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        generated_abstract = tokenizer.decode(input_ids.squeeze().tolist(), skip_special_tokens=True)
        generated_abstract = generated_abstract[len(title):]

        return generated_abstract.strip()


def evaluate_model(model, test_data):
    bleu_scores = []
    rouge_scores = []
    lime_scores = []
    meteor_scores = []
    explainer = LimeTextExplainer()
    results = []

    rouge = Rouge()

    for data in tqdm(test_data):
        title = data["title"]
        real_abstract = data["abstract"]

        generated_abstract = generate_abstract(model, title, len(real_abstract.split()))

        print("title: ", title)
        print("real abstract: ", real_abstract)
        print("best generated abstract: ", generated_abstract)

        # bleu = bleu_score([generated_abstract.split()], [real_abstract.split()])
        bleu = sentence_bleu([real_abstract.split()], generated_abstract.split(), weights=(1, 0, 0, 0))
        bleu_scores.append(bleu)

        rouge_score = rouge.get_scores(generated_abstract, real_abstract, avg=True)
        rouge_scores.append(rouge_score['rouge-l']['f'])

        # Tokenize hypothesis and reference text
        tokenized_hypothesis = nltk.word_tokenize(generated_abstract)
        tokenized_reference = nltk.word_tokenize(real_abstract)

        # Calculate the METEOR score
        meteor = single_meteor_score(tokenized_reference, tokenized_hypothesis)
        meteor_scores.append(meteor)


        print("bleuscores: ", bleu_scores)
        print("rouge_scores: ", rouge_scores)
        print("meteor_scores: ", meteor_scores)

        results.append({"title": title, "real_abstract": real_abstract, "generated_abstract": generated_abstract,
                        "bleu_score": bleu, "rouge_score": rouge_score['rouge-l']['f'],
                        "meteor_score": meteor})

    return pd.DataFrame(results)


def main():
    model_path = "best_fine_tuned_model_final_cuneyt.pth"
    model = GPT2Model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)

    with open("random_test_data_1000.json", "r") as f:
        test_data = json.load(f)

    results_df = evaluate_model(model, test_data)
    print(results_df)

    avg_bleu_score = results_df["bleu_score"].mean()
    avg_rouge_score = results_df["rouge_score"].mean()
    avg_meteor_score = results_df["meteor_score"].mean()

    print(f"Average BLEU score: {avg_bleu_score}")
    print(f"Average ROUGE-L score: {avg_rouge_score}")
    print(f"Average METEOR score: {avg_meteor_score}")

    # Save results_df to CSV file
    results_df.to_csv("results_from_fine-tuned-gpt2_bu.csv",
                      columns=["title", "real_abstract", "generated_abstract", "bleu_score", "rouge_score",
                               "meteor_score"],
                      index=False)


if __name__ == "__main__":
    main()
