# SentimentScope

SentimentScope is a PyTorch sentiment-analysis project that classifies IMDB movie reviews as positive or negative using a custom transformer architecture and the `bert-base-uncased` tokenizer.

## Results

- Validation accuracy: **81.44%**
- Test accuracy: **79.52%**
- Required test accuracy: **greater than 75%**
- Test set size: **25,000 reviews**

## Repository contents

- `SentimentScope.ipynb` — fully executed project notebook, including data loading, exploratory analysis, visualizations, tokenization, the custom PyTorch dataset, transformer architecture, training loop, evaluation, and conclusions.
- `SentimentScope_DemoGPT_checkpoint.pt` — trained model checkpoint containing the state dictionary, configuration, tokenizer identifier, seed, and measured accuracy.
- `verification.json` — concise verification results.
- `EXECUTION_RESULTS.md` — assertion results, descriptive statistics, rendered visualizations, and execution evidence.

## Running the notebook

1. Download and extract the [Stanford IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz).
2. Place the extracted `aclImdb` directory beside the notebook.
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Open and run `SentimentScope.ipynb`.

The notebook follows the starter-specific 22,500/2,500/25,000 train-validation-test split, uses fixed random seeds, and retains the required assertions for DataFrame dimensions, Dataset lengths and types, `DemoGPT` output shape, and the accuracy threshold.
