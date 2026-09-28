# SentimentScope

SentimentScope is a PyTorch sentiment-analysis project that classifies IMDB movie reviews as positive or negative using a custom transformer architecture and the `bert-base-uncased` tokenizer.

## Results

- Validation accuracy: **79.96%**
- Test accuracy: **79.12%**
- Required test accuracy: **greater than 75%**
- Test set size: **25,000 reviews**

## Repository contents

- `SentimentScope.ipynb` — fully executed project notebook, including data loading, exploratory analysis, visualizations, tokenization, the custom PyTorch dataset, transformer architecture, training loop, evaluation, and conclusions.
- `SentimentScope_DemoGPT_checkpoint.pt` — trained model checkpoint containing the state dictionary, configuration, tokenizer identifier, seed, and measured accuracy.
- `verification.json` — concise verification results.

## Running the notebook

1. Download and extract the [Stanford IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz).
2. Place the extracted `aclImdb` directory beside the notebook.
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Open and run `SentimentScope.ipynb`.

The notebook uses stratified training and validation splits, fixed random seeds, and includes assertions for dataset dimensions, tensor types, model output shape, and the required accuracy threshold.

