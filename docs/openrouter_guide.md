# OpenRouter Integration Guide

This project now supports multiple LLM providers through [OpenRouter](https://openrouter.ai/), giving you access to hundreds of AI models through a single API.

## Quick Start

### 1. Get an OpenRouter API Key

1. Go to [OpenRouter](https://openrouter.ai/)
2. Sign up for an account
3. Go to [API Keys](https://openrouter.ai/keys) and create a new key
4. Copy your API key

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key
USE_OPENROUTER=true
OPENROUTER_API_KEY=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
MODEL_NAME=openai/gpt-4o
```

### 3. Test Different Models

```bash
# Test with Claude 3.5 Sonnet
python scripts/run_extraction.py --model claude-3.5-sonnet

# Test with GPT-4o  
python scripts/run_extraction.py --model gpt-4o

# Test with Gemini Pro
python scripts/run_extraction.py --model gemini-pro

# List all available models
python scripts/run_extraction.py --list-models
```

## Advanced Usage

### Direct Model Testing

```bash
# Use the main extraction script with specific models
python src/llm_text_extract.py --model anthropic/claude-3.5-sonnet
python src/llm_text_extract.py --model google/gemini-pro
python src/llm_text_extract.py --model meta-llama/llama-3.1-70b-instruct
```

### Model Comparison

```bash
# Compare multiple models on the same PDF
python model_comparison.py quick

# Compare specific models
python model_comparison.py compare --models gpt-4o claude-3.5-sonnet gemini-pro

# Full comparison with custom PDF
python model_comparison.py compare --pdf-path path/to/your.pdf --models gpt-4o claude-3.5-sonnet
```

## Popular Models Available

| Shortcut | Full Model ID | Provider |
|----------|---------------|----------|
| `gpt-4o` | `openai/gpt-4o` | OpenAI |
| `gpt-5` | `openai/gpt-5` | OpenAI |
| `claude-3.5-sonnet` | `anthropic/claude-3.5-sonnet` | Anthropic |
| `claude-3.5-haiku` | `anthropic/claude-3.5-haiku` | Anthropic |
| `gemini-pro` | `google/gemini-pro` | Google |
| `llama-3.1-70b` | `meta-llama/llama-3.1-70b-instruct` | Meta |
| `llama-3.1-405b` | `meta-llama/llama-3.1-405b-instruct` | Meta |
| `mistral-large` | `mistralai/mistral-large` | Mistral |
| `deepseek-chat` | `deepseek/deepseek-chat` | DeepSeek |

See the full list at [OpenRouter Models](https://openrouter.ai/models)

## Cost Considerations

Different models have different pricing. Some key points:

- **Free Models**: Some models offer free tiers (check OpenRouter for current offers)
- **Cost-Effective**: Models like `openai/gpt-4o-mini` and `anthropic/claude-3.5-haiku` 
- **Premium**: `openai/gpt-5` and `anthropic/claude-3-opus` are more expensive but higher quality
- **Open Source**: Many Llama and Mistral models are very affordable

Check current pricing at [OpenRouter Models](https://openrouter.ai/models)

## Logging and Tracking

All model interactions are automatically logged with:

- Model name and provider
- Token usage (when available)
- Response time
- Accuracy metrics against baseline
- Prompt version used

View logs in:
- SQLite database: `logs/prompts/prompt_logs.db`
- Accuracy metrics: `logs/accuracy/`

## Integration with Existing Workflow

The OpenRouter integration is fully compatible with existing features:

### Prompt Engineering
```bash
# Use different models with same prompt version
python src/llm_text_extract.py --prompt-version v1.1.2 --model claude-3.5-sonnet
python src/llm_text_extract.py --prompt-version v1.1.2 --model gpt-4o
```

### Accuracy Tracking
```bash
# All models get the same accuracy evaluation
./backfill-accuracy  # Will include OpenRouter results
```

### Comparison Analysis
```bash
# Compare model performance in analysis notebooks
# exploration/discrepancy_insights.qmd now includes model info
```

## Troubleshooting

### Common Issues

1. **API Key Not Working**
   ```bash
   # Check your .env file
   cat .env | grep OPENROUTER_API_KEY
   
   # Verify the key works
   curl -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/models
   ```

2. **Model Not Found**
   ```bash
   # List available models
   python scripts/run_extraction.py --list-models
   
   # Check OpenRouter directly
   curl https://openrouter.ai/api/v1/models
   ```

3. **Rate Limits**
   - OpenRouter has different rate limits per model
   - Check your usage at [OpenRouter Dashboard](https://openrouter.ai/activity)
   - Consider using multiple API keys for higher throughput

### Getting Help

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [OpenRouter Discord](https://discord.gg/fVyRaUDgxW)
- Project issues: File an issue in this repository

## Model Performance Comparison

Once you've tested multiple models, you can compare their performance:

```python
# In Python
from src.reporting import get_model_comparison_summary
summary = get_model_comparison_summary()
print(summary)
```

The system tracks accuracy, speed, and cost for each model, helping you choose the best option for your use case.
