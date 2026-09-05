# Wikipedia Ingestion Script

## Overview

`ingest_wikipedia.py` is a script that fetches Wikipedia articles and stores each paragraph as a separate memory in slowave. This allows you to build a knowledge base from Wikipedia content that can be queried and retrieved using slowave's semantic memory system.

## Features

- **Zero external dependencies**: Uses only Python stdlib (`urllib`, `re`) for fetching and parsing
- **Configurable filtering**: Set minimum word count for paragraphs (default: 10 words)
- **Automatic scoping**: Each page is scoped to `wikipedia:<page_title>` by default
- **Custom scoping**: Override the scope to integrate with your own project namespaces
- **Progress tracking**: Verbose mode shows ingestion progress
- **Error handling**: Continues on individual paragraph failures

## Usage

### Basic usage

```bash
python scripts/ingest_wikipedia.py "https://en.wikipedia.org/wiki/Artificial_intelligence"
```

This will:
1. Fetch the Wikipedia page
2. Extract all paragraphs with at least 10 words
3. Store each paragraph as a `fact` in slowave
4. Use scope: `wikipedia:Artificial intelligence`

### Advanced usage

```bash
# Use verbose output to see progress
python scripts/ingest_wikipedia.py --verbose "https://en.wikipedia.org/wiki/Machine_learning"

# Set custom minimum word count
python scripts/ingest_wikipedia.py --min-words 20 "https://en.wikipedia.org/wiki/Deep_learning"

# Use custom scope (e.g., for project knowledge base)
python scripts/ingest_wikipedia.py --scope "project:ai-kb" "https://en.wikipedia.org/wiki/Neural_network"

# Use custom database location
python scripts/ingest_wikipedia.py --db /tmp/my_kb.db "https://en.wikipedia.org/wiki/Python_(programming_language)"

# Combine options
python scripts/ingest_wikipedia.py \
  --scope "project:ml-research" \
  --min-words 15 \
  --verbose \
  "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)"
```

## Command-line options

- `url` (required): Wikipedia article URL
- `--scope SCOPE`: Custom scope for memories (default: `wikipedia:<page_title>`)
- `--min-words N`: Minimum word count for paragraphs (default: 10)
- `--db PATH`: Exact database path override. Without it, the script uses the
  same per-user runtime database as Slowave (`SLOWAVE_HOME` relocates that
  complete runtime tree; legacy `SLOWAVE_DB` is also honored).
- `-v, --verbose`: Enable verbose output with progress

## Examples

### Build a machine learning knowledge base

```bash
# Ingest several ML-related articles into a shared scope
python scripts/ingest_wikipedia.py --scope "project:ml-kb" \
  "https://en.wikipedia.org/wiki/Machine_learning"

python scripts/ingest_wikipedia.py --scope "project:ml-kb" \
  "https://en.wikipedia.org/wiki/Deep_learning"

python scripts/ingest_wikipedia.py --scope "project:ml-kb" \
  "https://en.wikipedia.org/wiki/Neural_network"
```

### Query the ingested content

After ingestion, you can query the content using slowave:

```bash
# Using the CLI
slowave recall "what is machine learning"

# Or in Python
from slowave.core.config import SlowaveConfig
from slowave.core.engine import SlowaveEngine

cfg = SlowaveConfig()  # uses the same resolved per-user runtime database
eng = SlowaveEngine(cfg)
result = eng.recall("neural networks", top_k=5)

for schema in result.schemas:
    print(schema.content_text)
    print(f"Scope: {schema.scope_id}")
    print("---")
```

## How it works

1. **Fetch**: Downloads the Wikipedia page HTML using `urllib`
2. **Parse**: Uses regex to extract content from the main article div (`id="mw-content-text"`)
3. **Extract**: Finds all `<p>` tags and extracts text content
4. **Clean**: Removes HTML tags and decodes HTML entities
5. **Filter**: Only keeps paragraphs with >= `min_words` words
6. **Store**: Each paragraph is stored using `engine.remember()` with:
   - `type="fact"`
   - `scope="wikipedia:<page_title>"` (or custom scope)

## Notes

- The script uses regex-based HTML parsing to avoid external dependencies like BeautifulSoup
- Only paragraph text is extracted; tables, lists, and other structured content are skipped
- Each paragraph becomes a separate memory, allowing fine-grained retrieval
- The scope helps organize memories and can be used for filtering during recall

## Troubleshooting

### No paragraphs extracted

The Wikipedia page structure might be unusual. Try:
- Using a different article
- Lowering `--min-words` to capture shorter paragraphs
- Enabling `--verbose` to see what's being extracted

### HTTP errors

- Check your internet connection
- Verify the URL is valid and accessible
- Wikipedia may rate-limit if you make too many requests quickly

## Integration with slowave ecosystem

The ingested memories work seamlessly with:
- **slowave CLI**: Query using `slowave recall "your query"`
- **slowave MCP server**: Available to Claude, Cline, Cursor, etc.
- **slowave dashboard**: View and explore ingested content
- **Python API**: Programmatic access via `SlowaveEngine`

All memories can be consolidated, generalized, and superseded just like any other slowave memories.
