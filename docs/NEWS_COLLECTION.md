# Resumable news collection

`collect_news_prd.py` fills gaps in historical news coverage using the Alpha Vantage `NEWS_SENTIMENT` endpoint. It is conservative by design. It works within a small request budget, never overwrites existing data, and never marks a session as having had zero news.

## Usage

The default mode only prints the collection plan and writes a local audit summary. It makes **no API calls**:

```bash
python -m unittest discover -s tests/news_collection -p test_collect_news_prd.py -v
python collect_news_prd.py --tickers MSFT NVDA
```

Start with a small probe that checks access and coverage:

```bash
python -u collect_news_prd.py --tickers MSFT NVDA --fetch --max-requests 3
```

Then collect bounded batches for all configured stocks. Running the command again resumes where it stopped:

```bash
python -u collect_news_prd.py --fetch --max-requests 10
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--tickers` | all in `config.yaml` | Stocks to plan or collect |
| `--fetch` | off | Actually make API requests |
| `--max-requests` | 3 | Requests per invocation |
| `--daily-budget` | 25 | Local rolling 24-hour ceiling; must fit your account quota |

### API key

The collector reads `ALPHAVANTAGE_API_KEY` from the environment. Copy [`.env.example`](../.env.example) for the format, or run:

```bash
export ALPHAVANTAGE_API_KEY=your-api-key
```

If the variable is not set, it asks for the key with hidden input. The key is never written to the audit files, and errors containing it are redacted.

## Collection behaviour

- **Non-destructive.** Existing raw files, trained models and processed datasets are preserved.
- **Legacy files.** Non-empty legacy files with fewer than 1,000 rows are kept and their intervals are skipped. They remain `legacy_unverified`, not certified complete. Empty legacy files, and files with 1,000 or more rows, are not accepted as complete intervals.
- **Fair queue.** Stocks alternate in the request queue, so one stock cannot use up the whole batch.
- **Storage.** New articles are saved under `data/raw/news/TICKER/` with a `prd_` filename prefix. Raw JSON responses and checksums go to `reports/prd/news_collection/responses/`. `collection_status.json` records intervals, counts, timestamps and stop reasons.
- **Cap handling.** A feed that hits the 1,000-item limit is split into smaller date intervals. A capped single-day feed stops for review instead of being accepted truncated.
- **Time windows.** Request end times extend to the next UTC midnight. Saved articles use half-open intervals, so the final minute of a day is not lost.
- **Fail closed.** Malformed responses, out-of-range timestamps, HTTP errors and API rejections stop collection and never create no-news markers. A valid empty feed is recorded as `empty_unverified`, because a successful query does not prove the provider's history is complete.
- **Resumable.** Finished work is reused after checksum verification. An interrupted request may be repeated and is counted against the local budget.

## Request limits

The defaults are at most 3 requests per invocation, 15 seconds between requests, and 25 recorded requests in any rolling 24 hours. Calls from other scripts or machines are invisible to this counter, so the provider may stop you earlier. An error also uses one recorded request. Do not run another collector at the same time. The script exits when its batch or budget is used up. It does not wait or schedule itself.

## Next steps after collection

Collection is only the first stage of a sentiment experiment. New articles still need to be combined, cleaned, scored with FinBERT and VADER, and checked for eligible chronological windows before `python -m src.prd sentiment` can use them. No guarantee is made that the provider can supply the entire required history.

References: [NEWS_SENTIMENT parameters and 1,000-item limit](https://www.alphavantage.co/documentation/#news-sentiment), [usage limits](https://www.alphavantage.co/premium/).
