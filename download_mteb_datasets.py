import argparse
from pathlib import Path
from typing import Optional
from datasets import load_dataset, get_dataset_config_names
from datasets.exceptions import DatasetNotFoundError
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

console = Console()


def get_id_column(dataset) -> str:
    """Get the ID column name from a dataset (either '_id' or 'id')."""
    if "_id" in dataset.column_names:
        return "_id"
    elif "id" in dataset.column_names:
        return "id"
    else:
        raise ValueError(f"No ID column found. Available: {dataset.column_names}")


def detect_dataset_structure(
    dataset_name: str,
) -> tuple[bool, list[str]]:
    """
    Detect whether dataset uses language-prefixed configs (e.g., 'de-corpus').

    Returns:
        Tuple of (is_language_prefixed, available_languages)
    """
    try:
        configs = get_dataset_config_names(dataset_name)
    except Exception:
        return False, []

    # Check for standard configs
    if "corpus" in configs and "queries" in configs:
        return False, []

    # Check for language-prefixed configs (e.g., 'de-corpus', 'en-corpus')
    language_prefixes = set()
    for config in configs:
        if "-corpus" in config:
            lang = config.replace("-corpus", "")
            # Verify this language has all required configs
            if f"{lang}-queries" in configs and f"{lang}-qrels" in configs:
                language_prefixes.add(lang)

    if language_prefixes:
        return True, sorted(language_prefixes)

    return False, []


def download_mteb_dataset(
    dataset_name: str,
    output_dir: Path,
    sample_size: Optional[int] = None,
    language: Optional[str] = None,
) -> None:
    """
    Download an MTEB v2 retrieval dataset from HuggingFace.

    Args:
        dataset_name: Name of the MTEB dataset on HuggingFace (e.g., "mteb/scifact")
        output_dir: Directory to save the dataset files
        sample_size: If specified, only download this many samples (for corpus, queries, and qrels)
        language: For multilingual datasets with language-prefixed configs (e.g., 'de', 'en').
                  If None, auto-detects and prompts or uses first available.
    """
    console.print(f"\n📥 Downloading dataset: [yellow]{dataset_name}[/yellow]")
    if sample_size:
        console.print(f"   Sample size: [cyan]{sample_size}[/cyan] documents")

    # Detect dataset structure (standard vs language-prefixed)
    is_lang_prefixed, available_languages = detect_dataset_structure(dataset_name)

    if is_lang_prefixed:
        console.print(
            f"   🌐 Multilingual dataset detected. Available languages: [cyan]{', '.join(available_languages)}[/cyan]"
        )
        if language:
            if language not in available_languages:
                console.print(
                    f"\n❌ [red]Language '{language}' not available. Choose from: {', '.join(available_languages)}[/red]"
                )
                return
            selected_lang = language
        else:
            # Default to first language if not specified
            selected_lang = available_languages[0]
            console.print(
                f"   💡 No language specified, using: [green]{selected_lang}[/green]"
            )
            console.print("   [dim]Use --language to select a different language[/dim]")

        corpus_config = f"{selected_lang}-corpus"
        queries_config = f"{selected_lang}-queries"
        qrels_config = f"{selected_lang}-qrels"
        console.print(
            f"   📦 Using configs: {corpus_config}, {queries_config}, {qrels_config}"
        )
    else:
        corpus_config = "corpus"
        queries_config = "queries"
        qrels_config = "qrels"
        selected_lang = None

    # Create output directory (include language suffix for multilingual datasets)
    base_name = dataset_name.split("/")[-1]
    if selected_lang:
        dataset_dir = output_dir / f"{base_name}_{selected_lang}"
    else:
        dataset_dir = output_dir / base_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            # Download corpus
            corpus_task = progress.add_task("Downloading corpus...", total=None)
            try:
                corpus_dataset = load_dataset(
                    dataset_name,
                    name=corpus_config,
                )
            except ValueError as e:
                if "BuilderConfig" in str(e) and "not found" in str(e):
                    progress.stop()
                    console.print(
                        f"\n⚠️  [yellow]Dataset '{dataset_name}' is not compatible with MTEB retrieval format.[/yellow]"
                    )
                    console.print(
                        "   This dataset does not have the required 'corpus' and 'queries' configurations."
                    )
                    console.print(
                        "\n   [dim]Only MTEB retrieval datasets are supported (e.g., 'mteb/scifact', 'mteb/nfcorpus').[/dim]"
                    )
                    return
                raise

            # MTEB datasets may have different split names, use the first available split
            if isinstance(corpus_dataset, dict):
                available_splits = list(corpus_dataset.keys())
                corpus_dataset = corpus_dataset[available_splits[0]]
            progress.update(corpus_task, completed=True, total=1)

            # Download queries
            queries_task = progress.add_task("Downloading queries...", total=None)
            queries_dataset = load_dataset(
                dataset_name,
                name=queries_config,
            )
            if isinstance(queries_dataset, dict):
                available_splits = list(queries_dataset.keys())
                queries_dataset = queries_dataset[available_splits[0]]
            progress.update(queries_task, completed=True, total=1)

            # Download qrels (relevance judgments)
            qrels_task = progress.add_task("Downloading qrels...", total=None)
            # Try specified qrels config first, fall back to 'default' for older MTEB datasets
            try:
                qrels_dataset = load_dataset(
                    dataset_name,
                    name=qrels_config,
                )
            except ValueError as e:
                if "BuilderConfig" in str(e) and qrels_config in str(e):
                    # Fall back to 'default' config for older MTEB datasets
                    qrels_dataset = load_dataset(
                        dataset_name,
                        name="default",
                    )
                else:
                    raise
            if isinstance(qrels_dataset, dict):
                available_splits = list(qrels_dataset.keys())
                qrels_dataset = qrels_dataset[available_splits[0]]
            progress.update(qrels_task, completed=True, total=1)

        # Apply sampling if requested
        if sample_size:
            console.print(f"\n✂️  Sampling {sample_size} documents...")

            # Determine ID column names (varies between datasets)
            corpus_id_col = get_id_column(corpus_dataset)
            queries_id_col = get_id_column(queries_dataset)

            # Sample corpus
            num_corpus = len(corpus_dataset)
            if sample_size < num_corpus:
                corpus_dataset = corpus_dataset.select(range(sample_size))
                console.print(
                    f"   • Corpus: {num_corpus} → {len(corpus_dataset)} documents"
                )

            # Filter qrels to only include sampled corpus IDs
            sampled_corpus_ids = set(corpus_dataset[corpus_id_col])

            def filter_qrels(example):
                return example["corpus-id"] in sampled_corpus_ids

            original_qrels_count = len(qrels_dataset)
            qrels_dataset = qrels_dataset.filter(filter_qrels)
            console.print(
                f"   • Qrels: {original_qrels_count} → {len(qrels_dataset)} judgments"
            )

            # Filter queries to only include those with relevance judgments
            sampled_query_ids = set(qrels_dataset["query-id"])

            def filter_queries(example):
                return example[queries_id_col] in sampled_query_ids

            original_queries_count = len(queries_dataset)
            queries_dataset = queries_dataset.filter(filter_queries)
            console.print(
                f"   • Queries: {original_queries_count} → {len(queries_dataset)} queries"
            )

        # Convert to pandas and save as parquet
        console.print("\n💾 Saving to parquet files...")

        corpus_df = corpus_dataset.to_pandas()
        queries_df = queries_dataset.to_pandas()
        qrels_df = qrels_dataset.to_pandas()

        # Rename _id to id for consistency with MTEB format
        if "_id" in corpus_df.columns:
            corpus_df = corpus_df.rename(columns={"_id": "id"})
        if "_id" in queries_df.columns:
            queries_df = queries_df.rename(columns={"_id": "id"})

        # Save files
        corpus_path = dataset_dir / "corpus.parquet"
        queries_path = dataset_dir / "queries.parquet"
        qrels_path = dataset_dir / "qrels.parquet"

        corpus_df.to_parquet(corpus_path, index=False)
        queries_df.to_parquet(queries_path, index=False)
        qrels_df.to_parquet(qrels_path, index=False)

        console.print(
            f"   ✓ Corpus: [green]{corpus_path}[/green] ({len(corpus_df)} documents)"
        )
        console.print(
            f"   ✓ Queries: [green]{queries_path}[/green] ({len(queries_df)} queries)"
        )
        console.print(
            f"   ✓ Qrels: [green]{qrels_path}[/green] ({len(qrels_df)} relevance judgments)"
        )

        # Print summary statistics
        console.print("\n📊 Dataset statistics:")
        console.print(f"   • Corpus size: {len(corpus_df)} documents")
        console.print(f"   • Queries: {len(queries_df)}")
        console.print(f"   • Relevance judgments: {len(qrels_df)}")

        # Calculate average relevance judgments per query
        avg_rels_per_query = (
            len(qrels_df) / len(queries_df) if len(queries_df) > 0 else 0
        )
        console.print(f"   • Avg relevance per query: {avg_rels_per_query:.2f}")

    except DatasetNotFoundError:
        console.print(
            f"\n⚠️  [yellow]Dataset '{dataset_name}' not found on HuggingFace.[/yellow]"
        )
        console.print("   Please check the dataset name and try again.")
        console.print(
            "\n   [dim]Example: 'mteb/scifact', 'mteb/nfcorpus', 'mteb/fiqa'[/dim]"
        )
        raise
    except Exception as e:
        console.print(f"\n❌ [red]Error downloading dataset: {e}[/red]")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download MTEB v2 retrieval datasets from HuggingFace"
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="HuggingFace dataset identifier (e.g., 'mteb/scifact', 'mteb/nfcorpus')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./_data/mteb"),
        help="Output directory for downloaded datasets (default: ./_data/mteb)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Download only a sample of N documents (and corresponding queries/qrels)",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language code for multilingual datasets with language-prefixed configs (e.g., 'de', 'en', 'es')",
    )

    args = parser.parse_args()

    # Print header
    console.print(Panel("📦 MTEB Dataset Downloader", style="bold magenta"))

    try:
        download_mteb_dataset(
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            sample_size=args.sample,
            language=args.language,
        )

        console.print(Panel("✅ Download complete!", style="bold green"))
        return 0

    except Exception as e:
        console.print(f"\n❌ [red]Failed to download dataset: {e}[/red]")
        return 1


if __name__ == "__main__":
    exit(main())
