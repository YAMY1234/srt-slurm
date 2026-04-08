"""
Cache manager for storing processed data in parquet format.

Provides efficient disk-based caching to avoid re-parsing log files and JSON data
every time the Streamlit app loads.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages parquet-based caching for benchmark data."""

    def __init__(self, run_dir: str):
        """Initialize cache manager for a specific run directory.

        Args:
            run_dir: Path to the run directory (e.g., "3667_1P_1D_20251110_192145")
        """
        self.run_dir = Path(run_dir)
        self.cache_dir = self.run_dir / "cached_assets"
        self.cache_dir.mkdir(exist_ok=True)

    def is_cache_valid(
        self, cache_name: str, source_patterns: list[str] | None = None, sentinel: str | None = None
    ) -> bool:
        """Check if cached parquet is up-to-date.

        Validation strategy (one stat() each, no file content reads):
        - parquet must exist
        - If a sentinel file is given (e.g. "logs/benchmark.out"), compare its
          current size against the size recorded when the cache was written.
          A size change means the job produced more results → cache stale.
        - If no sentinel, existence alone is sufficient (immutable data).

        Args:
            cache_name: Name of the cache (e.g., "benchmark_results", "node_metrics")
            source_patterns: Ignored (kept for API compatibility)
            sentinel: Optional path relative to run_dir whose file size is
                      used as a lightweight staleness check.
        """
        cache_file = self.cache_dir / f"{cache_name}.parquet"
        if not cache_file.exists():
            return False

        if sentinel is None:
            return True

        sentinel_path = self.run_dir / sentinel
        size_file = self.cache_dir / f"{cache_name}.size"

        try:
            current_size = sentinel_path.stat().st_size
        except OSError:
            return True  # sentinel doesn't exist (yet) → cache is fine

        if not size_file.exists():
            # Legacy cache without size record — trust it and backfill the size
            # so future checks can detect real changes.
            try:
                size_file.write_text(str(current_size))
            except OSError:
                pass
            return True

        try:
            recorded_size = int(size_file.read_text().strip())
        except (ValueError, OSError):
            return False

        return current_size == recorded_size

    def save_to_cache(
        self,
        cache_name: str,
        data: pd.DataFrame | list[dict],
        source_patterns: list[str] | None = None,
        sentinel: str | None = None,
    ) -> None:
        """Save data to parquet cache.

        Args:
            cache_name: Name of the cache (e.g., "benchmark_results", "node_metrics")
            data: Data to cache (DataFrame or list of dicts)
            source_patterns: Ignored (kept for API compatibility)
            sentinel: Optional path relative to run_dir; its current file size
                      is recorded for later staleness checks.
        """
        if isinstance(data, list):
            if not data:
                df = pd.DataFrame()
            else:
                df = pd.DataFrame(data)
        else:
            df = data

        cache_file = self.cache_dir / f"{cache_name}.parquet"
        df.to_parquet(cache_file, index=False, compression="snappy")

        if sentinel:
            sentinel_path = self.run_dir / sentinel
            try:
                size = sentinel_path.stat().st_size
                size_file = self.cache_dir / f"{cache_name}.size"
                size_file.write_text(str(size))
            except OSError:
                pass

        logger.info(f"Cached {len(df)} rows to {cache_file.name}")

    def load_from_cache(self, cache_name: str) -> pd.DataFrame | None:
        """Load data from parquet cache.

        Args:
            cache_name: Name of the cache (e.g., "benchmark_run", "node_metrics")

        Returns:
            DataFrame if cache exists, None otherwise
        """
        cache_file = self.cache_dir / f"{cache_name}.parquet"
        if not cache_file.exists():
            return None

        try:
            df = pd.read_parquet(cache_file)
            logger.info(f"Loaded {len(df)} rows from {cache_file.name}")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file.name}: {e}")
            return None

    def invalidate_cache(self, cache_name: str | None = None) -> None:
        """Invalidate cache (delete cached files).

        Args:
            cache_name: Specific cache to invalidate, or None to invalidate all
        """
        if cache_name:
            for ext in (".parquet", ".size"):
                f = self.cache_dir / f"{cache_name}{ext}"
                if f.exists():
                    f.unlink()
            logger.info(f"Invalidated cache: {cache_name}")
        else:
            for f in self.cache_dir.iterdir():
                if f.suffix in (".parquet", ".size", ".json"):
                    f.unlink()
            logger.info("Invalidated all caches")
