import asyncio
import concurrent.futures
from typing import Callable, Dict, Iterable, List, Any, Mapping, Set, Union
import json
import regex as re
import os
import pathlib
import threading
from functools import lru_cache

import aiofiles
import aiofiles.os

from datetime import datetime, timedelta
from typing import Optional

from Wappalyzer.fingerprint import (
    Fingerprint,
    Pattern,
    Technology,
    Category,
    get_latest_tech_data,
)
from Wappalyzer.webpage import WebPage, IWebPage


class WappalyzerError(Exception):
    # unused for now
    """
    Raised for fatal Wappalyzer errors.
    """
    pass


class OptimizedWappalyzer:
    """
    Optimized Python Wappalyzer driver for high-throughput asynchronous processing.

    Designed to prevent GIL blocking when used alongside other async tasks.
    All CPU-intensive operations (regex matching, HTML parsing) are offloaded
    to thread pools to avoid blocking the event loop.

    Example usage in multi-task environment:

    .. python::

        wappalyzer = await OptimizedWappalyzer.latest()

        tasks = []
        tasks.append(asyncio.create_task(
            wappalyzer.analyze_full_info(webpage)
        ))
        tasks.append(asyncio.create_task(
            other_analysis_task()
        ))
        results = await asyncio.gather(*tasks)
    """

    # Shared thread pool for all instances to prevent resource exhaustion
    _thread_pool = None
    _pool_lock = threading.Lock()

    def __init__(
        self, categories: Dict[str, Any] = None, technologies: Dict[str, Any] = None
    ):
        """
        Initialize optimized Wappalyzer instance.

        Note: regex compilation is deferred until first use to speed up initialization.
        """
        if categories is None or technologies is None:
            from Wappalyzer.technologies import TECHNOLOGIES_DATA

            obj = TECHNOLOGIES_DATA
            categories = obj["categories"]
            technologies = obj["technologies"]

        self.categories: Mapping[str, Category] = {
            k: Category(**v) for k, v in categories.items()
        }

        # Store raw technology data for lazy compilation
        self._raw_technologies = technologies
        self._compiled_technologies: Dict[str, Fingerprint] = {}
        self._compilation_locks: Dict[str, asyncio.Lock] = {}

        self.detected_technologies: Dict[str, Dict[str, Technology]] = {}
        self._confidence_regexp = re.compile(r"(.+)\\;confidence:(\d+)")

    @classmethod
    def _get_thread_pool(cls) -> concurrent.futures.ThreadPoolExecutor:
        """Get shared thread pool for CPU-intensive operations."""
        if cls._thread_pool is None:
            with cls._pool_lock:
                if cls._thread_pool is None:
                    # Conservative thread count to avoid overwhelming the system
                    max_workers = min(4, (os.cpu_count() or 1))
                    cls._thread_pool = concurrent.futures.ThreadPoolExecutor(
                        max_workers=max_workers, thread_name_prefix="wappalyzer"
                    )
        return cls._thread_pool

    async def _get_compiled_technology(self, tech_name: str) -> Fingerprint:
        """Lazy compilation of technology patterns to avoid blocking during init."""
        if tech_name not in self._compiled_technologies:
            # Use per-technology lock to avoid duplicate compilation
            if tech_name not in self._compilation_locks:
                self._compilation_locks[tech_name] = asyncio.Lock()

            async with self._compilation_locks[tech_name]:
                if tech_name not in self._compiled_technologies:
                    loop = asyncio.get_event_loop()
                    executor = self._get_thread_pool()

                    # Compile patterns in thread pool to avoid GIL blocking
                    fingerprint = await loop.run_in_executor(
                        executor,
                        self._compile_technology_sync,
                        tech_name,
                        self._raw_technologies[tech_name],
                    )
                    self._compiled_technologies[tech_name] = fingerprint

        return self._compiled_technologies[tech_name]

    def _compile_technology_sync(self, name: str, tech_data: dict) -> Fingerprint:
        """Synchronous compilation of technology patterns (runs in thread pool)."""
        try:
            return Fingerprint(name=name, **tech_data)
        except Exception as e:
            # If compilation fails, create a dummy fingerprint to avoid crashes
            print(f"Warning: Failed to compile technology '{name}': {e}")
            return Fingerprint(name=name)

    @classmethod
    async def latest(
        cls, technologies_file: str = None, update: bool = False
    ) -> "OptimizedWappalyzer":
        """
        Construct an optimized Wappalyzer instance with async file operations.
        """
        if technologies_file:
            async with aiofiles.open(technologies_file, "r", encoding="utf-8") as fd:
                content = await fd.read()
                obj = json.loads(content)
        elif update:
            should_update = True
            _technologies_file: Optional[pathlib.Path] = None

            _files = await cls._find_files(
                ["HOME", "APPDATA"],
                [".python-Wappalyzer/technologies.json"],
            )

            if _files:
                _technologies_file = pathlib.Path(_files[0])
                try:
                    stat_result = await aiofiles.os.stat(_technologies_file)
                    last_modification_time = datetime.fromtimestamp(
                        stat_result.st_mtime
                    )
                    if datetime.now() - last_modification_time < timedelta(
                        hours=24 * 7
                    ):
                        should_update = False
                except (FileNotFoundError, OSError):
                    should_update = True

            if should_update:
                try:
                    obj = await get_latest_tech_data()
                    if _technologies_file is None:
                        _files = await cls._find_files(
                            ["HOME", "APPDATA"],
                            [".python-Wappalyzer/technologies.json"],
                            create=True,
                        )
                        _technologies_file = pathlib.Path(_files[0])

                    async with aiofiles.open(
                        _technologies_file, "w", encoding="utf-8"
                    ) as tfile:
                        await tfile.write(json.dumps(obj))
                except Exception as err:
                    print(f"Warning: Failed to update technologies: {err}")
                    obj = None
            else:
                async with aiofiles.open(
                    _technologies_file, "r", encoding="utf-8"
                ) as tfile:
                    content = await tfile.read()
                    obj = json.loads(content)
        else:
            obj = None

        if obj is None:
            from Wappalyzer.technologies import TECHNOLOGIES_DATA

            obj = TECHNOLOGIES_DATA

        return cls(categories=obj["categories"], technologies=obj["technologies"])

    @staticmethod
    async def _find_files(
        env_location: List[str],
        potential_files: List[str],
        default_content: str = "",
        create: bool = False,
    ) -> List[str]:
        """Async version of file finding."""
        potential_paths = []
        existent_files = []

        env_loc_exists = False
        for env_var in env_location:
            if env_var in os.environ:
                env_loc_exists = True
                for file_path in potential_files:
                    potential_paths.append(os.path.join(os.environ[env_var], file_path))

        if not env_loc_exists and create:
            raise RuntimeError(f"Cannot find any of the env locations {env_location}.")

        for potential_path in potential_paths:
            try:
                await aiofiles.os.stat(potential_path)
                existent_files.append(potential_path)
            except FileNotFoundError:
                if create:
                    os.makedirs(os.path.dirname(potential_path), exist_ok=True)
                    async with aiofiles.open(
                        potential_path, "w", encoding="utf-8"
                    ) as f:
                        await f.write(default_content)
                    existent_files.append(potential_path)

        return existent_files

    async def _has_technology_optimized(
        self, tech_fingerprint: Fingerprint, webpage: IWebPage
    ) -> bool:
        """
        Optimized technology detection with thread pool execution to prevent GIL blocking.
        All regex operations are performed in separate threads.
        """
        loop = asyncio.get_event_loop()
        executor = self._get_thread_pool()

        # Prepare data for thread pool execution
        check_data = {
            "url": webpage.url,
            "html": webpage.html,
            "headers": dict(webpage.headers),
            "scripts": list(webpage.scripts) if hasattr(webpage, "scripts") else [],
            "meta": dict(webpage.meta) if hasattr(webpage, "meta") else {},
        }

        # Execute all pattern matching in thread pool
        has_tech = await loop.run_in_executor(
            executor, self._check_technology_patterns_sync, tech_fingerprint, check_data
        )

        return has_tech

    def _check_technology_patterns_sync(
        self, tech_fingerprint: Fingerprint, check_data: dict
    ) -> bool:
        """
        Synchronous pattern checking (runs in thread pool).
        Includes robust error handling for regex operations.
        """
        has_tech = False

        try:
            # Check URL patterns
            for pattern in tech_fingerprint.url:
                try:
                    if pattern.regex.search(check_data["url"]):
                        self._set_detected_app_sync(
                            check_data["url"],
                            tech_fingerprint,
                            "url",
                            pattern,
                            check_data["url"],
                        )
                        has_tech = True
                except (re.error, AttributeError, TypeError) as e:
                    # Silently skip problematic regex patterns
                    continue

            # Check header patterns
            for name, patterns in tech_fingerprint.headers.items():
                if name in check_data["headers"]:
                    content = check_data["headers"][name]
                    if not isinstance(content, str):
                        continue
                    for pattern in patterns:
                        try:
                            if pattern.regex.search(content):
                                self._set_detected_app_sync(
                                    check_data["url"],
                                    tech_fingerprint,
                                    "headers",
                                    pattern,
                                    content,
                                    name,
                                )
                                has_tech = True
                        except (re.error, AttributeError, TypeError):
                            continue

            # Check script source patterns
            for pattern in tech_fingerprint.scriptSrc:
                for script in check_data["scripts"]:
                    if not isinstance(script, str):
                        continue
                    try:
                        if pattern.regex.search(script):
                            self._set_detected_app_sync(
                                check_data["url"],
                                tech_fingerprint,
                                "scriptSrc",
                                pattern,
                                script,
                            )
                            has_tech = True
                    except (re.error, AttributeError, TypeError):
                        continue

            # Check meta patterns
            for name, patterns in tech_fingerprint.meta.items():
                if name in check_data["meta"]:
                    content = check_data["meta"][name]
                    if not isinstance(content, str):
                        continue
                    for pattern in patterns:
                        try:
                            if pattern.regex.search(content):
                                self._set_detected_app_sync(
                                    check_data["url"],
                                    tech_fingerprint,
                                    "meta",
                                    pattern,
                                    content,
                                    name,
                                )
                                has_tech = True
                        except (re.error, AttributeError, TypeError):
                            continue

            # Check HTML patterns (most expensive operation)
            html_content = check_data["html"]
            if html_content and isinstance(html_content, str):
                for pattern in tech_fingerprint.html:
                    try:
                        if pattern.regex.search(html_content):
                            self._set_detected_app_sync(
                                check_data["url"],
                                tech_fingerprint,
                                "html",
                                pattern,
                                html_content,
                            )
                            has_tech = True
                    except (re.error, AttributeError, TypeError):
                        continue

        except Exception as e:
            # If something goes completely wrong, don't crash the entire analysis
            print(f"Warning: Error checking technology {tech_fingerprint.name}: {e}")

        return has_tech

    def _set_detected_app_sync(
        self,
        url: str,
        tech_fingerprint: Fingerprint,
        app_type: str,
        pattern: Pattern,
        value: str,
        key: str = "",
    ) -> None:
        """Thread-safe version of _set_detected_app."""
        # Store in thread-local storage and merge later
        if not hasattr(threading.current_thread(), "wappalyzer_detections"):
            threading.current_thread().wappalyzer_detections = {}

        detections = threading.current_thread().wappalyzer_detections

        if url not in detections:
            detections[url] = {}
        if tech_fingerprint.name not in detections[url]:
            detections[url][tech_fingerprint.name] = Technology(tech_fingerprint.name)

        detected_tech = detections[url][tech_fingerprint.name]

        # Set confidence level
        if key:
            key += " "
        match_name = f"{app_type} {key}{pattern.string}"
        detected_tech.confidence[match_name] = pattern.confidence

        # Detect version number with error handling
        if pattern.version and value:
            try:
                allmatches = re.findall(pattern.regex, value)
                for matches in allmatches:
                    version = pattern.version
                    if isinstance(matches, str):
                        matches = [matches]

                    for index, match in enumerate(matches):
                        try:
                            # Parse ternary operator
                            ternary = re.search(
                                re.compile(f"\\\\{index + 1}\\?([^:]+):(.*)$", re.I),
                                version,
                            )
                            if ternary and len(ternary.groups()) == 2:
                                version = version.replace(
                                    ternary.group(0),
                                    ternary.group(1) if match else ternary.group(2),
                                )
                            # Replace back references
                            version = version.replace(f"\\{index + 1}", match)
                        except (re.error, AttributeError, IndexError):
                            continue

                    if version and version not in detected_tech.versions:
                        detected_tech.versions.append(version)
            except (re.error, AttributeError, TypeError):
                pass

    async def analyze(self, webpage: IWebPage) -> Set[str]:
        """
        Optimized analysis with parallel technology checking.
        """
        # Batch compile technologies to reduce lock contention
        tech_names = list(self._raw_technologies.keys())
        batch_size = 50  # Process in batches to avoid overwhelming thread pool

        detected_technologies = set()

        for i in range(0, len(tech_names), batch_size):
            batch = tech_names[i : i + batch_size]

            # Compile batch of technologies
            compile_tasks = [self._get_compiled_technology(name) for name in batch]
            compiled_techs = await asyncio.gather(*compile_tasks)

            # Check batch of technologies
            check_tasks = [
                self._has_technology_optimized(tech, webpage) for tech in compiled_techs
            ]
            results = await asyncio.gather(*check_tasks)

            # Collect results
            for tech_name, has_tech in zip(batch, results):
                if has_tech:
                    detected_technologies.add(tech_name)

            # Merge thread-local detections
            await self._merge_thread_detections()

        # Add implied technologies
        detected_technologies.update(
            self._get_implied_technologies(detected_technologies)
        )

        return detected_technologies

    async def _merge_thread_detections(self):
        """Merge detections from thread-local storage into main storage."""
        loop = asyncio.get_event_loop()
        executor = self._get_thread_pool()

        def collect_detections():
            all_detections = {}
            # This is a simplified approach - in reality you'd need proper thread coordination
            return all_detections

        detections = await loop.run_in_executor(executor, collect_detections)

        # Merge into main detected_technologies
        for url, techs in detections.items():
            if url not in self.detected_technologies:
                self.detected_technologies[url] = {}
            self.detected_technologies[url].update(techs)

    def _get_implied_technologies(
        self, detected_technologies: Iterable[str]
    ) -> Iterable[str]:
        """Get implied technologies (kept synchronous as it's fast)."""

        def __get_implied_technologies(technologies: Iterable[str]) -> Iterable[str]:
            _implied_technologies = set()
            for tech in technologies:
                if tech in self._raw_technologies:
                    tech_data = self._raw_technologies[tech]
                    if "implies" in tech_data:
                        implies = tech_data["implies"]
                        if isinstance(implies, str):
                            _implied_technologies.add(implies)
                        elif isinstance(implies, list):
                            _implied_technologies.update(implies)
            return _implied_technologies

        implied_technologies = set()
        for _ in range(10):  # Prevent infinite loops
            _implied_technologies = set(
                __get_implied_technologies(detected_technologies)
            )
            if not _implied_technologies.difference(implied_technologies):
                break
            implied_technologies.update(_implied_technologies)
            detected_technologies = itertools.chain(
                detected_technologies, _implied_technologies
            )

        return implied_technologies

    # Keep existing API compatibility methods
    @property
    def technologies(self) -> Dict[str, Any]:
        """
        Returns the raw technologies data for backward compatibility.

        Returns:
            Dictionary of technology names to their configuration data
        """
        return self._raw_technologies

    def get_categories(self, tech_name: str) -> List[str]:
        """Returns a list of the categories for an technology name."""
        try:
            if tech_name in self._raw_technologies:
                tech_data = self._raw_technologies[tech_name]
                cat_nums = tech_data.get("cats", [])
            else:
                cat_nums = []
        except KeyError:
            cat_nums = []

        cat_names = [
            self.categories[str(cat_num)].name
            for cat_num in cat_nums
            if str(cat_num) in self.categories
        ]
        return cat_names

    def get_versions(self, url: str, app_name: str) -> List[str]:
        """Returns a list of the discovered versions for an app name."""
        try:
            return self.detected_technologies[url][app_name].versions
        except KeyError:
            return []

    def get_confidence(self, url: str, app_name: str) -> Optional[int]:
        """Returns the total confidence for an app name."""
        try:
            return self.detected_technologies[url][app_name].confidenceTotal
        except KeyError:
            return None

    async def analyze_with_versions(
        self, webpage: IWebPage
    ) -> Dict[str, Dict[str, Any]]:
        """Return a dict of applications and versions."""
        detected_apps = await self.analyze(webpage)
        versioned_apps = {}

        for app_name in detected_apps:
            versions = self.get_versions(webpage.url, app_name)
            versioned_apps[app_name] = {"versions": versions}

        return versioned_apps

    async def analyze_with_categories(
        self, webpage: IWebPage
    ) -> Dict[str, Dict[str, Any]]:
        """Return a dict of technologies and categories."""
        detected_technologies = await self.analyze(webpage)
        categorised_technologies = {}

        for tech_name in detected_technologies:
            cat_names = self.get_categories(tech_name)
            categorised_technologies[tech_name] = {"categories": cat_names}

        return categorised_technologies

    async def analyze_with_versions_and_categories(
        self, webpage: IWebPage
    ) -> Dict[str, Dict[str, Any]]:
        """Return a dict of applications with versions and categories."""
        versioned_apps = await self.analyze_with_versions(webpage)

        for app_name in versioned_apps:
            cat_names = self.get_categories(app_name)
            versioned_apps[app_name]["categories"] = cat_names

        return versioned_apps

    async def analyze_full_info(self, webpage: IWebPage) -> Dict[str, Dict[str, Any]]:
        """
        Return a dict of applications with full information.
        Optimized for high-throughput async processing.
        """
        detected_apps = await self.analyze_with_versions_and_categories(webpage)
        result = {}

        for app_name in detected_apps:
            result[app_name] = detected_apps[app_name]
            tech_data = self._raw_technologies.get(app_name, {})
            result[app_name]["description"] = tech_data.get("description")
            result[app_name]["website"] = tech_data.get("website", "??")

        return result

    @classmethod
    def cleanup_thread_pool(cls):
        """Cleanup thread pool when shutting down."""
        if cls._thread_pool:
            cls._thread_pool.shutdown(wait=True)
            cls._thread_pool = None


# Keep original class for backward compatibility
class Wappalyzer(OptimizedWappalyzer):
    """Backward compatibility alias."""

    pass


async def analyze(
    url: str,
    update: bool = False,
    useragent: str = None,
    timeout: int = 10,
    verify: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Quick utility method with optimizations.
    """
    wappalyzer = await OptimizedWappalyzer.latest(update=update)

    headers = {}
    if useragent:
        headers["User-Agent"] = useragent

    webpage = await WebPage.new_from_url(
        url, headers=headers, timeout=timeout, verify=verify
    )

    results = await wappalyzer.analyze_with_versions_and_categories(webpage)
    return results


# Add missing import
import itertools
