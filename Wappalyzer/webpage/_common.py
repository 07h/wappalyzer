"""
Containers for a Web page and its components.
Wraps only the information strictly necessary to run the Wappalyzer engine.
Optimized to prevent GIL blocking in multi-task environments.
"""

import abc
import asyncio
import concurrent.futures
import threading
from typing import Iterable, List, Mapping, Any

try:
    from typing import Protocol
except ImportError:
    Protocol = object  # type: ignore

import httpx


def _raise_not_dict(obj: Any, name: str) -> None:
    try:
        list(obj.keys())
    except AttributeError:
        raise ValueError(f"{name} must be a dictionary-like object")


class ITag(Protocol):
    """
    A HTML tag, decoupled from any particular HTTP library's API.
    """

    name: str
    attributes: Mapping[str, str]
    inner_html: str


class BaseTag(ITag, abc.ABC):
    """
    Subclasses must implement inner_html().
    """

    def __init__(self, name: str, attributes: Mapping[str, str]) -> None:
        _raise_not_dict(attributes, "attributes")
        self.name = name
        self.attributes = attributes

    @property
    def inner_html(self) -> str:  # type: ignore
        """Returns the inner HTML of an element as a UTF-8 encoded bytestring"""
        raise NotImplementedError()


class IWebPage(Protocol):
    """
    Interface declaring the required methods/attributes of a WebPage object.

    Simple representation of a web page, decoupled from any particular HTTP library's API.
    """

    url: str
    html: str
    headers: Mapping[str, str]
    scripts: List[str]  # list of the script sources urls
    meta: Mapping[str, str]

    def select(self, selector: str) -> Iterable[ITag]:
        raise NotImplementedError()


class OptimizedBaseWebPage(IWebPage):
    """
    Optimized WebPage implementation that prevents GIL blocking.

    HTML parsing is performed in thread pools to avoid blocking the event loop
    when used alongside other async tasks.
    """

    # Shared thread pool for all instances
    _parse_pool = None
    _pool_lock = threading.Lock()

    def __init__(self, url: str, html: str, headers: Mapping[str, str]):
        """
        Initialize a new WebPage object.

        HTML parsing is deferred and performed asynchronously to avoid blocking.
        """
        _raise_not_dict(headers, "headers")
        self.url = url
        self.html = html
        self.headers = headers
        self.scripts: List[str] = []
        self.meta: Mapping[str, str] = {}
        self._parsed_html = None
        self._parse_completed = False
        self._parse_lock = (
            asyncio.Lock() if hasattr(asyncio, "_get_running_loop") else None
        )

        # For backward compatibility, always perform synchronous parsing in constructor
        # This ensures that .meta and .scripts are immediately available
        try:
            self._parse_html_sync()
            self._parse_completed = True
        except Exception as e:
            print(f"Warning: Synchronous HTML parsing failed: {e}")
            # Set safe defaults
            self.scripts = []
            self.meta = {}
            self._parsed_html = None
            self._parse_completed = False

    @classmethod
    def _get_parse_pool(cls) -> concurrent.futures.ThreadPoolExecutor:
        """Get shared thread pool for HTML parsing operations."""
        if cls._parse_pool is None:
            with cls._pool_lock:
                if cls._parse_pool is None:
                    # Conservative thread count for parsing
                    max_workers = min(2, (4 or 1))  # Reduced for parsing tasks
                    cls._parse_pool = concurrent.futures.ThreadPoolExecutor(
                        max_workers=max_workers, thread_name_prefix="html_parser"
                    )
        return cls._parse_pool

    async def _ensure_parsed(self):
        """Ensure HTML is parsed, performing async parsing if needed."""
        if self._parse_completed:
            return

        if self._parse_lock is None:
            # Fallback for edge cases
            self._parse_html_sync()
            self._parse_completed = True
            return

        async with self._parse_lock:
            if not self._parse_completed:
                loop = asyncio.get_event_loop()
                executor = self._get_parse_pool()

                # Parse HTML in thread pool to prevent GIL blocking
                try:
                    result = await loop.run_in_executor(executor, self._parse_html_sync)
                    if result:
                        self.scripts, self.meta, self._parsed_html = result
                except Exception as e:
                    print(f"Warning: HTML parsing failed: {e}")
                    # Set safe defaults
                    self.scripts = []
                    self.meta = {}
                    self._parsed_html = None

                self._parse_completed = True

    def _parse_html_sync(self):
        """
        Synchronous HTML parsing (runs in thread pool).

        Subclasses should override this method.
        Returns tuple of (scripts, meta, parsed_html) or None on failure.
        """
        try:
            # This will be implemented by subclasses
            self._parse_html()
            return self.scripts, self.meta, self._parsed_html
        except Exception as e:
            print(f"Warning: Failed to parse HTML: {e}")
            return [], {}, None

    def _parse_html(self):
        """Override in subclasses."""
        raise NotImplementedError()

    async def select(self, selector: str) -> Iterable[ITag]:
        """
        Execute a CSS selector asynchronously.

        Ensures HTML is parsed before selection and performs selection
        in thread pool if needed.
        """
        await self._ensure_parsed()

        # If parsing failed, return empty result
        if self._parsed_html is None:
            return ()

        # For simple selectors, we can do sync selection
        # For complex cases, might need thread pool
        try:
            return self._select_sync(selector)
        except Exception as e:
            print(f"Warning: CSS selection failed: {e}")
            return ()

    def _select_sync(self, selector: str) -> Iterable[ITag]:
        """Synchronous selection (override in subclasses)."""
        return ()

    @classmethod
    async def new_from_url(cls, url: str, **kwargs: Any) -> IWebPage:
        """
        Constructs a new WebPage object for the URL with optimized HTTP client.
        """
        # Use connection pooling and timeout settings optimized for batch processing
        timeout = kwargs.pop("timeout", 30)

        # Create client with optimal settings for high-throughput
        client_kwargs = {
            "timeout": httpx.Timeout(timeout),
            "limits": httpx.Limits(
                max_connections=20, max_keepalive_connections=10, keepalive_expiry=30
            ),
            "follow_redirects": True,
            **kwargs,
        }

        async with httpx.AsyncClient(**client_kwargs) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return await cls.new_from_response(response)
            except httpx.RequestError as e:
                # Return a minimal webpage for failed requests to prevent crashes
                print(f"Warning: Failed to fetch {url}: {e}")
                return cls(url, html="", headers={})
            except httpx.HTTPStatusError as e:
                print(f"Warning: HTTP error for {url}: {e}")
                return cls(url, html="", headers={})

    @classmethod
    async def new_from_response(cls, response: httpx.Response) -> IWebPage:
        """
        Constructs a new WebPage object for the response.
        """
        try:
            html = response.text
        except UnicodeDecodeError as e:
            print(f"Warning: Failed to decode response text: {e}")
            html = ""

        return cls(str(response.url), html=html, headers=response.headers)

    @classmethod
    def cleanup_parse_pool(cls):
        """Cleanup parsing thread pool when shutting down."""
        if cls._parse_pool:
            cls._parse_pool.shutdown(wait=True)
            cls._parse_pool = None


# Keep original class for backward compatibility
class BaseWebPage(OptimizedBaseWebPage):
    """Backward compatibility alias."""

    def __init__(self, url: str, html: str, headers: Mapping[str, str]):
        super().__init__(url, html, headers)
        # For backward compatibility, parse immediately (but this blocks)
        try:
            self._parse_html()
            self._parse_completed = True
        except Exception as e:
            print(f"Warning: Immediate HTML parsing failed: {e}")
            self.scripts = []
            self.meta = {}
            self._parsed_html = None
            self._parse_completed = True
