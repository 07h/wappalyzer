"""
Implementation of WebPage based on selectolax.
Optimized to prevent GIL blocking in multi-task environments.
"""

from typing import Iterable, Iterator, Mapping
import asyncio

# Just to check if it's available
import lxml  # type: ignore
from selectolax.parser import HTMLParser, Node
from cached_property import cached_property  # type: ignore

from ._common import OptimizedBaseWebPage, BaseTag


class Tag(BaseTag):
    def __init__(self, name: str, attributes: Mapping[str, str], node: Node) -> None:
        super().__init__(name, attributes)
        self._node = node

    @cached_property
    def inner_html(self) -> str:
        try:
            return self._node.html or ""
        except Exception:
            return ""


class WebPage(OptimizedBaseWebPage):
    """
    Optimized WebPage implementation using selectolax.

    HTML parsing is performed in thread pools to prevent GIL blocking
    when used alongside other async tasks.

    This object is designed for high-throughput processing where multiple
    webpages are analyzed concurrently.
    """

    def _parse_html(self):
        """
        Parse the HTML with selectolax (runs in thread pool).
        Includes robust error handling for malformed HTML.
        """
        if not self.html:
            self.scripts = []
            self.meta = {}
            self._parsed_html = None
            return

        try:
            self._parsed_html = parser = HTMLParser(self.html)

            # Extract script sources with error handling
            scripts = []
            try:
                script_elements = parser.css("script[src]")
                for script in script_elements:
                    if script and script.attributes and "src" in script.attributes:
                        src = script.attributes["src"]
                        if src and isinstance(src, str):
                            scripts.append(src)
            except Exception as e:
                print(f"Warning: Failed to extract scripts: {e}")

            self.scripts = scripts

            # Extract meta tags with error handling
            meta = {}
            try:
                meta_elements = parser.css("meta[name][content]")
                for meta_el in meta_elements:
                    if meta_el and meta_el.attributes:
                        name = meta_el.attributes.get("name")
                        content = meta_el.attributes.get("content")
                        if (
                            name
                            and content
                            and isinstance(name, str)
                            and isinstance(content, str)
                        ):
                            meta[name.lower()] = content
            except Exception as e:
                print(f"Warning: Failed to extract meta tags: {e}")

            self.meta = meta

        except Exception as e:
            print(f"Warning: HTML parsing failed completely: {e}")
            self.scripts = []
            self.meta = {}
            self._parsed_html = None

    def _select_sync(self, selector: str) -> Iterable[Tag]:
        """Execute a CSS select and returns results as Tag objects."""
        if not self._parsed_html:
            return []

        try:
            results = []
            for item in self._parsed_html.css(selector):
                if item and item.tag and item.attributes is not None:
                    results.append(Tag(item.tag, item.attributes, item))
            return results
        except Exception as e:
            print(f"Warning: CSS selector '{selector}' failed: {e}")
            return []
