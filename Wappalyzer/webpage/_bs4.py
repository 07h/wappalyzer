"""
Implementation of WebPage based on selectolax.
"""

from typing import Iterable, Iterator, Mapping

# Just to check if it's available
import lxml  # type: ignore
from selectolax.parser import HTMLParser, Node
from cached_property import cached_property  # type: ignore

from ._common import BaseWebPage, BaseTag


class Tag(BaseTag):
    def __init__(self, name: str, attributes: Mapping[str, str], node: Node) -> None:
        super().__init__(name, attributes)
        self._node = node

    @cached_property
    def inner_html(self) -> str:
        return self._node.html


class WebPage(BaseWebPage):
    """
    Simple representation of a web page, decoupled
    from any particular HTTP library's API.

    Well, except for the class methods that use `requests`
    or `aiohttp` to create the WebPage.

    This object is designed to be created for each website scanned
    by python-Wappalyzer.
    It will parse the HTML with selectolax to find <script> and <meta> tags.

    You can create it from manually from HTML with the `WebPage()` method
    or from the class methods.
    """

    def _parse_html(self):
        """
        Parse the HTML with selectolax to find <script> and <meta> tags.
        """
        self._parsed_html = parser = HTMLParser(self.html)
        self.scripts.extend(
            script.attributes["src"] for script in parser.css("script[src]")
        )
        self.meta = {
            meta.attributes["name"].lower(): meta.attributes["content"]
            for meta in parser.css("meta[name][content]")
            if "name" in meta.attributes
            and meta.attributes["name"] is not None
            and "content" in meta.attributes
        }

    def select(self, selector: str) -> Iterable[Tag]:
        """Execute a CSS select and returns results as Tag objects."""
        try:
            for item in self._parsed_html.css(selector):
                yield Tag(item.tag, item.attributes, item)
        except Exception as e:
            return ()
