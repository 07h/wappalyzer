"""
Containers for a Web page and its components.
Wraps only the information strictly necessary to run the Wappalyzer engine.
"""

import abc
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


class BaseWebPage(IWebPage):
    """
    Implements factory methods for a WebPage.

    Subclasses must implement _parse_html() and select(string).
    """

    def __init__(self, url: str, html: str, headers: Mapping[str, str]):
        """
        Initialize a new WebPage object manually.

        >>> from Wappalyzer import WebPage
        >>> w = WebPage('example.com', html='<strong>Hello World</strong>', headers={'Server': 'Apache', })

        :param url: The web page URL.
        :param html: The web page content (HTML)
        :param headers: The HTTP response headers
        """
        _raise_not_dict(headers, "headers")
        self.url = url
        self.html = html
        self.headers = headers
        self.scripts: List[str] = []
        self.meta: Mapping[str, str] = {}
        self._parse_html()

    def _parse_html(self):
        raise NotImplementedError()

    @classmethod
    async def new_from_url(cls, url: str, **kwargs: Any) -> IWebPage:
        """
        Constructs a new WebPage object for the URL,
        using the `httpx` module to fetch the HTML asynchronously.

        >>> from Wappalyzer import WebPage
        >>> page = await WebPage.new_from_url('example.com', timeout=5)

        :param url: URL
        :param headers: (optional) Dictionary of HTTP Headers to send.
        :param cookies: (optional) Dict or CookieJar object to send.
        :param timeout: (optional) How many seconds to wait for the server to send data before giving up.
        :param proxies: (optional) Dictionary mapping protocol to the URL of the proxy.
        :param verify: (optional) Boolean, it controls whether we verify the SSL certificate validity.
        :param **kwargs: Any other arguments are passed to `httpx.AsyncClient.get` method as well.
        """
        async with httpx.AsyncClient(**kwargs) as client:
            response = await client.get(url)
        return await cls.new_from_response(response)

    @classmethod
    async def new_from_response(cls, response: httpx.Response) -> IWebPage:
        """
        Constructs a new WebPage object for the response.

        :param response: `httpx.Response` object
        """
        return cls(str(response.url), html=response.text, headers=response.headers)
