import pytest
import json

from pathlib import Path

import respx
import httpx
import aiofiles

from Wappalyzer.fingerprint import Fingerprint, get_latest_tech_data
from Wappalyzer import WebPage, Wappalyzer


@pytest.mark.asyncio
async def test_analyze():
    wappalyzer = Wappalyzer()
    webpage = await WebPage.new_from_url_async(
        "https://web.archive.org/web/20180323055000/http://honeyletter.com/"
    )
    result = await wappalyzer.analyze_full_info(webpage)

    assert "PHP" in result


@pytest.mark.asyncio
async def test_new_from_url():
    with respx.mock(base_url="https://www.delish.com") as respx_mock:
        respx_mock.get("/").mock(return_value=httpx.Response(200, text="snerble"))

        webpage = await WebPage.new_from_url("https://www.delish.com/")

        assert webpage.html == "snerble"


@pytest.mark.asyncio
async def test_latest():
    analyzer = await Wappalyzer.latest()

    assert analyzer.categories["1"].name == "CMS"
    assert "Apache" in analyzer.technologies


@pytest.mark.asyncio
async def test_latest_update(tmp_path: Path):
    # Получаем последний файл технологий
    lastest_technologies = await get_latest_tech_data()

    tmp_file = tmp_path.joinpath("technologies.json")
    # Пишем содержимое во временный файл
    async with aiofiles.open(tmp_file, "w", encoding="utf-8") as t_file:
        await t_file.write(json.dumps(lastest_technologies))

    # Создаем экземпляр Wappalyzer с указанным файлом технологий
    wappalyzer1 = await Wappalyzer.latest(technologies_file=str(tmp_file))

    wappalyzer2 = await Wappalyzer.latest(update=True)

    assert len(wappalyzer1.technologies) >= len(wappalyzer2.technologies)
    assert len(wappalyzer1.categories) >= len(wappalyzer2.categories)


@pytest.mark.asyncio
async def test_analyze_no_technologies():
    analyzer = Wappalyzer(categories={}, technologies={})
    webpage = WebPage("https://www.delish.com/", "<html></html>", {})

    detected_technologies = await analyzer.analyze(webpage)

    assert detected_technologies == set()


def test_get_implied_technologies():
    analyzer = Wappalyzer(
        categories={},
        technologies={
            "a": {
                "implies": ["b"],
            },
            "b": {
                "implies": ["c"],
            },
            "c": {
                "implies": ["a"],
            },
        },
    )

    implied_technologies = analyzer._get_implied_technologies(["a"])

    assert implied_technologies == set(["a", "b", "c"])


@pytest.mark.asyncio
async def test_get_analyze_with_categories():
    webpage = WebPage("https://www.delish.com/", "<html>aaa</html>", {})
    categories = {
        "1": {"name": "cat1", "priority": 1},
        "2": {"name": "cat2", "priority": 1},
    }

    technologies = {
        "a": {
            "html": "aaa",
            "cats": [1],
        },
        "b": {
            "html": "bbb",
            "cats": [1, 2],
        },
    }

    analyzer = Wappalyzer(categories=categories, technologies=technologies)
    result = await analyzer.analyze_with_categories(webpage)

    assert result == {"a": {"categories": ["cat1"]}}


@pytest.mark.asyncio
async def test_get_analyze_with_versions():
    webpage = WebPage(
        "http://wordpress-example.com",
        '<html><head><meta name="generator" content="WordPress 5.4.2"></head></html>',
        {},
    )

    categories = {
        "1": {"name": "CMS", "priority": 1},
        "11": {"name": "Blog", "priority": 1},
    }

    technologies = {
        "WordPress": {
            "cats": [1, 11],
            "html": [],
            "icon": "WordPress.svg",
            "implies": ["PHP", "MySQL"],
            "meta": {"generator": "^WordPress ?([\\d.]+)?\\;version:\\1"},
            "website": "https://wordpress.org",
        },
        "b": {
            "html": "bbb",
            "cats": [1, 2],
        },
        "PHP": {"website": "http://php.net"},
        "MySQL": {"website": "http://mysql.com"},
    }

    analyzer = Wappalyzer(categories=categories, technologies=technologies)
    result = await analyzer.analyze_with_versions(webpage)

    assert ("WordPress", {"versions": ["5.4.2"]}) in result.items()


@pytest.mark.asyncio
async def test_analyze_with_versions_and_categories():
    webpage = WebPage(
        "http://wordpress-example.com",
        '<html><head><meta name="generator" content="WordPress 5.4.2"></head></html>',
        {},
    )

    categories = {
        "1": {"name": "CMS", "priority": 1},
        "11": {"name": "Blog", "priority": 1},
    }

    technologies = {
        "WordPress": {
            "cats": [1, 11],
            "html": [],
            "icon": "WordPress.svg",
            "implies": ["PHP", "MySQL"],
            "meta": {"generator": "^WordPress ?([\\d.]+)?\\;version:\\1"},
            "website": "https://wordpress.org",
        },
        "b": {
            "html": "bbb",
            "cats": [1, 2],
        },
        "PHP": {"website": "http://php.net"},
        "MySQL": {"website": "http://mysql.com"},
    }

    analyzer = Wappalyzer(categories=categories, technologies=technologies)
    result = await analyzer.analyze_with_versions_and_categories(webpage)
    assert analyzer.get_versions(webpage.url, "WordPress") == [
        "5.4.2"
    ], analyzer.detected_technologies[webpage.url]
    assert (
        "WordPress",
        {"categories": ["CMS", "Blog"], "versions": ["5.4.2"]},
    ) in result.items()


@pytest.mark.asyncio
async def test_analyze_with_versions_and_categories_pattern_lists():
    webpage = WebPage(
        "http://wordpress-example.com",
        '<html><head><meta name="generator" content="WordPress 5.4.2"></head></html>',
        {},
    )

    categories = {
        "1": {"name": "CMS", "priority": 1},
        "11": {"name": "Blog", "priority": 1},
    }

    technologies = {
        "WordPress": {
            "cats": [1, 11],
            "html": [],
            "icon": "WordPress.svg",
            "implies": ["PHP", "MySQL"],
            "meta": {
                "generator": [
                    "Whatever123",
                    "Whatever456",
                    "^WordPress ?([\\d.]+)?\\;version:\\1",
                    "Whatever",
                ]
            },
            "website": "https://wordpress.org",
        },
        "b": {
            "html": "bbb",
            "cats": [1, 2],
        },
        "PHP": {"website": "http://php.net"},
        "MySQL": {"website": "http://mysql.com"},
    }

    analyzer = Wappalyzer(categories=categories, technologies=technologies)
    result = await analyzer.analyze_with_versions_and_categories(webpage)

    assert (
        "WordPress",
        {"categories": ["CMS", "Blog"], "versions": ["5.4.2"]},
    ) in result.items()


@pytest.mark.asyncio
async def test_analyze_dom_string():
    webpageA = WebPage(
        "https://www.delish.com/", '<html><p class="aaa">webpage a</p></html>', {}
    )
    webpageB = WebPage(
        "https://www.delish.com/", '<html><p id="bbb">webpage b</p></html>', {}
    )
    categories = {}
    technologies = {
        "a": {
            "dom": ".aaa",
        },
        "b": {
            "dom": "#bbb",
        },
    }
    analyzer = Wappalyzer(categories=categories, technologies=technologies)

    resultA = await analyzer.analyze(webpageA)
    resultB = await analyzer.analyze(webpageB)

    assert resultA == {"a"}
    assert resultB == {"b"}


@pytest.mark.asyncio
async def test_analyze_dom_list():
    webpageA = WebPage(
        "https://www.delish.com/", '<html><p class="aaa">webpage a</p></html>', {}
    )
    webpageB = WebPage(
        "https://www.delish.com/", '<html><p id="bbb">webpage b</p></html>', {}
    )
    categories = {}
    technologies = {
        "a": {
            "dom": [".aaa", "[some-other-css-selector]"],
        },
        "b": {
            "dom": ["[some-other-css-selector]", "#bbb"],
        },
    }
    analyzer = Wappalyzer(categories=categories, technologies=technologies)
    resultA = await analyzer.analyze(webpageA)
    resultB = await analyzer.analyze(webpageB)

    assert resultA == {"a"}
    assert resultB == {"b"}


@pytest.mark.asyncio
async def test_analyze_dom_dict_text():
    webpageA = WebPage(
        "https://www.delish.com/", '<html><p class="aaa">webpage a</p></html>', {}
    )
    webpageB = WebPage(
        "https://www.delish.com/", '<html><p id="bbb">webpage b</p></html>', {}
    )
    categories = {}
    get_dom_val = lambda cat: {
        "#bbb": {
            "text": f"webpage\\ {cat}",
        },
        ".aaa": {
            "text": f"webpage\\ {cat}",
        },
    }
    technologies = {
        "a": {"dom": get_dom_val("a")},
        "b": {
            "dom": get_dom_val("b"),
        },
    }
    analyzer = Wappalyzer(categories=categories, technologies=technologies)

    resultA = await analyzer.analyze(webpageA)
    resultB = await analyzer.analyze(webpageB)

    assert resultA == {"a"}
    assert resultB == {"b"}


@pytest.mark.asyncio
async def test_analyze_dom_dict_exists():
    webpageA = WebPage(
        "https://www.delish.com/", '<html><p class="aaa">webpage a</p></html>', {}
    )
    webpageB = WebPage(
        "https://www.delish.com/", '<html><p id="bbb">webpage b</p></html>', {}
    )
    categories = {}
    get_dom_val = lambda cat: {
        f"#{cat*3}": {
            "exists": True,
        },
        f".{cat*3}": {
            "exists": True,
        },
    }
    technologies = {
        "a": {"dom": get_dom_val("a")},
        "b": {
            "dom": get_dom_val("b"),
        },
    }
    analyzer = Wappalyzer(categories=categories, technologies=technologies)

    resultA = await analyzer.analyze(webpageA)
    resultB = await analyzer.analyze(webpageB)

    assert resultA == {"a"}
    assert resultB == {"b"}


@pytest.mark.asyncio
async def test_analyze_dom_dict_attributes():
    webpageA = WebPage(
        "https://www.delish.com/",
        '<html><p class="aaa" onclick="webpageAScript()">webpage a</p></html>',
        {},
    )
    webpageB = WebPage(
        "https://www.delish.com/",
        '<html><p id="bbb" onclick="webpageBScript()">webpage b</p></html>',
        {},
    )
    categories = {}
    get_dom_val = lambda cat: {
        f"#{cat*3}": {
            "attributes": {"onclick": f"webpage{cat.upper()}Script.*"},
        },
        f".{cat*3}": {
            "attributes": {"onclick": f"webpage{cat.upper()}Script.*"},
        },
    }
    technologies = {
        "a": {"dom": get_dom_val("a")},
        "b": {
            "dom": get_dom_val("b"),
        },
    }
    analyzer = Wappalyzer(categories=categories, technologies=technologies)
    resultA = await analyzer.analyze(webpageA)
    resultB = await analyzer.analyze(webpageB)

    assert resultA == {"a"}
    assert resultB == {"b"}


def test_fingerprint():
    tech_fingerprint = Fingerprint(
        name="WordPress",
        **{
            "cats": [1, 11],
            "html": [],
            "icon": "WordPress.svg",
            "implies": ["PHP", "MySQL"],
            "meta": {
                "generator": [
                    "Whatever123",
                    "Whatever456",
                    "^WordPress ?([\\d.]+)?\\;version:\\1",
                    "Whatever",
                ]
            },
            "website": "https://wordpress.org",
        },
    )
    assert tech_fingerprint.meta["generator"][-2].version == "\\1"
    assert (
        tech_fingerprint.meta["generator"][-2].regex.pattern == "^WordPress ?([\\d.]+)?"
    )
