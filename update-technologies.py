import argparse
import datetime
import pprint
from pathlib import Path
import asyncio
import aiofiles

from Wappalyzer.fingerprint import get_latest_tech_data


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("technologies_py_module", action="store", type=Path, nargs=1)
    return parser


TECHNOLOGIES_PY_MOD_DOC = """
This module contains the raw fingerprints data. It has been automatically generated on the %s.
"""


async def main():
    args = get_parser().parse_args()
    tech_data = await get_latest_tech_data()
    text = f"TECHNOLOGIES_DATA = {pprint.pformat(tech_data, indent=4, width=120)}"
    text = f'"""{TECHNOLOGIES_PY_MOD_DOC % datetime.datetime.now().isoformat()}"""\n\n{text}'
    async with aiofiles.open(
        args.technologies_py_module[0], mode="w", encoding="utf-8"
    ) as f:
        await f.write(text)


if __name__ == "__main__":
    asyncio.run(main())
