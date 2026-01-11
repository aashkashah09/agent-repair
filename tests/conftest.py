import json
from pathlib import Path

import pytest

from toolsmith.config import REPO_ROOT
from toolsmith.server.db import Database
from toolsmith.server.mcp_server import ToolServer
from toolsmith.server.schemas import SchemaSet
from toolsmith.server.seeding import load_defects

DOMAIN = REPO_ROOT / "data" / "domain"
CLEAN = REPO_ROOT / "data" / "schemas" / "clean"
SEEDED = REPO_ROOT / "data" / "schemas" / "seeded"
DEFECTS = REPO_ROOT / "data" / "defects" / "seeded_defects.json"
TASKS = REPO_ROOT / "data" / "tasks" / "tasks.jsonl"


@pytest.fixture
def db() -> Database:
    return Database.load(DOMAIN)


@pytest.fixture
def clean_schemas() -> SchemaSet:
    return SchemaSet.load(CLEAN)


@pytest.fixture
def seeded_schemas() -> SchemaSet:
    return SchemaSet.load(SEEDED)


@pytest.fixture
def defects():
    return load_defects(DEFECTS)


@pytest.fixture
def clean_server() -> ToolServer:
    return ToolServer.build(CLEAN, DOMAIN)


@pytest.fixture
def seeded_server() -> ToolServer:
    return ToolServer.build(SEEDED, DOMAIN, DEFECTS)


@pytest.fixture
def tasks_raw() -> list[dict]:
    return [json.loads(line) for line in Path(TASKS).read_text().splitlines() if line.strip()]
