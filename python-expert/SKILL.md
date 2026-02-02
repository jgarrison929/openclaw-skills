---
name: python-expert
description: Use when writing Python 3.10+ code requiring type hints, async/await, testing with pytest, packaging, virtual environments, decorators, generators, or any Python development. Invoke for performance optimization, refactoring, or complex Python patterns.
triggers:
  - Python
  - pytest
  - asyncio
  - FastAPI
  - Django
  - Flask
  - pip
  - poetry
  - pyproject.toml
  - venv
  - type hints
  - decorator
  - generator
  - Pydantic
  - dataclass
role: specialist
scope: implementation
output-format: code
---

# Python Expert

Senior Python specialist with deep expertise in modern Python 3.10+, type systems, async programming, and production-grade application development.

## Role Definition

You are a senior Python engineer specializing in clean, performant, and idiomatic Python. You leverage modern language features (3.10+ pattern matching, type unions, dataclasses) and build production-ready applications with comprehensive testing, proper packaging, and performance awareness.

## Core Principles

1. **Type everything** — use `typing` extensively; prefer `X | None` over `Optional[X]` (3.10+)
2. **Explicit over implicit** — clear error handling, no silent failures
3. **Composition over inheritance** — protocols and dataclasses over deep class hierarchies
4. **Standard library first** — reach for third-party only when justified
5. **Test-driven** — pytest with fixtures, parametrize, and proper mocking
6. **Async where appropriate** — I/O-bound work benefits from asyncio; CPU-bound does not

---

## Project Structure

```
myproject/
├── pyproject.toml           # Modern packaging (PEP 621)
├── src/
│   └── myproject/
│       ├── __init__.py
│       ├── main.py
│       ├── models.py
│       ├── services/
│       │   ├── __init__.py
│       │   └── user_service.py
│       └── utils/
│           ├── __init__.py
│           └── helpers.py
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── unit/
│   │   └── test_models.py
│   └── integration/
│       └── test_services.py
├── .python-version          # pyenv version
└── README.md
```

## pyproject.toml (Modern Packaging)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "myproject"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.5",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=4.1",
    "ruff>=0.3",
    "mypy>=1.8",
]

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "A", "SIM", "TCH"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

---

## Type Hints — Modern Style

```python
# Python 3.10+ — use built-in generics and union syntax
def process_items(items: list[str], limit: int | None = None) -> dict[str, int]:
    ...

# Protocols for structural typing (duck typing with types)
from typing import Protocol, runtime_checkable

@runtime_checkable
class Repository(Protocol):
    def get(self, id: str) -> dict[str, Any]: ...
    def save(self, entity: dict[str, Any]) -> None: ...

# TypeVar and Generic for reusable types
from typing import TypeVar, Generic

T = TypeVar("T")

class Result(Generic[T]):
    def __init__(self, value: T | None = None, error: str | None = None):
        self._value = value
        self._error = error

    @property
    def is_ok(self) -> bool:
        return self._error is None

    def unwrap(self) -> T:
        if self._error is not None:
            raise ValueError(self._error)
        assert self._value is not None
        return self._value

    @classmethod
    def ok(cls, value: T) -> "Result[T]":
        return cls(value=value)

    @classmethod
    def err(cls, error: str) -> "Result[T]":
        return cls(error=error)

# TypedDict for structured dicts
from typing import TypedDict, NotRequired

class UserConfig(TypedDict):
    name: str
    email: str
    age: NotRequired[int]

# ParamSpec for decorator typing
from typing import ParamSpec, Callable
import functools

P = ParamSpec("P")

def retry(max_attempts: int = 3) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == max_attempts - 1:
                        raise
            raise RuntimeError("Unreachable")
        return wrapper
    return decorator
```

---

## Dataclasses and Pydantic

```python
from dataclasses import dataclass, field
from datetime import datetime

# Dataclasses for internal domain objects
@dataclass(frozen=True, slots=True)
class Money:
    amount: int  # cents
    currency: str = "USD"

    def __add__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError(f"Cannot add {self.currency} and {other.currency}")
        return Money(amount=self.amount + other.amount, currency=self.currency)

    def display(self) -> str:
        return f"${self.amount / 100:.2f} {self.currency}"

@dataclass
class Order:
    id: str
    items: list[str] = field(default_factory=list)
    total: Money = field(default_factory=lambda: Money(0))
    created_at: datetime = field(default_factory=datetime.utcnow)

# Pydantic for validation and serialization (API boundaries)
from pydantic import BaseModel, Field, field_validator, EmailStr

class CreateUserRequest(BaseModel):
    model_config = {"strict": True}

    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    age: int = Field(ge=0, le=150)

    @field_validator("name")
    @classmethod
    def name_must_be_titlecase(cls, v: str) -> str:
        return v.strip()

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    created_at: datetime
```

---

## Async / Await Patterns

```python
import asyncio
from collections.abc import AsyncIterator
import httpx

# Async context manager
class AsyncDBPool:
    async def __aenter__(self) -> "AsyncDBPool":
        self._pool = await create_pool(dsn="postgres://...")
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._pool.close()

# Async generator
async def stream_events(url: str) -> AsyncIterator[dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url) as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    yield json.loads(line[5:])

# Structured concurrency with TaskGroup (3.11+)
async def fetch_all_users(user_ids: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    async with asyncio.TaskGroup() as tg:
        for uid in user_ids:
            tg.create_task(_fetch_and_append(uid, results))

    return results

async def _fetch_and_append(uid: str, results: list[dict[str, Any]]) -> None:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.example.com/users/{uid}")
        results.append(resp.json())

# Semaphore for rate limiting
async def fetch_with_rate_limit(
    urls: list[str], max_concurrent: int = 10
) -> list[str]:
    semaphore = asyncio.Semaphore(max_concurrent)
    async with httpx.AsyncClient() as client:

        async def fetch(url: str) -> str:
            async with semaphore:
                resp = await client.get(url)
                return resp.text

        return await asyncio.gather(*[fetch(url) for url in urls])
```

---

## Decorators and Context Managers

```python
import functools
import time
import logging
from contextlib import contextmanager, asynccontextmanager
from collections.abc import Generator, AsyncGenerator

logger = logging.getLogger(__name__)

# Timing decorator with proper typing
def timed(func: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

# Context manager for temporary state
@contextmanager
def temporary_env(**env_vars: str) -> Generator[None, None, None]:
    import os
    old_values = {}
    for key, value in env_vars.items():
        old_values[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

# Async context manager
@asynccontextmanager
async def db_transaction(pool: Any) -> AsyncGenerator[Any, None]:
    conn = await pool.acquire()
    tx = await conn.begin()
    try:
        yield conn
        await tx.commit()
    except Exception:
        await tx.rollback()
        raise
    finally:
        await pool.release(conn)

# Class-based decorator with state
class CacheResult:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self.cache: dict[str, tuple[float, Any]] = {}

    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            key = str(args) + str(kwargs)
            now = time.time()
            if key in self.cache:
                cached_time, cached_value = self.cache[key]
                if now - cached_time < self.ttl:
                    return cached_value
            result = func(*args, **kwargs)
            self.cache[key] = (now, result)
            return result
        return wrapper
```

---

## Error Handling

```python
# Custom exception hierarchy
class AppError(Exception):
    """Base application error."""
    def __init__(self, message: str, code: str = "UNKNOWN"):
        super().__init__(message)
        self.code = code

class NotFoundError(AppError):
    def __init__(self, resource: str, id: str):
        super().__init__(f"{resource} with id={id} not found", code="NOT_FOUND")

class ValidationError(AppError):
    def __init__(self, field: str, message: str):
        super().__init__(f"Validation failed for {field}: {message}", code="VALIDATION")

# Pattern: Result type instead of exceptions for expected failures
def parse_config(path: str) -> Result[dict[str, Any]]:
    try:
        with open(path) as f:
            data = json.load(f)
        return Result.ok(data)
    except FileNotFoundError:
        return Result.err(f"Config file not found: {path}")
    except json.JSONDecodeError as e:
        return Result.err(f"Invalid JSON in {path}: {e}")

# ExceptionGroup handling (3.11+)
async def process_batch(items: list[str]) -> list[str]:
    results = []
    try:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(process_item(item)) for item in items]
    except* ValueError as eg:
        logger.error(f"Validation errors: {eg.exceptions}")
        raise
    except* IOError as eg:
        logger.error(f"IO errors: {eg.exceptions}")
        raise
    return [t.result() for t in tasks]
```

---

## Generators and Itertools

```python
from collections.abc import Iterator, Generator
import itertools

# Generator for memory-efficient processing
def read_large_file(path: str, chunk_size: int = 8192) -> Iterator[str]:
    with open(path) as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

# Generator with send/return
def running_average() -> Generator[float, float, str]:
    total = 0.0
    count = 0
    while True:
        value = yield total / count if count else 0.0
        if value is None:
            break
        total += value
        count += 1
    return f"Final average: {total / count:.2f}" if count else "No values"

# Practical itertools patterns
def batched(iterable: Iterator[T], n: int) -> Iterator[list[T]]:
    """Batch items into lists of size n. (Use itertools.batched in 3.12+)"""
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch

def flatten(nested: list[list[T]]) -> Iterator[T]:
    return itertools.chain.from_iterable(nested)
```

---

## Testing with pytest

```python
# conftest.py — shared fixtures
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_db() -> MagicMock:
    db = MagicMock()
    db.get_user.return_value = {"id": "1", "name": "Test User"}
    return db

@pytest.fixture
def async_mock_client() -> AsyncMock:
    client = AsyncMock()
    client.get.return_value.json.return_value = {"status": "ok"}
    return client

@pytest.fixture(scope="session")
def db_url() -> str:
    return "sqlite:///test.db"

# test_models.py — unit tests
import pytest
from myproject.models import Money, Order

class TestMoney:
    def test_add_same_currency(self) -> None:
        a = Money(100, "USD")
        b = Money(250, "USD")
        assert (a + b).amount == 350

    def test_add_different_currency_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot add"):
            Money(100, "USD") + Money(100, "EUR")

    def test_display(self) -> None:
        assert Money(1550, "USD").display() == "$15.50 USD"

    @pytest.mark.parametrize(
        "amount,expected",
        [
            (0, "$0.00 USD"),
            (1, "$0.01 USD"),
            (100, "$1.00 USD"),
            (9999, "$99.99 USD"),
        ],
    )
    def test_display_parametrized(self, amount: int, expected: str) -> None:
        assert Money(amount).display() == expected

# test_services.py — async and mocked tests
import pytest
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_fetch_user(async_mock_client: AsyncMock) -> None:
    from myproject.services.user_service import fetch_user

    with patch("myproject.services.user_service.httpx.AsyncClient") as mock:
        mock.return_value.__aenter__.return_value = async_mock_client
        result = await fetch_user("123")
        assert result["status"] == "ok"

# Snapshot testing with inline snapshots
def test_user_serialization() -> None:
    user = UserResponse(
        id="1", name="Alice", email="alice@example.com",
        created_at=datetime(2024, 1, 1)
    )
    assert user.model_dump() == {
        "id": "1",
        "name": "Alice",
        "email": "alice@example.com",
        "created_at": datetime(2024, 1, 1),
    }
```

---

## Virtual Environments

```bash
# Modern: use uv (fastest)
uv venv
uv pip install -e ".[dev]"

# Standard: venv + pip
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -e ".[dev]"

# Poetry
poetry init
poetry add httpx pydantic
poetry add --group dev pytest ruff mypy

# pyenv for Python version management
pyenv install 3.12.2
pyenv local 3.12.2
```

---

## Common Anti-Patterns

```python
# ❌ BAD: Mutable default argument
def append_to(item, target=[]):
    target.append(item)
    return target

# ✅ GOOD: Use None sentinel
def append_to(item: str, target: list[str] | None = None) -> list[str]:
    if target is None:
        target = []
    target.append(item)
    return target

# ❌ BAD: Bare except
try:
    do_something()
except:
    pass

# ✅ GOOD: Specific exceptions
try:
    do_something()
except (ValueError, KeyError) as e:
    logger.error(f"Expected error: {e}")
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    raise

# ❌ BAD: isinstance chains
if isinstance(shape, Circle):
    area = math.pi * shape.radius ** 2
elif isinstance(shape, Rectangle):
    area = shape.width * shape.height

# ✅ GOOD: Pattern matching (3.10+)
match shape:
    case Circle(radius=r):
        area = math.pi * r ** 2
    case Rectangle(width=w, height=h):
        area = w * h
    case _:
        raise ValueError(f"Unknown shape: {shape}")

# ❌ BAD: Global state
db_connection = None
def get_db():
    global db_connection
    if db_connection is None:
        db_connection = connect()
    return db_connection

# ✅ GOOD: Dependency injection
@dataclass
class UserService:
    db: Database
    cache: Cache

    async def get_user(self, user_id: str) -> User:
        if cached := await self.cache.get(f"user:{user_id}"):
            return cached
        user = await self.db.fetch_user(user_id)
        await self.cache.set(f"user:{user_id}", user, ttl=300)
        return user
```

---

## Performance Tips

```python
# Use __slots__ for memory-efficient classes
class Point:
    __slots__ = ("x", "y")
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

# Use lru_cache for expensive pure functions
from functools import lru_cache

@lru_cache(maxsize=256)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Use dict.get() instead of try/except KeyError
value = data.get("key", default_value)

# Prefer set for membership testing
valid_ids = frozenset(range(1000))
if user_id in valid_ids:  # O(1)
    ...

# Use str.join for string concatenation
parts = ["hello", "world", "foo"]
result = " ".join(parts)  # Not: result += part

# Profile before optimizing
# python -m cProfile -s cumulative script.py
# python -m timeit -s "setup" "expression"
```

---

## Logging Best Practices

```python
import logging
import structlog

# Standard library logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Use lazy formatting
logger.info("Processing user %s with %d items", user_id, len(items))

# Structured logging with structlog
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)
log = structlog.get_logger()
log.info("user_processed", user_id=user_id, item_count=len(items))
```

---

*Adapted from buildwithclaude by Dave Poon (MIT)*
