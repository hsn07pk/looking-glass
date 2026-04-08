import pytest
from httpx import ASGITransport, AsyncClient

from looking_glass.api.main import app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health(client):
    async with client as c:
        r = await c.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "ok" in data
        assert "frame_count" in data


@pytest.mark.asyncio
async def test_cameras(client):
    async with client as c:
        r = await c.get("/cameras")
        assert r.status_code == 200
        cams = r.json()
        assert isinstance(cams, list)


@pytest.mark.asyncio
async def test_search(client):
    async with client as c:
        r = await c.post("/search", json={"q": "truck", "top_k": 3})
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert "query" in data


@pytest.mark.asyncio
async def test_analytics(client):
    async with client as c:
        r = await c.post("/analytics/ask", json={"q": "how many objects?"})
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
