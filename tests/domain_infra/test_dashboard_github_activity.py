

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from src.dashboard import github_activity as activity_module
from src.dashboard.dashboard_state import DashboardState
from src.dashboard.github_activity import (
    GITHUB_DEFAULT_HIGHLIGHT,
    GitHubActivityError,
    build_feed,
    fetch_activity,
    is_highlighted,
    latest_release_from_events,
    merge_events,
    repo_full_name,
)
from src.dashboard.routers import monitor as monitor_module
from src.dashboard.routers.monitor import MonitorRouter

PUSH_EVENT = {
    "id": "1001",
    "type": "PushEvent",
    "actor": {"login": "karolswdev"},
    "repo": {"id": 7, "name": "karolswdev/HoldSpeak"},
    "payload": {
        "ref": "refs/heads/main",
        "size": 2,
        "commits": [
            {"sha": "a1", "message": "Add TTS endpoint\n\nlong body"},
            {"sha": "b2", "message": "Fix queue"},
        ],
    },
    "created_at": "2026-10-28T07:20:36Z",
}

LLM_TRADER_PUSH = {
    "id": "1002",
    "type": "PushEvent",
    "actor": {"login": "qrak"},
    "repo": {"name": "qrak/LLM_trader"},
    "payload": {
        "ref": "refs/heads/master",
        "size": 1,
        "commits": [{"sha": "c3", "message": "feat: activity feed"}],
    },
    "created_at": "2026-10-27T09:12:47Z",
}

MERGED_PR = {
    "id": "1003",
    "type": "PullRequestEvent",
    "actor": {"login": "qrak"},
    "repo": {"name": "qrak/LLM_trader"},
    "payload": {
        "action": "closed",
        "number": 12,
        "pull_request": {
            "title": "Add GitHub activity panel",
            "merged": True,
            "state": "closed",
            "number": 12,
            "html_url": "https://github.com/qrak/LLM_trader/pull/12",
        },
    },
    "created_at": "2026-10-28T07:20:36Z",
}


RELEASE_EVENT = {
    "id": "3001",
    "type": "ReleaseEvent",
    "actor": {"login": "github-actions[bot]"},
    "repo": {"name": "qrak/LLM_trader"},
    "payload": {
        "action": "published",
        "release": {
            "name": "Latest build — 2026.10.28-0915 (b658750)",
            "tag_name": "build-2026.10.28-0915",
            "html_url": "https://github.com/qrak/LLM_trader/releases/tag/build-2026.10.28-0915",
        },
    },
    "created_at": "2026-10-28T09:15:44Z",
}


def test_push_uses_github_wording_and_first_commit_message():
    rows = build_feed([PUSH_EVENT])
    assert len(rows) == 1
    row = rows[0]
    assert row["headline"] == "karolswdev contributed to karolswdev/HoldSpeak"
    assert row["label"] == "push"
    assert row["type"] == "push"
    assert row["detail"] == "Add TTS endpoint"
    assert row["meta"] == "2 commits · main"
    assert row["repo_url"] == "https://github.com/karolswdev/HoldSpeak"
    assert row["url"] == "https://github.com/karolswdev/HoldSpeak"
    assert row["highlighted"] is False


def test_llm_trader_repository_is_highlighted_by_name():
    rows = build_feed([LLM_TRADER_PUSH], highlight="LLM_trader")
    assert rows[0]["highlighted"] is True
    assert rows[0]["meta"] == "1 commit · master"
    assert build_feed([LLM_TRADER_PUSH], highlight="somethingelse")[0]["highlighted"] is False


def test_highlight_matches_repository_name_not_owner():
    assert is_highlighted("someone/llm_trader", "LLM_trader") is True
    assert is_highlighted("qrak/LLM_trader_backup", "LLM_trader") is False
    assert is_highlighted("", "LLM_trader") is False
    assert is_highlighted("qrak/LLM_trader", "") is False


def test_merged_pull_request_says_merged_and_links_the_pr():
    row = build_feed([MERGED_PR])[0]
    assert row["headline"] == "qrak merged a pull request in qrak/LLM_trader"
    assert row["detail"] == "Add GitHub activity panel"
    assert row["meta"] == "#12 · merged"
    assert row["url"] == "https://github.com/qrak/LLM_trader/pull/12"
    assert row["label"] == "pull request"


def test_closed_unmerged_pull_request_is_not_called_merged():
    event = {
        "type": "PullRequestEvent",
        "actor": {"login": "qrak"},
        "repo": {"name": "qrak/LLM_trader"},
        "payload": {
            "action": "closed",
            "number": 4,
            "pull_request": {"title": "WIP", "merged": False, "state": "closed", "number": 4},
        },
    }
    row = build_feed([event])[0]
    assert row["headline"] == "qrak closed a pull request in qrak/LLM_trader"
    assert row["meta"] == "#4 · closed"


def test_issue_branch_tag_release_star_and_fork_wording():
    events = [
        {
            "type": "IssuesEvent",
            "actor": {"login": "qrak"},
            "repo": {"name": "qrak/LLM_trader"},
            "payload": {"action": "opened", "issue": {"title": "Panel jest pusty", "number": 3}},
        },
        {
            "type": "CreateEvent",
            "actor": {"login": "qrak"},
            "repo": {"name": "qrak/LLM_trader"},
            "payload": {"ref": "feature/activity", "ref_type": "branch"},
        },
        {
            "type": "CreateEvent",
            "actor": {"login": "qrak"},
            "repo": {"name": "qrak/LLM_trader"},
            "payload": {"ref": "v1.2.0", "ref_type": "tag"},
        },
        {
            "type": "CreateEvent",
            "actor": {"login": "qrak"},
            "repo": {"name": "qrak/NEW_repo"},
            "payload": {"ref": None, "ref_type": "repository"},
        },
        {
            "type": "ReleaseEvent",
            "actor": {"login": "qrak"},
            "repo": {"name": "qrak/LLM_trader"},
            "payload": {"action": "published", "release": {"name": "v1.2.0", "tag_name": "v1.2.0"}},
        },
        {
            "type": "WatchEvent",
            "actor": {"login": "qrak"},
            "repo": {"name": "akitaonrails/ai-memory"},
            "payload": {"action": "started"},
        },
        {
            "type": "ForkEvent",
            "actor": {"login": "qrak"},
            "repo": {"name": "someorg/lib"},
            "payload": {"forkee": {"full_name": "qrak/lib"}},
        },
    ]
    rows = build_feed(events)
    assert [row["headline"] for row in rows] == [
        "qrak opened an issue in qrak/LLM_trader",
        "qrak created a branch in qrak/LLM_trader",
        "qrak created a tag in qrak/LLM_trader",
        "qrak created a repository",
        "qrak published a release in qrak/LLM_trader",
        "qrak starred akitaonrails/ai-memory",
        "qrak forked someorg/lib",
    ]
    assert [row["detail"] for row in rows][:4] == [
        "Panel jest pusty",
        "feature/activity",
        "v1.2.0",
        "",
    ]
    assert rows[6]["detail"] == "qrak/lib"
    assert rows[0]["meta"] == "#3"


def test_unknown_event_type_still_renders_a_readable_row():
    row = build_feed([{
        "type": "GollumEvent",
        "actor": {"login": "qrak"},
        "repo": {"name": "qrak/LLM_trader"},
        "payload": {},
    }])[0]
    assert row["headline"] == "qrak gollum in qrak/LLM_trader"
    assert row["label"] == "gollum"


def test_real_public_api_shapes_without_commits_or_pr_titles():

    push = {
        "id": "2001",
        "type": "PushEvent",
        "actor": {"login": "qrak"},
        "repo": {"name": "qrak/LLM_trader"},
        "payload": {
            "repository_id": 940742140,
            "push_id": 47937272423,
            "ref": "refs/heads/master",
            "head": "56088bb12a3ae20902ac38c308bea34d5e6e4eb2",
            "before": "bbd0cbe200207172105d31d32dcb63e55eb6b808",
        },
        "created_at": "2026-10-27T09:12:47Z",
    }
    pull_request = {
        "id": "2002",
        "type": "PullRequestEvent",
        "actor": {"login": "qrak"},
        "repo": {"name": "qrak/LLM_trader"},
        "payload": {
            "action": "opened",
            "number": 29,
            "pull_request": {
                "url": "https://api.github.com/repos/qrak/LLM_trader/pulls/29",
                "id": 5046754835,
                "number": 29,
                "head": {"ref": "ci-probe-cancel", "sha": "9859b60"},
                "base": {"ref": "master", "sha": "506e35d"},
            },
        },
        "created_at": "2026-10-28T07:20:36Z",
    }
    push_row, pr_row = build_feed([push, pull_request])

    assert push_row["detail"] == "commit 56088bb"
    assert push_row["meta"] == "master"
    assert push_row["url"] == "https://github.com/qrak/LLM_trader/commit/56088bb12a3ae20902ac38c308bea34d5e6e4eb2"

    assert pr_row["headline"] == "qrak opened a pull request in qrak/LLM_trader"
    assert pr_row["detail"] == "ci-probe-cancel → master"
    assert pr_row["meta"] == "#29 · opened"
    assert pr_row["url"] == "https://github.com/qrak/LLM_trader/pull/29"


def test_issue_comment_falls_back_to_the_comment_body():
    event = {
        "type": "IssueCommentEvent",
        "actor": {"login": "qrak"},
        "repo": {"name": "qrak/LLM_trader"},
        "payload": {"action": "created", "comment": {"body": "Fixed in 506e35d"}},
    }
    row = build_feed([event])[0]
    assert row["headline"] == "qrak commented on an issue in qrak/LLM_trader"
    assert row["detail"] == "Fixed in 506e35d"


def test_malformed_entries_are_skipped_and_limit_is_respected():
    events = [
        {"type": "PushEvent", "repo": {"name": "qrak/LLM_trader"}, "payload": {"commits": []}},
        {"type": "", "repo": {"name": "qrak/LLM_trader"}},
        {"type": "PushEvent", "repo": {}},
        "not-an-event",
        LLM_TRADER_PUSH,
        PUSH_EVENT,
    ]
    rows = build_feed(events, limit=2)
    assert len(rows) == 2
    assert rows[0]["headline"] == "someone contributed to qrak/LLM_trader"
    assert rows[0]["detail"] == ""
    assert rows[1]["headline"] == "qrak contributed to qrak/LLM_trader"
    assert rows[1]["detail"] == "feat: activity feed"


def test_non_list_payload_is_refused():
    with pytest.raises(GitHubActivityError):
        build_feed({"message": "Bad credentials"})


def _config(user="qrak", token=""):
    return SimpleNamespace(GITHUB_USER=user, GITHUB_TOKEN=token)


def _client(monkeypatch, fetch, state=None, config=None):
    monkeypatch.setattr(monitor_module, "fetch_activity", fetch)
    app = FastAPI()
    router = MonitorRouter(
        config=config or _config(),
        logger=logging.getLogger("test.github_activity"),
        dashboard_state=state or DashboardState(),
        analysis_engine=None,
        rag_engine=None,
    )
    app.include_router(router.router)
    return TestClient(app)


def test_repo_full_name_is_the_account_plus_the_highlighted_repository():
    assert repo_full_name("qrak", "LLM_trader") == "qrak/LLM_trader"
    assert repo_full_name("qrak", "  ") == ""
    assert repo_full_name("  ", "LLM_trader") == ""


def test_merge_events_dedupes_the_shared_events_and_sorts_newest_first():
    merged = merge_events([RELEASE_EVENT, LLM_TRADER_PUSH], [LLM_TRADER_PUSH, PUSH_EVENT])
    assert [event["id"] for event in merged] == ["3001", "1001", "1002"]


def test_merge_events_ignores_junk_groups_and_entries():
    assert merge_events(None, "nope", [None, 5, LLM_TRADER_PUSH]) == [LLM_TRADER_PUSH]


def test_latest_release_is_the_newest_release_of_the_highlighted_repository():
    older = {
        **RELEASE_EVENT,
        "id": "3000",
        "created_at": "2026-10-20T20:48:16Z",
        "payload": {"action": "published", "release": {"name": "v1.1.5", "tag_name": "v1.1.5"}},
    }
    other_repo = {
        **RELEASE_EVENT,
        "id": "3002",
        "repo": {"name": "someone/else"},
        "created_at": "2026-10-28T10:00:00Z",
    }
    release = latest_release_from_events(
        [older, other_repo, RELEASE_EVENT], repo="qrak/LLM_trader"
    )
    assert release == {
        "tag": "build-2026.10.28-0915",
        "name": "Latest build — 2026.10.28-0915 (b658750)",
        "url": "https://github.com/qrak/LLM_trader/releases/tag/build-2026.10.28-0915",
        "published_at": "2026-10-28T09:15:44Z",
        "actor": "github-actions[bot]",
    }


def test_latest_release_is_absent_without_a_release_event():
    assert latest_release_from_events([LLM_TRADER_PUSH], repo="qrak/LLM_trader") is None
    assert latest_release_from_events("nope") is None


def test_fetch_activity_merges_the_repository_stream_and_exposes_the_bot_release(monkeypatch):
    calls = {}

    async def account(user, *, token=None, limit=30, session=None):
        calls["account"] = {"user": user, "limit": limit}
        return [LLM_TRADER_PUSH]

    async def project(repo, *, token=None, limit=30, session=None):
        calls["project"] = repo
        return [RELEASE_EVENT, LLM_TRADER_PUSH]

    async def latest(*args, **kwargs):
        raise AssertionError("the event stream already carries the release")

    monkeypatch.setattr(activity_module, "fetch_user_events", account)
    monkeypatch.setattr(activity_module, "fetch_repo_events", project)
    monkeypatch.setattr(activity_module, "fetch_latest_release", latest)

    result = asyncio.run(fetch_activity("qrak", highlight="LLM_trader", limit=30))

    assert calls["account"] == {"user": "qrak", "limit": 30}
    assert calls["project"] == "qrak/LLM_trader"
    assert [event["id"] for event in result["events"]] == ["3001", "1002"]
    assert result["latest_release"]["tag"] == "build-2026.10.28-0915"


def test_fetch_activity_falls_back_to_the_releases_endpoint(monkeypatch):
    async def account(user, *, token=None, limit=30, session=None):
        return [LLM_TRADER_PUSH]

    async def project(repo, *, token=None, limit=30, session=None):
        return [LLM_TRADER_PUSH]

    async def latest(repo, *, token=None, session=None):
        return {
            "tag": "v1.1.5",
            "name": "v1.1.5",
            "url": "https://github.com/qrak/LLM_trader/releases/tag/v1.1.5",
            "published_at": "2026-10-20T20:48:16Z",
            "actor": "qrak",
        }

    monkeypatch.setattr(activity_module, "fetch_user_events", account)
    monkeypatch.setattr(activity_module, "fetch_repo_events", project)
    monkeypatch.setattr(activity_module, "fetch_latest_release", latest)

    result = asyncio.run(fetch_activity("qrak", highlight="LLM_trader"))
    assert result["latest_release"]["tag"] == "v1.1.5"


def test_a_missing_repository_stream_costs_its_rows_only(monkeypatch):
    async def account(user, *, token=None, limit=30, session=None):
        return [LLM_TRADER_PUSH]

    async def project(repo, *, token=None, limit=30, session=None):
        raise GitHubActivityError("GitHub repository 'qrak/LLM_trader' was not found")

    async def latest(repo, *, token=None, session=None):
        raise GitHubActivityError("no releases")

    monkeypatch.setattr(activity_module, "fetch_user_events", account)
    monkeypatch.setattr(activity_module, "fetch_repo_events", project)
    monkeypatch.setattr(activity_module, "fetch_latest_release", latest)

    result = asyncio.run(fetch_activity("qrak", highlight="LLM_trader"))
    assert [event["id"] for event in result["events"]] == ["1002"]
    assert result["latest_release"] is None


def test_the_account_stream_failure_still_surfaces(monkeypatch):
    async def account(user, *, token=None, limit=30, session=None):
        raise GitHubActivityError("GitHub API answered HTTP 500")

    async def project(repo, *, token=None, limit=30, session=None):
        return [RELEASE_EVENT]

    monkeypatch.setattr(activity_module, "fetch_user_events", account)
    monkeypatch.setattr(activity_module, "fetch_repo_events", project)

    with pytest.raises(GitHubActivityError, match="500"):
        asyncio.run(fetch_activity("qrak", highlight="LLM_trader"))


def test_fetch_activity_refuses_an_empty_account():
    with pytest.raises(GitHubActivityError):
        asyncio.run(fetch_activity("  "))


def test_endpoint_returns_normalized_rows_and_caches_one_page(monkeypatch):
    calls = []

    async def fake_fetch(user, *, highlight=GITHUB_DEFAULT_HIGHLIGHT, token=None, limit=30):
        calls.append({"user": user, "token": token, "limit": limit, "highlight": highlight})
        return {
            "events": [RELEASE_EVENT, PUSH_EVENT, LLM_TRADER_PUSH],
            "latest_release": latest_release_from_events([RELEASE_EVENT], repo="qrak/LLM_trader"),
        }

    client = _client(monkeypatch, fake_fetch)
    first = client.get("/api/monitor/github_activity").json()
    second = client.get("/api/monitor/github_activity").json()

    assert first["user"] == "qrak"
    assert first["repo"] == "qrak/LLM_trader"
    assert first["count"] == 3
    assert first["highlight"] == "LLM_trader"
    assert first["stale"] is False
    assert first["error"] is None
    assert first["fetched_at"]
    assert first["events"][0]["label"] == "release"
    assert first["events"][0]["headline"] == (
        "github-actions[bot] published a release in qrak/LLM_trader"
    )
    assert first["events"][0]["meta"] == "build-2026.10.28-0915"
    assert first["events"][2]["highlighted"] is True
    assert first["latest_release"]["tag"] == "build-2026.10.28-0915"
    assert second["fetched_at"] == first["fetched_at"]
    assert len(calls) == 1
    assert calls[0]["limit"] == 100
    assert calls[0]["highlight"] == "LLM_trader"


def test_small_limit_does_not_truncate_the_cached_page(monkeypatch):
    calls = []

    async def fake_fetch(user, *, highlight=GITHUB_DEFAULT_HIGHLIGHT, token=None, limit=30):
        calls.append(limit)
        return {
            "events": [RELEASE_EVENT, PUSH_EVENT, LLM_TRADER_PUSH],
            "latest_release": None,
        }

    client = _client(monkeypatch, fake_fetch)
    narrow = client.get("/api/monitor/github_activity?limit=2").json()
    wide = client.get("/api/monitor/github_activity?limit=30").json()

    assert narrow["count"] == 2
    assert wide["count"] == 3
    assert len(calls) == 1


def test_endpoint_passes_the_token_and_clamps_the_limit(monkeypatch):
    calls = []

    async def fake_fetch(user, *, highlight=GITHUB_DEFAULT_HIGHLIGHT, token=None, limit=30):
        calls.append({"token": token, "limit": limit})
        return {"events": [], "latest_release": None}

    client = _client(monkeypatch, fake_fetch, config=_config(token="ghp_secret"))
    body = client.get("/api/monitor/github_activity?limit=500").json()

    assert calls[0]["limit"] == 100
    assert calls[0]["token"] == "ghp_secret"
    assert body["events"] == []
    assert body["latest_release"] is None


def test_refresh_bypasses_the_cache(monkeypatch):
    calls = []

    async def fake_fetch(user, *, highlight=GITHUB_DEFAULT_HIGHLIGHT, token=None, limit=30):
        calls.append(user)
        return {"events": [PUSH_EVENT], "latest_release": None}

    client = _client(monkeypatch, fake_fetch)
    client.get("/api/monitor/github_activity")
    client.get("/api/monitor/github_activity?refresh=true")

    assert len(calls) == 2


def test_unconfigured_user_answers_with_a_hint_and_no_network_call(monkeypatch):
    async def fake_fetch(user, *, highlight=GITHUB_DEFAULT_HIGHLIGHT, token=None, limit=30):
        raise AssertionError("fetch must not run without GITHUB_USER")

    client = _client(monkeypatch, fake_fetch, config=_config(user="  "))
    body = client.get("/api/monitor/github_activity").json()

    assert body["user"] is None
    assert body["repo"] is None
    assert body["events"] == []
    assert "GITHUB_USER" in body["error"]


def test_github_outage_serves_the_last_page_as_stale(monkeypatch):
    async def ok_fetch(user, *, highlight=GITHUB_DEFAULT_HIGHLIGHT, token=None, limit=30):
        return {
            "events": [PUSH_EVENT],
            "latest_release": {"tag": "v1.1.5", "name": "v1.1.5", "url": "", "published_at": ""},
        }

    state = DashboardState()
    client = _client(monkeypatch, ok_fetch, state=state)
    assert client.get("/api/monitor/github_activity").json()["count"] == 1

    async def broken_fetch(user, *, highlight=GITHUB_DEFAULT_HIGHLIGHT, token=None, limit=30):
        raise GitHubActivityError("GitHub API answered HTTP 503")

    monkeypatch.setattr(monitor_module, "GITHUB_ACTIVITY_CACHE_SECONDS", 0.0)
    client = _client(monkeypatch, broken_fetch, state=state)
    body = client.get("/api/monitor/github_activity").json()

    assert body["count"] == 1
    assert body["stale"] is True
    assert "503" in body["error"]
    assert body["latest_release"]["tag"] == "v1.1.5"
    assert body["events"][0]["headline"] == "karolswdev contributed to karolswdev/HoldSpeak"


def test_github_outage_without_any_cache_reports_the_error(monkeypatch):
    async def broken_fetch(user, *, highlight=GITHUB_DEFAULT_HIGHLIGHT, token=None, limit=30):
        raise GitHubActivityError("GitHub API rate limit reached")

    client = _client(monkeypatch, broken_fetch)
    body = client.get("/api/monitor/github_activity").json()

    assert body["events"] == []
    assert body["count"] == 0
    assert "rate limit" in body["error"]


def test_activity_endpoint_is_a_heavy_path():
    from src.dashboard.server import DashboardServer

    assert DashboardServer._is_heavy_path("/api/monitor/github_activity") is True
    assert DashboardServer._is_heavy_path("/api/monitor/news") is True
