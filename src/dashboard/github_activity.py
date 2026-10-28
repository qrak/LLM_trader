"""GitHub activity feed rows for the dashboard.

The dashboard echoes the public GitHub event stream with GitHub's own wording
("qrak contributed to qrak/LLM_trader"), so the panel is a faithful mirror of
github.com instead of a second, differently worded activity log. Three parts:

- ``fetch_activity`` makes the network calls: the account's public events, the
  highlighted repository's events and, when the feed carries no release, the
  latest release of that repository;
- ``merge_events`` folds the two streams into one newest-first list;
- ``build_feed`` turns raw events into flat rows the panel renders.

The account stream alone is not enough: GitHub reports an event under the actor
that caused it, so a release published by ``github-actions[bot]`` (every build
tag) and a collaborator's issue never appear in ``/users/<login>/events``. The
per-repository stream carries both, which is why the panel requests it too.

Everything except the HTTP calls is pure, so every wording, merge and highlight
rule is covered by tests without touching the network.
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

import aiohttp

GITHUB_API_ROOT = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"
GITHUB_FETCH_TIMEOUT_SECONDS = 10
GITHUB_ACTIVITY_CACHE_SECONDS = 300.0
GITHUB_MAX_EVENTS = 100
GITHUB_DEFAULT_HIGHLIGHT = "LLM_trader"
GITHUB_DETAIL_MAX_CHARS = 200
GITHUB_ACTOR_FALLBACK = "someone"

_EVENT_LABELS: dict[str, str] = {
    "PushEvent": "push",
    "PullRequestEvent": "pull request",
    "PullRequestReviewEvent": "review",
    "PullRequestReviewCommentEvent": "review comment",
    "IssuesEvent": "issue",
    "IssueCommentEvent": "comment",
    "CommitCommentEvent": "comment",
    "CreateEvent": "create",
    "DeleteEvent": "delete",
    "WatchEvent": "star",
    "ForkEvent": "fork",
    "ReleaseEvent": "release",
    "PublicEvent": "public",
    "MemberEvent": "member",
}

_PR_VERBS = {"opened": "opened", "closed": "closed", "reopened": "reopened"}
_ISSUE_VERBS = {"opened": "opened", "closed": "closed", "reopened": "reopened"}


class GitHubActivityError(RuntimeError):
    """Raised when the public event stream cannot be fetched or parsed."""


def _clean(value: Any) -> str:
    """Return a stripped single-line string for a possibly missing payload field."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _shorten(text: str, limit: int = GITHUB_DETAIL_MAX_CHARS) -> str:
    """Trim an arbitrary commit or issue title to a single feed-safe line."""
    cleaned = _clean(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _repo_of(event: dict[str, Any]) -> str:
    repo = event.get("repo")
    if not isinstance(repo, dict):
        return ""
    return _clean(repo.get("name"))


def _actor_of(event: dict[str, Any]) -> str:
    actor = event.get("actor")
    if not isinstance(actor, dict):
        return GITHUB_ACTOR_FALLBACK
    login = _clean(actor.get("login"))
    return login or GITHUB_ACTOR_FALLBACK


def _payload_of(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def is_highlighted(repo: str, highlight: str) -> bool:
    """Return True when the repository name matches the highlighted project."""
    if not repo or not highlight:
        return False
    name = repo.rsplit("/", 1)[-1]
    return name.casefold() == highlight.casefold()


def _commit_branch(payload: dict[str, Any]) -> str:
    ref = _clean(payload.get("ref"))
    if ref.startswith("refs/heads/"):
        return ref.removeprefix("refs/heads/")
    if ref.startswith("refs/tags/"):
        return ref.removeprefix("refs/tags/")
    return ref


def _commit_subject(message: Any) -> str:
    """Return a commit's first line, the way GitHub shows it in the feed."""
    if not isinstance(message, str):
        return ""
    stripped = message.strip()
    if not stripped:
        return ""
    return _shorten(stripped.splitlines()[0])


def _push_detail(payload: dict[str, Any]) -> tuple[str, str]:
    """Describe a push. The public events API usually strips the commit list."""
    commits = payload.get("commits")
    messages: list[str] = []
    if isinstance(commits, list):
        for commit in commits:
            if isinstance(commit, dict):
                message = _commit_subject(commit.get("message"))
                if message:
                    messages.append(message)
    head = _clean(payload.get("head"))
    if messages:
        detail = messages[0]
    elif head:
        detail = f"commit {head[:7]}"
    else:
        detail = ""
    size = payload.get("size")
    if isinstance(size, int) and size > 0:
        count = size
    else:
        count = len(messages) if messages else 0
    pieces = [f"{count} commit" if count == 1 else f"{count} commits"] if count else []
    branch = _commit_branch(payload)
    if branch:
        pieces.append(branch)
    return detail, " · ".join(pieces)


def _headline(actor: str, event_type: str, repo: str, payload: dict[str, Any]) -> str:
    """Render the GitHub-style sentence for one event."""
    action = _clean(payload.get("action"))
    if event_type == "PushEvent":
        return f"{actor} contributed to {repo}"
    if event_type == "WatchEvent":
        return f"{actor} starred {repo}"
    if event_type == "ForkEvent":
        return f"{actor} forked {repo}"
    if event_type == "PublicEvent":
        return f"{actor} made {repo} public"
    if event_type == "MemberEvent":
        return f"{actor} updated collaborators on {repo}"
    if event_type == "CreateEvent":
        ref_type = _clean(payload.get("ref_type")) or "ref"
        if ref_type == "repository":
            return f"{actor} created a repository"
        return f"{actor} created a {ref_type} in {repo}"
    if event_type == "DeleteEvent":
        ref_type = _clean(payload.get("ref_type")) or "ref"
        return f"{actor} deleted a {ref_type} in {repo}"
    if event_type == "PullRequestEvent":
        pull_request = payload.get("pull_request")
        merged = isinstance(pull_request, dict) and bool(pull_request.get("merged"))
        if action == "closed" and merged:
            return f"{actor} merged a pull request in {repo}"
        verb = _PR_VERBS.get(action, action or "updated")
        return f"{actor} {verb} a pull request in {repo}"
    if event_type == "PullRequestReviewEvent":
        return f"{actor} reviewed a pull request in {repo}"
    if event_type == "PullRequestReviewCommentEvent":
        return f"{actor} commented on a pull request in {repo}"
    if event_type == "IssuesEvent":
        verb = _ISSUE_VERBS.get(action, action or "updated")
        return f"{actor} {verb} an issue in {repo}"
    if event_type == "IssueCommentEvent":
        return f"{actor} commented on an issue in {repo}"
    if event_type == "CommitCommentEvent":
        return f"{actor} commented on a commit in {repo}"
    if event_type == "ReleaseEvent":
        return f"{actor} published a release in {repo}"
    label = event_type.removesuffix("Event").lower() or "event"
    return f"{actor} {label} in {repo}"


def _pr_branches(pull_request: dict[str, Any]) -> str:
    head = pull_request.get("head")
    base = pull_request.get("base")
    head_ref = _clean(head.get("ref")) if isinstance(head, dict) else ""
    base_ref = _clean(base.get("ref")) if isinstance(base, dict) else ""
    if head_ref and base_ref:
        return f"{head_ref} → {base_ref}"
    return head_ref or base_ref


def _detail(event_type: str, payload: dict[str, Any]) -> tuple[str, str]:
    """Return (detail line, meta line) for one event."""
    if event_type == "PushEvent":
        return _push_detail(payload)
    if event_type == "PullRequestEvent":
        pull_request = payload.get("pull_request")
        if not isinstance(pull_request, dict):
            return "", ""
        number = pull_request.get("number") or payload.get("number")
        merged = bool(pull_request.get("merged"))
        state = "merged" if merged else _clean(pull_request.get("state")) or _clean(payload.get("action"))
        meta_bits = [f"#{number}" if number else "", state]
        title = _shorten(pull_request.get("title") or "")
        return title or _pr_branches(pull_request), " · ".join(bit for bit in meta_bits if bit)
    if event_type == "IssuesEvent":
        issue = payload.get("issue")
        if not isinstance(issue, dict):
            return "", ""
        return _shorten(issue.get("title") or ""), f"#{issue.get('number')}" if issue.get("number") else ""
    if event_type in {"CreateEvent", "DeleteEvent"}:
        return _clean(payload.get("ref")), _clean(payload.get("ref_type"))
    if event_type == "ReleaseEvent":
        release = payload.get("release")
        if not isinstance(release, dict):
            return "", ""
        tag = _clean(release.get("tag_name"))
        return _shorten(release.get("name") or ""), tag
    if event_type in {"IssueCommentEvent", "PullRequestReviewEvent", "PullRequestReviewCommentEvent"}:
        pull_request = payload.get("pull_request")
        if isinstance(pull_request, dict):
            return _shorten(pull_request.get("title") or "") or _pr_branches(pull_request), ""
        issue = payload.get("issue")
        if isinstance(issue, dict):
            return _shorten(issue.get("title") or ""), ""
        comment = payload.get("comment")
        if isinstance(comment, dict):
            return _shorten(comment.get("body") or ""), ""
        return "", ""
    if event_type == "ForkEvent":
        forkee = payload.get("forkee")
        if isinstance(forkee, dict):
            return _clean(forkee.get("full_name")), ""
    if event_type == "MemberEvent":
        member = payload.get("member")
        if isinstance(member, dict):
            return _clean(member.get("login")), ""
    return "", ""


def _event_url(event: dict[str, Any]) -> str:
    payload = _payload_of(event)
    repo = _repo_of(event)
    pull_request = payload.get("pull_request")
    if isinstance(pull_request, dict):
        url = pull_request.get("html_url")
        if isinstance(url, str) and url:
            return url
        number = pull_request.get("number") or payload.get("number")
        if repo and number:
            return f"https://github.com/{repo}/pull/{number}"
    issue = payload.get("issue")
    if isinstance(issue, dict):
        url = issue.get("html_url")
        if isinstance(url, str) and url:
            return url
        if repo and issue.get("number"):
            return f"https://github.com/{repo}/issues/{issue.get('number')}"
    release = payload.get("release")
    if isinstance(release, dict):
        url = release.get("html_url")
        if isinstance(url, str) and url:
            return url
    head = _clean(payload.get("head"))
    if repo and head:
        return f"https://github.com/{repo}/commit/{head}"
    return f"https://github.com/{repo}" if repo else ""


def build_feed(
    events: Any,
    *,
    highlight: str = GITHUB_DEFAULT_HIGHLIGHT,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Turn a raw event list into dashboard rows, newest event first.

    Unknown event types and malformed entries are skipped rather than rendered as
    empty rows, so GitHub adding a new type cannot blank the panel.
    """
    if not isinstance(events, list):
        raise GitHubActivityError("Unexpected GitHub response: expected a list of events")
    rows: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = _clean(event.get("type"))
        repo = _repo_of(event)
        if not event_type or not repo:
            continue
        payload = _payload_of(event)
        detail, meta = _detail(event_type, payload)
        rows.append({
            "id": _clean(event.get("id")),
            "type": event_type.removesuffix("Event").lower() or "event",
            "label": _EVENT_LABELS.get(event_type, event_type.removesuffix("Event").lower()),
            "repo": repo,
            "repo_url": f"https://github.com/{repo}",
            "headline": _headline(_actor_of(event), event_type, repo, payload),
            "detail": detail,
            "meta": meta,
            "url": _event_url(event),
            "created_at": _clean(event.get("created_at")),
            "highlighted": is_highlighted(repo, highlight),
        })
        if len(rows) >= limit:
            break
    return rows


def repo_full_name(user: str, highlight: str) -> str:
    """Return the highlighted repository as ``owner/name`` (the account owns it)."""
    login = _clean(user)
    name = _clean(highlight)
    if not login or not name:
        return ""
    return f"{login}/{name}"


def _quote_repo(repo: str) -> str:
    return quote(repo, safe="/")


def _request_headers(token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "User-Agent": "llm-trader-dashboard",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _request_json(
    session: aiohttp.ClientSession,
    url: str,
    *,
    token: str | None,
    params: dict[str, Any] | None = None,
    not_found: str | None = None,
) -> Any:
    """GET one GitHub endpoint, translating every failure into ``GitHubActivityError``."""
    timeout = aiohttp.ClientTimeout(total=GITHUB_FETCH_TIMEOUT_SECONDS)
    try:
        async with session.get(
            url, headers=_request_headers(token), params=params, timeout=timeout
        ) as response:
            if response.status == 404:
                raise GitHubActivityError(not_found or f"GitHub returned HTTP 404 for {url}")
            if response.status in {403, 429}:
                raise GitHubActivityError(
                    "GitHub API rate limit reached — set GITHUB_TOKEN for a higher quota"
                )
            if response.status != 200:
                raise GitHubActivityError(f"GitHub API answered HTTP {response.status}")
            return await response.json(content_type=None)
    except aiohttp.ClientError as exc:
        raise GitHubActivityError(f"GitHub request failed: {exc}") from exc
    except TimeoutError as exc:
        raise GitHubActivityError("GitHub request timed out") from exc


def _event_list(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise GitHubActivityError("Unexpected GitHub response: expected a list of events")
    return [event for event in payload if isinstance(event, dict)]


async def fetch_user_events(
    user: str,
    *,
    token: str | None = None,
    limit: int = 30,
    session: aiohttp.ClientSession | None = None,
) -> list[dict[str, Any]]:
    """Fetch one page of the account's public GitHub events."""
    login = _clean(user)
    if not login:
        raise GitHubActivityError("GitHub user is not configured")
    per_page = max(1, min(limit, GITHUB_MAX_EVENTS))
    url = f"{GITHUB_API_ROOT}/users/{quote(login, safe='')}/events/public"
    owns_session = session is None
    active_session = session or aiohttp.ClientSession()
    try:
        payload = await _request_json(
            active_session,
            url,
            token=token,
            params={"per_page": per_page},
            not_found=f"GitHub user '{login}' was not found",
        )
    finally:
        if owns_session:
            await active_session.close()
    return _event_list(payload)


async def fetch_repo_events(
    repo: str,
    *,
    token: str | None = None,
    limit: int = 30,
    session: aiohttp.ClientSession | None = None,
) -> list[dict[str, Any]]:
    """Fetch one page of a repository's events.

    Unlike the account stream this one carries events caused by anybody: the
    ``github-actions[bot]`` release on every build tag, and collaborators' issues.
    """
    name = _clean(repo)
    if "/" not in name:
        return []
    per_page = max(1, min(limit, GITHUB_MAX_EVENTS))
    url = f"{GITHUB_API_ROOT}/repos/{_quote_repo(name)}/events"
    owns_session = session is None
    active_session = session or aiohttp.ClientSession()
    try:
        payload = await _request_json(
            active_session,
            url,
            token=token,
            params={"per_page": per_page},
            not_found=f"GitHub repository '{name}' was not found",
        )
    finally:
        if owns_session:
            await active_session.close()
    return _event_list(payload)


def _release_row(
    release: dict[str, Any], *, published_at: str, actor: str = ""
) -> dict[str, Any]:
    return {
        "tag": _clean(release.get("tag_name")),
        "name": _shorten(release.get("name") or ""),
        "url": _clean(release.get("html_url")),
        "published_at": published_at,
        "actor": _clean(actor),
    }


def latest_release_from_events(events: Any, *, repo: str = "") -> dict[str, Any] | None:
    """Return the newest release carried by an event stream, if any."""
    if not isinstance(events, list):
        return None
    wanted = _clean(repo).casefold()
    newest: dict[str, Any] | None = None
    for event in events:
        if not isinstance(event, dict) or _clean(event.get("type")) != "ReleaseEvent":
            continue
        if wanted and _repo_of(event).casefold() != wanted:
            continue
        release = _payload_of(event).get("release")
        if not isinstance(release, dict):
            continue
        published_at = _clean(event.get("created_at"))
        if newest is None or published_at > _clean(newest["published_at"]):
            newest = _release_row(
                release, published_at=published_at, actor=_actor_of(event)
            )
    return newest


async def fetch_latest_release(
    repo: str,
    *,
    token: str | None = None,
    session: aiohttp.ClientSession | None = None,
) -> dict[str, Any] | None:
    """Fetch the repository's newest release, for when the event page predates it."""
    name = _clean(repo)
    if "/" not in name:
        return None
    url = f"{GITHUB_API_ROOT}/repos/{_quote_repo(name)}/releases/latest"
    owns_session = session is None
    active_session = session or aiohttp.ClientSession()
    try:
        payload = await _request_json(
            active_session,
            url,
            token=token,
            not_found=f"GitHub repository '{name}' has no releases",
        )
    finally:
        if owns_session:
            await active_session.close()
    if not isinstance(payload, dict):
        return None
    author = payload.get("author")
    login = _clean(author.get("login")) if isinstance(author, dict) else ""
    return _release_row(
        payload, published_at=_clean(payload.get("published_at")), actor=login
    )


def merge_events(*groups: Any) -> list[dict[str, Any]]:
    """Fold event pages into one list, newest first, without duplicate ids."""
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, list):
            continue
        for event in group:
            if not isinstance(event, dict):
                continue
            event_id = _clean(event.get("id"))
            if event_id:
                if event_id in seen:
                    continue
                seen.add(event_id)
            merged.append(event)
    merged.sort(key=lambda event: _clean(event.get("created_at")), reverse=True)
    return merged


async def fetch_activity(
    user: str,
    *,
    highlight: str = GITHUB_DEFAULT_HIGHLIGHT,
    token: str | None = None,
    limit: int = 30,
    session: aiohttp.ClientSession | None = None,
) -> dict[str, Any]:
    """Fetch the account feed plus the highlighted repository feed.

    The account feed decides success: when it fails the dashboard serves its last
    good copy. The repository feed and the release lookup are enhancements, so a
    404 (wrong owner, renamed repository) costs their rows, not the whole panel.
    """
    login = _clean(user)
    if not login:
        raise GitHubActivityError("GitHub user is not configured")
    per_page = max(1, min(limit, GITHUB_MAX_EVENTS))
    repo = repo_full_name(login, highlight)
    owns_session = session is None
    active_session = session or aiohttp.ClientSession()
    try:
        requests = [fetch_user_events(login, token=token, limit=per_page, session=active_session)]
        if repo:
            requests.append(
                fetch_repo_events(repo, token=token, limit=per_page, session=active_session)
            )
        results = await asyncio.gather(*requests, return_exceptions=True)
        account = results[0]
        if isinstance(account, BaseException):
            raise account
        project = results[1] if len(results) > 1 else []
        project_events = project if isinstance(project, list) else []
        merged = merge_events(project_events, account)
        release = latest_release_from_events(merged, repo=repo)
        if release is None and repo:
            try:
                release = await fetch_latest_release(repo, token=token, session=active_session)
            except GitHubActivityError:
                release = None
    finally:
        if owns_session:
            await active_session.close()
    return {"events": merged, "latest_release": release}
