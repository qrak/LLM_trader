# Local settings panel

`http://localhost:8000/admin/` is the operator's desk for the two local files:

- `config/config.ini` — the keys present in the local file, grouped by section, with the file's
  comments as field help. Loader defaults for absent keys are not shown.
- `keys.env` — the optional local secret file. The panel shows whether a key is set and where the
  running process gets it from, **never the value**.

The panel also renders the project documentation from `docs/`, `README.md` and `CONTRIBUTING.md`.
It is not a second console: there are no live logs, approvals, force-analysis or feed controls.

## Access

The desk is **loopback-only**. The page and the documentation are readable, but every read of
`config.ini`/`keys.env` and every write requires:

- the request's client address to be loopback (`127.0.0.0/8`, `::1` or an IPv4-mapped loopback),
- no nonempty `CF-Connecting-IP`; if `X-Forwarded-For` is nonempty, its first address must be local,
- a `Host` with a loopback hostname or `localhost`,
- a local hostname in each nonempty `Origin` and `Referer`.

The panel checks hostnames, not port equality.

There is no login form, session, cookie or admin password. The trust boundary is the machine: anyone
who can already edit those two files can use the desk. A request through `semanticsignal.qrak.org`
gets `403` for the settings API; its documentation view still works.

## Tabs

### Settings

- Lists the sections and keys present in `config/config.ini`, in file order.
- The help text comes from the comment block above the key; inline comments after the value are kept
  and shown separately.
- Values recognized by the editor as booleans are checkboxes, supported enums are selects, and the rest use text fields with
  an appropriate input mode.
- **Search** filters keys by name and help text. **Restart** marks every key that needs a new process.
- `Save` sends only the changed keys. `Discard` restores the last loaded values.

The server validates the whole candidate file with the real loader (`Config(config_path=...)`) before
replacing anything. If a field check or loader validation fails, the server returns `400` and leaves
the live file untouched. Field checks identify the section/key separately; loader errors return the
loader's message.

The write is atomic and keeps the file's comments, order, indentation, inline comments and line
endings. Each load returns an `etag`; a save with a stale etag returns `409` instead of overwriting a
hand edit or another browser tab. Reload the page to get the current file.

### API keys

- Lists the keys from `keys.env.example` (plus any extra keys already present in `keys.env`).
- A badge says `set (keys.env)`, `set (OS env)`, `set (OS env wins)`, `unset (empty OS env)` or
  `not set`.
- The input is always empty: the server never sends a stored value to the browser. `Set` writes a new
  value; `Clear` writes an empty value.
- If the OS environment defines the key, it wins over `keys.env` (including an explicitly empty
  variable). Editing the file will not change what the running bot reads until that environment
  variable is removed.
- If `keys.env` does not exist, `Create keys.env from the template` copies `keys.env.example` once.
  The server attempts to set mode `0600` after creation; a chmod failure does not abort creation.

`keys.env` is gitignored and plain text, exactly like editing it by hand. The server strips
whitespace and rejects newlines/control characters, but only the provider call can tell you whether
a secret is valid. The panel never logs a value; only the key name appears in the bot log.

### Documentation

The allowlisted pages are `docs/README.md`, `docs/SETUP.md`, `docs/CONFIGURATION.md`,
`docs/FEATURES.md`, `docs/STATE_MACHINES.md`, `docs/SETTINGS_PANEL.md`, `README.md` and
`CONTRIBUTING.md`. The same files are the source on GitHub.

Relative links to allowlisted `.md` pages navigate inside the panel. Local images under
`docs/assets/` come from `/docs-assets/`; links to repository files such as `AGENTS.md` or
`CHANGELOG.md` open on GitHub. Raw HTML in the source is
escaped; the rendered page is sanitized before it is inserted. Mermaid blocks show as code in the
panel; GitHub renders them.

## Restart

Nothing in the panel is hot-reloaded. `Config` is read once at import and the services bind their
provider/frame during startup, so every key is `restart`-scoped. After `Save`:

1. Press `Shift+R` in the bot console (the launcher restarts `start.py` in place), or restart the bot
   the way you normally do.
2. Check the startup line `Effective configuration: ...` — that confirms the logged subset of the effective configuration was
   read.
3. `keys.env` is read at startup too; an OS environment variable still wins over it.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `403 Settings are available only from this machine` | The page was opened through the tunnel, LAN IP, or a proxy that forwards the visitor | Open `http://localhost:8000/admin/` on the machine running the bot |
| `409 config/config.ini changed on disk` | The file was edited by hand or in another tab after the page loaded | Reload the page; the unsaved form values are discarded on reload |
| `400 Expected one of: ...` or a loader message | A value the loader refuses | Use the field name or loader message to correct the value; the live file was not written |
| `config/config.ini not found` | The bot was started without copying the template | `cp config/config.ini.example config/config.ini`, then restart |
| Keys tab says `keys.env does not exist` | Only the OS environment is configured | Use `Create keys.env from the template`, or keep using environment variables |
| A key still has the old value after restart | The OS environment defines it and wins | Remove the environment variable (or change it), then restart |
| Documentation does not render | The DOMPurify CDN was blocked | The settings tabs still work; use the GitHub links in the documentation list |

## What this panel does not do

- No hot reload and no restart button. The running process keeps the old values until it restarts.
- No editing of OS environment variables; only `keys.env`.
- No adding INI keys through the form; edit `config/config.ini` first. An API key can be added by the
  panel if it is in `keys.env.example`, or edited if it already exists in `keys.env`.
- No editing of documentation from the browser; `docs/` stays in git.
- No syntax highlighting or Mermaid rendering; both degrade to readable code blocks.
- No live logs, approvals, force analysis, feed toggle or human input — those retired with the old
  admin console.
