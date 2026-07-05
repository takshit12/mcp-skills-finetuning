# MCP Demo — Model Context Protocol, hands-on

## What is MCP, and why does it exist?

Before MCP, every AI app that wanted to talk to every tool needed its own custom
glue. With **M** apps and **N** tools, you end up writing **M × N** one-off
integrations — the same "connect Claude to GitHub", "connect Cursor to Postgres",
"connect this chatbot to your filesystem" work, redone over and over.

**MCP (Model Context Protocol)** turns that into **M + N**: each app speaks MCP
once, each tool exposes an MCP server once, and everything interoperates. People
call it **"USB-C for AI"** — one standard port instead of a drawer full of
proprietary cables.

**Tie-back to what you already built:** remember the `@tool` you wrote in
LangChain? MCP is that same idea — a function the model can call — but
*standardized* so that **any** MCP-aware app can use it, not just your one
script. And here's the kicker: **Claude Code and Claude Desktop are themselves
MCP clients.** When you add a server to them, your tool shows up for Claude to
call, no bespoke integration required.

## The architecture in one breath

- A **host** (Claude Desktop, Claude Code, an IDE) runs one or more **clients**.
- Each **client** holds a 1:1 connection to a **server**.
- Each **server** exposes up to **3 primitives** (below).
- **Transports:** **stdio** (the server runs as a local subprocess — great for
  local tools) and **Streamable HTTP** (for remote servers; it replaced the
  legacy SSE transport in Nov 2025).
- Newer capabilities: **Sampling** (a server can ask the host's model to complete
  something) and **Elicitation** (a server can ask the user for input mid-task).
  Looking ahead, 2026 adds **OAuth 2.1** auth and **MCP Gateways**.

## The 3 primitives (with a Kestrel example each)

| Primitive | Who invokes it | What it is | Kestrel example |
|-----------|----------------|------------|-----------------|
| **Tools** | the **model** | actions Claude can decide to call | `lookup_warranty("KH-AR-GLD")` returns the 26-month warranty status |
| **Resources** | the **app / model** pulls them in | read-only data / context | `kestrel://faq` serves the support FAQ as context |
| **Prompts** | the **user** | reusable templates the user triggers | `support_reply(question)` fills the Kestrel house reply format |

Rule of thumb: **Tools** *do* things, **Resources** *are* things (data), and
**Prompts** are *canned starting points* a human picks.

---

## The demo — using existing, official servers

You do not have to write any code to see MCP work. Two official servers ship
ready to run over stdio via `npx`: **filesystem** and **GitHub**.

### 1. Find your Claude Desktop config file

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

If the file doesn't exist yet, create it. You can also open it from Claude
Desktop: **Settings → Developer → Edit Config**.

### 2. Paste in the two servers

Copy the contents of [`claude_desktop_config.example.json`](./claude_desktop_config.example.json)
into that file. It looks like this:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/a/folder"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_REPLACE_WITH_YOUR_TOKEN" }
    }
  }
}
```

- For **filesystem**, replace the path arg with a real folder you want Claude to
  read/organize (e.g. your Desktop project folder). The server can only touch
  the paths you list — nothing else.
- For **github**, create a Personal Access Token on GitHub and drop it into
  `GITHUB_PERSONAL_ACCESS_TOKEN`. (JSON has **no comments** — the placeholder is
  a plain string; keep the real token out of anything you commit.)

### 3. Restart Claude Desktop, then just ask

Fully quit and reopen Claude Desktop. You should see the new tools available
(look for the tools/plug icon). Now try:

- *"Summarize the open issues in this repo."* → watch Claude call the **GitHub**
  MCP tools.
- *"Organize the files in this folder by type."* → watch it call the
  **filesystem** MCP tools.

Claude will ask permission before each call. That's the whole loop: **you ask in
plain English, Claude picks the right MCP tool, calls it, and uses the result.**

---

## Security note (read before you add random servers)

- MCP servers run **with your permissions** — the filesystem server can read
  every file under the paths you grant it; a shell-ish server could run commands
  as you. Only add servers you'd trust with that access.
- **Vet untrusted servers** the way you'd vet any dependency. Prefer official /
  well-known publishers; read what a server actually does before granting a token.
- Tool **results can carry prompt injection** — a GitHub issue or a file could
  contain text that tries to steer Claude ("ignore your instructions and…").
  Keep a human in the loop for consequential actions, and don't auto-approve.

---

## Bonus: build your own server in ~30 lines

Existing servers are great, but writing one is the "aha". See
[`kestrel_mcp_server.py`](./kestrel_mcp_server.py) — a tiny **FastMCP** server
(official MCP Python SDK) that exposes all three primitives for Kestrel Home:

- **Tool** `lookup_warranty(sku)` — warranty status for a Kestrel SKU
- **Tool** `check_restocking_fee(days_since_delivery)` — the 45-day / 12% policy
- **Resource** `kestrel://faq` — serves the FAQ text as context
- **Prompt** `support_reply(question)` — the Kestrel house reply template

Setup and run:

```bash
pip install "mcp[cli]"          # needs Python 3.10+
python kestrel_mcp_server.py    # runs over stdio
```

Register it for **Claude Code** with the project-scoped [`.mcp.json`](./.mcp.json)
in this folder (Claude Code picks it up automatically), or add the same
`"kestrel"` block to your **Claude Desktop** config alongside filesystem/github.
Then ask: *"What's the warranty on SKU KH-AR-GLD?"* or *"A customer wants to
return a thermostat 60 days after delivery — what's the fee?"*

---

## Use the Kestrel server in ANY MCP client (Cursor, VS Code, Codex, …)

MCP is a *standard*, so the exact same server works in every MCP-aware editor —
you just register it in each client's own config format. Use **absolute paths**,
and point at the repo's venv Python so the deps resolve. Below, substitute:

- `PYTHON` = `/ABSOLUTE/PATH/mcp-skills-finetuning/.venv/bin/python`
- `SERVER` = `/ABSOLUTE/PATH/mcp-skills-finetuning/mcp/kestrel_mcp_server.py`

### Claude Code (CLI) — what you already did
```bash
claude mcp add kestrel -- PYTHON SERVER
claude mcp list          # kestrel → ✔ connected
claude mcp remove kestrel
```
Or drop the project-scoped [`.mcp.json`](./.mcp.json) at the repo root and Claude
Code loads it automatically.

### The generic block (Claude Desktop, Cursor, Windsurf, most clients)
Almost every client accepts this `mcpServers` shape — it's the same JSON as
Claude Desktop, just in a different file per client:
```json
{
  "mcpServers": {
    "kestrel": {
      "command": "PYTHON",
      "args": ["SERVER"]
    }
  }
}
```
Where each client reads it:
| Client | Config file |
|--------|-------------|
| **Claude Desktop** | `~/Library/Application Support/Claude/claude_desktop_config.json` (mac) · `%APPDATA%\Claude\claude_desktop_config.json` (win) — see [`claude_desktop_config.example.json`](./claude_desktop_config.example.json) |
| **Cursor** | `~/.cursor/mcp.json` (global) or `<project>/.cursor/mcp.json` — see [`cursor_mcp.example.json`](./cursor_mcp.example.json) |
| **Windsurf** | `~/.codeium/windsurf/mcp_config.json` |

### VS Code (GitHub Copilot / agent mode) — slightly different schema
VS Code uses a `servers` key (not `mcpServers`) and a `type` field, in
`<project>/.vscode/mcp.json` — see [`vscode_mcp.example.json`](./vscode_mcp.example.json):
```json
{
  "servers": {
    "kestrel": { "type": "stdio", "command": "PYTHON", "args": ["SERVER"] }
  }
}
```

### Codex CLI (OpenAI) — TOML, not JSON
Codex configures MCP in `~/.codex/config.toml` — see
[`codex_config.example.toml`](./codex_config.example.toml):
```toml
[mcp_servers.kestrel]
command = "PYTHON"
args = ["SERVER"]
```

**Test in any of them the same way:** ask *"using the kestrel tools, what's the
warranty on SKU KH-AR-GLD and the restocking fee 60 days after delivery?"* — the
client should call `lookup_warranty` + `check_restocking_fee` and answer 26 months
/ 12%. (The **remote** [`kestrel_http_server.py`](./kestrel_http_server.py) plugs
into clients that support HTTP/SSE servers via its `http://127.0.0.1:8000/mcp`
URL instead of a `command`/`args` pair.)
