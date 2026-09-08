#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["fastmcp", "httpx"]
# ///
"""CLI to manage users on a remote AutoMol MCP server.

Usage:
  uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> create-user TestDude
  uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> list-users
  uv run mcp/admin_cli.py --url http://127.0.0.1:8001/mcp --token <ADMIN_TOKEN> revoke <OWNER_ID>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


async def call_admin(url: str, token: str, action: str, **kwargs) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    transport = StreamableHttpTransport(url, headers=headers)
    arguments = {"action": action, **{k: v for k, v in kwargs.items() if v is not None}}

    async with Client(transport) as client:
        result = await client.call_tool("admin_manage", arguments)

    if isinstance(result, list):
        for content in result:
            if hasattr(content, "text"):
                return json.loads(content.text)
    elif hasattr(result, "content"):
        for content in result.content:
            if hasattr(content, "text"):
                return json.loads(content.text)
    elif isinstance(result, str):
        return json.loads(result)
    return {}


def main():
    parser = argparse.ArgumentParser(description="AutoMol MCP admin CLI")
    parser.add_argument("--url", required=True, help="MCP server URL (e.g. http://127.0.0.1:8001/mcp)")
    parser.add_argument("--token", required=True, help="Admin token")

    sub = parser.add_subparsers(dest="command", required=True)

    create_p = sub.add_parser("create-user", help="Create a user token")
    create_p.add_argument("user_id", help="Username")

    sub.add_parser("list-users", help="List all users")

    revoke_p = sub.add_parser("revoke", help="Revoke a user by owner_id (see list-users)")
    revoke_p.add_argument("owner_id", help="Unique owner_id of the user to revoke")

    rotate_p = sub.add_parser("rotate", help="Issue a fresh token for a user by owner_id (old token stops working; models/data stay accessible)")
    rotate_p.add_argument("owner_id", help="Unique owner_id of the user to rotate (see list-users)")

    purge_p = sub.add_parser("purge-stale", help="Remove models/datasets not used in N days")
    purge_p.add_argument("--days", type=int, default=30, help="Max age in days (default: 30)")
    purge_p.add_argument("--force", action="store_true", help="Actually delete (default: dry-run)")

    orphan_p = sub.add_parser("purge-orphans", help="Remove unregistered run folders (failed training runs)")
    orphan_p.add_argument("--days", type=int, default=7, help="Max age in days (default: 7)")
    orphan_p.add_argument("--force", action="store_true", help="Actually delete (default: dry-run)")

    args = parser.parse_args()

    try:
        if args.command == "create-user":
            result = asyncio.run(call_admin(args.url, args.token, "create_token", user_id=args.user_id))
            print(f"Created user '{result.get('user_id')}': {result.get('token')}")

        elif args.command == "list-users":
            result = asyncio.run(call_admin(args.url, args.token, "list_users"))
            users = result.get("users", [])
            if not users:
                print("No users.")
            else:
                print(f"{'Owner ID':<24} {'User':<20} {'Status':<10} {'Created':<20} {'Token'}")
                print("-" * 100)
                for u in users:
                    status = "REVOKED" if u["revoked"] else "active"
                    created = u.get("created_at", "")[:19]
                    owner = u.get("owner_id", "")
                    prefix = u.get("token_prefix", "")
                    print(f"{owner:<24} {u['user_id']:<20} {status:<10} {created:<20} {prefix}")

        elif args.command == "revoke":
            result = asyncio.run(call_admin(args.url, args.token, "revoke_user", owner_id=args.owner_id))
            print(f"Revoked owner_id: {result.get('owner_id')}")

        elif args.command == "rotate":
            result = asyncio.run(call_admin(args.url, args.token, "rotate_token", owner_id=args.owner_id))
            print(f"Rotated token for '{result.get('user_id')}' (owner_id: {result.get('owner_id')})")
            print(f"New token: {result.get('token')}")

        elif args.command == "purge-stale":
            result = asyncio.run(call_admin(
                args.url, args.token, "purge_stale",
                max_age_days=args.days, force=args.force,
            ))
            if result.get("dry_run"):
                models = result.get("models_to_purge", [])
                datasets = result.get("datasets_to_purge", [])
                print(f"DRY RUN — would purge {len(models)} model(s), {len(datasets)} dataset(s):")
                if models:
                    print("\n  Models:")
                    for m in models:
                        print(f"    {m['id']} (last_used: {m.get('last_used') or m.get('created_at', 'unknown')})")
                if datasets:
                    print("\n  Datasets:")
                    for d in datasets:
                        print(f"    {d['id']} — {d.get('filename')} (last_used: {d.get('last_used', 'unknown')})")
                if not models and not datasets:
                    print("  Nothing to purge.")
                else:
                    print("\n  Run with --force to execute.")
            else:
                purged_m = result.get("purged_models", [])
                purged_d = result.get("purged_datasets", [])
                errors = result.get("errors", [])
                print(f"Purged {len(purged_m)} model(s), {len(purged_d)} dataset(s).")
                if errors:
                    print("Errors:")
                    for e in errors:
                        print(f"  - {e}")

        elif args.command == "purge-orphans":
            result = asyncio.run(call_admin(
                args.url, args.token, "purge_orphans",
                max_age_days=args.days, force=args.force,
            ))
            if result.get("dry_run"):
                folders = result.get("orphaned_folders", [])
                print(f"DRY RUN — found {len(folders)} orphaned folder(s):")
                for f in folders:
                    print(f"    {f}")
                if not folders:
                    print("  Nothing to purge.")
                else:
                    print("\n  Run with --force to execute.")
            else:
                purged = result.get("purged_folders", [])
                errors = result.get("errors", [])
                print(f"Purged {len(purged)} orphaned folder(s).")
                for f in purged:
                    print(f"    {f}")
                if errors:
                    print("Errors:")
                    for e in errors:
                        print(f"  - {e}")

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
