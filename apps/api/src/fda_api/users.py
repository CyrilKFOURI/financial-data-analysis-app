"""User administration from the command line (first admin, password resets).

    python -m fda_api.users create --email me@example.org --name "Me" --role admin
    python -m fda_api.users password --email me@example.org
    python -m fda_api.users list

The password is prompted (or read from stdin with --password-stdin), never passed as an argument.
The first admin adopts the built-in local user, so everything created in single-user mode
stays theirs.
"""

import argparse
import getpass
import sys

from fastapi import HTTPException
from sqlalchemy import select, update

from fda_api.auth import UserIn, check_password_rules, create_user, hash_password
from fda_data.db import make_engine
from fda_data.schema import users


def _password(stdin: bool) -> str:
    if stdin:
        return sys.stdin.readline().rstrip("\n")
    first = getpass.getpass("Password (12+ characters): ")
    if getpass.getpass("Repeat: ") != first:
        raise SystemExit("Passwords differ")
    return first


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fda_api.users")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create")
    c.add_argument("--email", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--role", choices=["viewer", "analyst", "admin"], default="analyst")
    c.add_argument("--password-stdin", action="store_true")
    p = sub.add_parser("password")
    p.add_argument("--email", required=True)
    p.add_argument("--password-stdin", action="store_true")
    sub.add_parser("list")
    args = parser.parse_args(argv)
    engine = make_engine()
    try:
        if args.command == "list":
            with engine.connect() as conn:
                for row in conn.execute(select(users).order_by(users.c.id)).mappings():
                    state = "active" if row["is_active"] else "inactive"
                    login = "can sign in" if row["password_hash"] else "no password"
                    print(f"{row['id']:>4}  {row['email']:<32} {row['role']:<8} {state}, {login}")
            return 0
        if args.command == "create":
            body = UserIn(
                email=args.email,
                display_name=args.name,
                role=args.role,
                password=_password(args.password_stdin),
            )
            user = create_user(engine, body)
            print(f"Created user {user['id']} {user['email']} ({user['role']})")
            return 0
        with engine.begin() as conn:
            uid = conn.execute(
                select(users.c.id).where(users.c.email == args.email.lower())
            ).scalar()
        if uid is None:
            print(f"No user {args.email}", file=sys.stderr)
            return 1
        password = _password(args.password_stdin)
        check_password_rules(password)
        with engine.begin() as conn:
            conn.execute(
                update(users).where(users.c.id == uid).values(password_hash=hash_password(password))
            )
        print(f"Password updated for {args.email}")
        return 0
    except HTTPException as exc:
        print(exc.detail, file=sys.stderr)
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
